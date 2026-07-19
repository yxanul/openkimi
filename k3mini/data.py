from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import IterableDataset

from .config import DataConfig


class BatchTokenizer(Protocol):
    eod_token_id: int
    revision: str

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...


class SuperBPETokenizer:
    """SuperBPE loaded directly through Hugging Face's Rust `tokenizers` crate."""

    def __init__(
        self,
        name: str = "alisawuffles/superbpe-tokenizer-128k",
        *,
        revision: str = "main",
        eod_token_id: int = 128_000,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tokenizer_path = hf_hub_download(
            repo_id=name,
            filename="tokenizer.json",
            revision=revision,
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self.name = name
        self.revision = Path(tokenizer_path).parent.name
        self.eod_token_id = eod_token_id
        vocab_size = self._tokenizer.get_vocab_size(with_added_tokens=True)
        if vocab_size != 128_001:
            raise ValueError(f"expected SuperBPE vocabulary size 128001, got {vocab_size}")
        if self._tokenizer.id_to_token(eod_token_id) != "<|endoftext|>":
            raise ValueError(f"token {eod_token_id} is not <|endoftext|>")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [encoding.ids for encoding in self._tokenizer.encode_batch(list(texts))]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


def stable_document_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=b"k3mini-v1").digest()
    return int.from_bytes(digest, "big")


def is_validation_document(text: str, fraction: float = 0.001) -> bool:
    threshold = int(fraction * (1 << 64))
    return stable_document_hash(text) < threshold


class DocumentPacker:
    """Concatenate encoded documents with EOD and emit unpadded shifted chunks."""

    def __init__(self, sequence_length: int, eod_token_id: int) -> None:
        self.sequence_length = sequence_length
        self.eod_token_id = eod_token_id
        self.tokens: deque[int] = deque()

    @property
    def chunk_size(self) -> int:
        return self.sequence_length + 1

    def add_encoded(self, documents: Iterable[Sequence[int]]) -> None:
        for token_ids in documents:
            self.tokens.extend(token_ids)
            self.tokens.append(self.eod_token_id)

    def pop(self) -> dict[str, torch.Tensor] | None:
        if len(self.tokens) < self.chunk_size:
            return None
        chunk = [self.tokens.popleft() for _ in range(self.chunk_size)]
        tensor = torch.tensor(chunk, dtype=torch.long)
        return {"input_ids": tensor[:-1], "labels": tensor[1:]}

    def state_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "eod_token_id": self.eod_token_id,
            "tokens": list(self.tokens),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["sequence_length"] != self.sequence_length:
            raise ValueError("cannot resume with a different sequence length")
        if state["eod_token_id"] != self.eod_token_id:
            raise ValueError("cannot resume with a different EOD token")
        self.tokens = deque(int(token) for token in state["tokens"])


class PackedClimbMixDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stateful ClimbMix streaming, rank sharding, Rust tokenization, and packing."""

    def __init__(
        self,
        cfg: DataConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        validation: bool = False,
        tokenizer: BatchTokenizer | None = None,
        source_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        cfg.validate()
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.validation = validation
        self.tokenizer = tokenizer
        self.source_factory = source_factory
        self.packer = DocumentPacker(cfg.sequence_length, cfg.eod_token_id)
        self.cluster_counts: Counter[int] = Counter()
        self.documents_seen = 0
        self.documents_selected = 0
        self._source: Any = None
        self._source_iterator: Iterator[Mapping[str, Any]] | None = None
        self._pending_state: Mapping[str, Any] | None = None
        self._rng = random.Random(cfg.seed + rank)
        self.resolved_dataset_revision: str | None = None

    def _initialize(self) -> None:
        if self._source is not None:
            return
        if self.source_factory is None:
            from datasets import load_dataset
            from huggingface_hub import HfApi

            source = load_dataset(
                self.cfg.dataset_name,
                self.cfg.dataset_config,
                split=self.cfg.dataset_split,
                revision=self.cfg.dataset_revision,
                streaming=True,
            )
            self.resolved_dataset_revision = (
                HfApi()
                .dataset_info(
                    self.cfg.dataset_name,
                    revision=self.cfg.dataset_revision,
                )
                .sha
            )
        else:
            source = self.source_factory()
            self.resolved_dataset_revision = f"local:{self.cfg.dataset_revision}"
        source = source.shuffle(
            seed=self.cfg.seed,
            buffer_size=self.cfg.shuffle_buffer_size,
        )
        source = source.shard(num_shards=self.world_size, index=self.rank)
        self._source = source
        if self.tokenizer is None:
            self.tokenizer = SuperBPETokenizer(
                self.cfg.tokenizer_name,
                revision=self.cfg.tokenizer_revision,
                eod_token_id=self.cfg.eod_token_id,
            )
        if self._pending_state is not None:
            expected_dataset_revision = self._pending_state.get("resolved_dataset_revision")
            if expected_dataset_revision != self.resolved_dataset_revision:
                raise ValueError(
                    "resolved dataset revision changed since the checkpoint: "
                    f"{expected_dataset_revision!r} != {self.resolved_dataset_revision!r}"
                )
            expected_tokenizer_revision = self._pending_state.get("tokenizer_revision")
            if expected_tokenizer_revision != self.tokenizer.revision:
                raise ValueError(
                    "resolved tokenizer revision changed since the checkpoint: "
                    f"{expected_tokenizer_revision!r} != {self.tokenizer.revision!r}"
                )
            source_state = self._pending_state.get("source")
            if source_state is not None:
                self._source.load_state_dict(source_state)
            self._pending_state = None
        self._source_iterator = iter(self._source)

    def _select(self, text: str) -> bool:
        in_validation = is_validation_document(text, self.cfg.validation_fraction)
        return in_validation if self.validation else not in_validation

    def _fill_tokens(self) -> None:
        assert self._source_iterator is not None
        assert self.tokenizer is not None
        texts: list[str] = []
        clusters: list[int] = []
        while len(texts) < self.cfg.tokenizer_batch_size:
            try:
                row = next(self._source_iterator)
            except StopIteration:
                if texts:
                    break
                raise
            text = row["text"]
            cluster_id = int(row["cluster_id"])
            if not isinstance(text, str):
                raise TypeError(f"ClimbMix `text` must be str, got {type(text).__name__}")
            self.documents_seen += 1
            self.cluster_counts[cluster_id] += 1
            if self._select(text):
                self.documents_selected += 1
                texts.append(text)
                clusters.append(cluster_id)
        del clusters  # Counts are retained; token_count is intentionally never consumed.
        self.packer.add_encoded(self.tokenizer.encode_batch(texts))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        self._initialize()
        while True:
            sample = self.packer.pop()
            if sample is not None:
                yield sample
                continue
            self._fill_tokens()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "documents_seen": self.documents_seen,
            "documents_selected": self.documents_selected,
            "top_clusters": self.cluster_counts.most_common(20),
            "buffered_tokens": len(self.packer.tokens),
        }

    def state_dict(self) -> dict[str, Any]:
        source_state = self._source.state_dict() if self._source is not None else None
        tokenizer_revision = self.tokenizer.revision if self.tokenizer is not None else None
        return {
            "format_version": 1,
            "rank": self.rank,
            "world_size": self.world_size,
            "validation": self.validation,
            "dataset_name": self.cfg.dataset_name,
            "dataset_revision": self.cfg.dataset_revision,
            "resolved_dataset_revision": self.resolved_dataset_revision,
            "tokenizer_name": self.cfg.tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "source": source_state,
            "packer": self.packer.state_dict(),
            "rng_state": self._rng.getstate(),
            "cluster_counts": dict(self.cluster_counts),
            "documents_seen": self.documents_seen,
            "documents_selected": self.documents_selected,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for field, expected in (
            ("rank", self.rank),
            ("world_size", self.world_size),
            ("validation", self.validation),
            ("dataset_name", self.cfg.dataset_name),
            ("dataset_revision", self.cfg.dataset_revision),
        ):
            if state[field] != expected:
                raise ValueError(f"stream checkpoint {field}={state[field]!r} does not match {expected!r}")
        self.packer.load_state_dict(state["packer"])
        self._rng.setstate(state["rng_state"])
        self.cluster_counts = Counter(
            {int(key): int(value) for key, value in state["cluster_counts"].items()}
        )
        self.documents_seen = int(state["documents_seen"])
        self.documents_selected = int(state["documents_selected"])
        self._pending_state = state
        if self._source is not None:
            self._source.load_state_dict(state["source"])
            self._source_iterator = iter(self._source)
            self._pending_state = None


class SyntheticTokenDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(self, vocab_size: int, sequence_length: int, seed: int = 1234) -> None:
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            chunk = torch.randint(
                self.vocab_size,
                (self.sequence_length + 1,),
                generator=self.generator,
            )
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.generator.set_state(state["generator_state"])

    def diagnostics(self) -> dict[str, Any]:
        return {"kind": "synthetic"}


def materialize_validation_cache(
    dataset: PackedClimbMixDataset,
    path: str | Path,
    target_tokens: int,
) -> dict[str, int]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, torch.Tensor]] = []
    consumed = 0
    for sample in dataset:
        samples.append(sample)
        consumed += sample["input_ids"].numel()
        if consumed >= target_tokens:
            break
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "sequence_length": dataset.cfg.sequence_length,
            "tokens": consumed,
            "samples": samples,
            "stream_diagnostics": dataset.diagnostics(),
        },
        temporary,
    )
    temporary.replace(output_path)
    return {"samples": len(samples), "tokens": consumed}


def load_validation_cache(path: str | Path) -> list[dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported validation cache format")
    return payload["samples"]
