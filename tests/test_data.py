from __future__ import annotations

import os

import pytest
import torch
from datasets import IterableDataset

from k3mini.data import (
    DocumentPacker,
    PackedClimbMixDataset,
    SuperBPETokenizer,
    is_validation_document,
    stable_document_hash,
)


class FakeTokenizer:
    eod_token_id = 99
    revision = "fake-v1"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode_batch(self, texts):
        self.seen.extend(texts)
        return [[ord(character) % 50 for character in text] for text in texts]


def _rows():
    for index in range(100):
        yield {
            "text": f"document-{index}-payload",
            "token_count": float(index),
            "cluster_id": index % 7,
        }


def _source():
    return IterableDataset.from_generator(_rows)


def _sharded_source():
    shards = [
        [
            {
                "text": f"{shard}-{index}",
                "token_count": 1.0,
                "cluster_id": shard,
            }
            for index in range(10)
        ]
        for shard in range(4)
    ]

    def generate(shards):
        for shard in shards:
            yield from shard

    return IterableDataset.from_generator(generate, gen_kwargs={"shards": shards})


def test_eod_insertion_and_packing_across_documents() -> None:
    packer = DocumentPacker(sequence_length=4, eod_token_id=99)
    packer.add_encoded([[1, 2], [3, 4, 5]])
    first = packer.pop()
    assert first is not None
    assert first["input_ids"].tolist() == [1, 2, 99, 3]
    assert first["labels"].tolist() == [2, 99, 3, 4]


def test_packer_resume_is_bit_identical() -> None:
    packer = DocumentPacker(sequence_length=3, eod_token_id=99)
    packer.add_encoded([[1, 2, 3], [4, 5], [6, 7, 8]])
    assert packer.pop() is not None
    state = packer.state_dict()
    expected = packer.pop()
    resumed = DocumentPacker(sequence_length=3, eod_token_id=99)
    resumed.load_state_dict(state)
    actual = resumed.pop()
    assert expected is not None and actual is not None
    torch.testing.assert_close(expected["input_ids"], actual["input_ids"])
    torch.testing.assert_close(expected["labels"], actual["labels"])


def test_stream_state_resume_is_bit_identical(tiny_data_config) -> None:
    tiny_data_config.sequence_length = 6
    tiny_data_config.eod_token_id = 99
    tiny_data_config.tokenizer_batch_size = 3
    first = PackedClimbMixDataset(
        tiny_data_config,
        tokenizer=FakeTokenizer(),
        source_factory=_source,
    )
    iterator = iter(first)
    next(iterator)
    state = first.state_dict()
    expected = next(iterator)

    second = PackedClimbMixDataset(
        tiny_data_config,
        tokenizer=FakeTokenizer(),
        source_factory=_source,
    )
    second.load_state_dict(state)
    actual = next(iter(second))
    torch.testing.assert_close(expected["input_ids"], actual["input_ids"])
    torch.testing.assert_close(expected["labels"], actual["labels"])
    assert second.diagnostics() == first.diagnostics()


def test_document_hash_split_is_deterministic_and_disjoint() -> None:
    texts = [f"doc-{index}" for index in range(50_000)]
    first = {text for text in texts if is_validation_document(text)}
    second = {text for text in texts if is_validation_document(text)}
    train = set(texts) - first
    assert first == second
    assert first.isdisjoint(train)
    assert 20 <= len(first) <= 80
    assert stable_document_hash("same") == stable_document_hash("same")


def test_deterministic_rank_shards_are_disjoint(tiny_data_config) -> None:
    tiny_data_config.sequence_length = 3
    tiny_data_config.eod_token_id = 99
    tiny_data_config.tokenizer_batch_size = 2
    tiny_data_config.validation_fraction = 1e-12
    tokenizers = [FakeTokenizer(), FakeTokenizer()]
    datasets = [
        PackedClimbMixDataset(
            tiny_data_config,
            rank=rank,
            world_size=2,
            tokenizer=tokenizers[rank],
            source_factory=_sharded_source,
        )
        for rank in range(2)
    ]
    for dataset in datasets:
        iterator = iter(dataset)
        for _ in range(8):
            next(iterator)
    rank_documents = [set(tokenizer.seen) for tokenizer in tokenizers]
    assert rank_documents[0]
    assert rank_documents[1]
    assert rank_documents[0].isdisjoint(rank_documents[1])

    repeat_tokenizer = FakeTokenizer()
    repeat = PackedClimbMixDataset(
        tiny_data_config,
        rank=0,
        world_size=2,
        tokenizer=repeat_tokenizer,
        source_factory=_sharded_source,
    )
    repeat_iterator = iter(repeat)
    for _ in range(8):
        next(repeat_iterator)
    assert repeat_tokenizer.seen == tokenizers[0].seen


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("K3MINI_RUN_NETWORK_TESTS") != "1",
    reason="set K3MINI_RUN_NETWORK_TESTS=1 to use the Hugging Face tokenizer",
)
def test_superbpe_rust_roundtrip_and_eod() -> None:
    tokenizer = SuperBPETokenizer()
    texts = ["hello world", "naïve café", "def f(x): return x + 1"]
    encoded = tokenizer.encode_batch(texts)
    assert len(encoded) == len(texts)
    for text, token_ids in zip(texts, encoded, strict=True):
        assert tokenizer.decode(token_ids) == text
    assert tokenizer.eod_token_id == 128_000
