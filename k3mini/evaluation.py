from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

from .config import DataConfig, TrainConfig
from .data import SuperBPETokenizer


@contextlib.contextmanager
def _preserve_training_mode(model: torch.nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


class OpenKimiEvalLM:
    """Factory namespace; the concrete lm-eval base is imported only on demand."""

    @staticmethod
    def create(
        model: torch.nn.Module,
        tokenizer: SuperBPETokenizer,
        *,
        device: torch.device,
        precision: str,
        max_length: int,
        batch_size: int,
    ) -> Any:
        try:
            from lm_eval import utils
            from lm_eval.api.model import TemplateLM
        except ImportError as error:
            raise RuntimeError(
                "periodic benchmark evaluation requires `uv sync --extra eval`"
            ) from error

        class _Adapter(TemplateLM):
            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.tokenizer = tokenizer
                self._device = device
                self._max_length = max_length
                self.batch_size = batch_size
                self.precision = precision
                self.max_gen_toks = 64

            @property
            def device(self) -> torch.device:
                return self._device

            @property
            def eot_token_id(self) -> int:
                return tokenizer.eod_token_id

            @property
            def max_length(self) -> int:
                return self._max_length

            @property
            def rank(self) -> int:
                return 0

            @property
            def world_size(self) -> int:
                return 1

            def tok_encode(
                self,
                string: str,
                add_special_tokens: bool | None = None,
                **kwargs: Any,
            ) -> list[int]:
                del add_special_tokens, kwargs
                return tokenizer.encode(string)

            def tok_decode(self, tokens: list[int]) -> str:
                return tokenizer.decode(tokens)

            def _autocast(self):
                return torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16" and device.type == "cuda",
                )

            @torch.inference_mode()
            def _loglikelihood_tokens(
                self,
                requests: list[
                    tuple[tuple[str, str] | None, list[int], list[int]]
                ],
                **kwargs: Any,
            ) -> list[tuple[float, bool]]:
                del kwargs
                results: list[tuple[float, bool]] = []
                for start in range(0, len(requests), self.batch_size):
                    chunk = requests[start : start + self.batch_size]
                    rows: list[list[int]] = []
                    continuation_lengths: list[int] = []
                    input_lengths: list[int] = []
                    for _, context_tokens, continuation_tokens in chunk:
                        if not context_tokens or not continuation_tokens:
                            raise ValueError(
                                "lm-eval requests require non-empty context and continuation"
                            )
                        combined = context_tokens + continuation_tokens
                        combined = combined[-(self.max_length + 1) :]
                        continuation_length = min(
                            len(continuation_tokens),
                            len(combined) - 1,
                        )
                        rows.append(combined)
                        continuation_lengths.append(continuation_length)
                        input_lengths.append(len(combined) - 1)
                    padded_length = max(input_lengths)
                    input_ids = torch.full(
                        (len(rows), padded_length),
                        self.eot_token_id,
                        device=device,
                        dtype=torch.long,
                    )
                    for row_idx, row in enumerate(rows):
                        input_ids[row_idx, : len(row) - 1] = torch.tensor(
                            row[:-1],
                            device=device,
                        )
                    with self._autocast():
                        output = self.model(input_ids, return_logits=True)
                    assert output.logits is not None
                    for row_idx, row in enumerate(rows):
                        continuation_length = continuation_lengths[row_idx]
                        input_length = input_lengths[row_idx]
                        selected = output.logits[
                            row_idx,
                            input_length - continuation_length : input_length,
                        ].float()
                        targets = torch.tensor(
                            row[-continuation_length:],
                            device=device,
                            dtype=torch.long,
                        )
                        log_probabilities = torch.log_softmax(selected, dim=-1)
                        score = log_probabilities.gather(
                            -1,
                            targets[:, None],
                        ).sum()
                        greedy = bool(torch.equal(selected.argmax(-1), targets))
                        results.append((float(score.item()), greedy))
                    del output
                return results

            def loglikelihood_rolling(
                self,
                requests: list[Any],
                disable_tqdm: bool = False,
            ) -> list[float]:
                del disable_tqdm
                results: list[float] = []
                for request in requests:
                    (text,) = request.args
                    windows = [
                        (None, context, continuation)
                        for context, continuation in map(
                            utils.make_disjoint_window,
                            utils.get_rolling_token_windows(
                                token_list=self.tok_encode(text),
                                prefix_token=self.eot_token_id,
                                max_seq_len=self.max_length,
                                context_len=1,
                            ),
                        )
                    ]
                    results.append(
                        sum(score for score, _ in self._loglikelihood_tokens(windows))
                    )
                return results

            @torch.inference_mode()
            def generate_until(
                self,
                requests: list[Any],
                disable_tqdm: bool = False,
            ) -> list[str]:
                del disable_tqdm
                generated: list[str] = []
                for request in requests:
                    context, generation_kwargs = request.args
                    until = generation_kwargs.get("until", [])
                    if isinstance(until, str):
                        until = [until]
                    max_new_tokens = int(
                        generation_kwargs.get("max_gen_toks", self.max_gen_toks)
                    )
                    tokens = self.tok_encode(context)[
                        -(self.max_length - max_new_tokens) :
                    ]
                    prefix_length = len(tokens)
                    for _ in range(max_new_tokens):
                        input_ids = torch.tensor(
                            tokens[-self.max_length :],
                            device=device,
                            dtype=torch.long,
                        )[None]
                        with self._autocast():
                            output = self.model(input_ids, return_logits=True)
                        assert output.logits is not None
                        next_token = int(output.logits[0, -1].argmax().item())
                        tokens.append(next_token)
                        text = self.tok_decode(tokens[prefix_length:])
                        if next_token == self.eot_token_id or any(
                            stop and stop in text for stop in until
                        ):
                            break
                    text = self.tok_decode(tokens[prefix_length:])
                    for stop in until:
                        if stop and stop in text:
                            text = text.split(stop, 1)[0]
                    generated.append(text)
                return generated

        return _Adapter()


def run_lm_evaluation(
    model: torch.nn.Module,
    data_config: DataConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    if not train_config.eval_tasks:
        return {}
    try:
        import lm_eval
    except ImportError as error:
        raise RuntimeError(
            "periodic benchmark evaluation requires `uv sync --extra eval`"
        ) from error

    tokenizer = SuperBPETokenizer(
        data_config.tokenizer_name,
        revision=data_config.tokenizer_revision,
        eod_token_id=data_config.eod_token_id,
    )
    adapter = OpenKimiEvalLM.create(
        model,
        tokenizer,
        device=device,
        precision=train_config.precision,
        max_length=data_config.sequence_length,
        batch_size=train_config.eval_batch_size,
    )
    with _preserve_training_mode(model):
        result = lm_eval.simple_evaluate(
            model=adapter,
            tasks=list(train_config.eval_tasks),
            num_fewshot=train_config.eval_num_fewshot,
            batch_size=train_config.eval_batch_size,
            limit=train_config.eval_limit,
            bootstrap_iters=train_config.eval_bootstrap_iters,
            use_cache=train_config.eval_cache,
            log_samples=False,
        )
    if result is None:
        return {}
    flattened: dict[str, float] = {}
    for task, metrics in result.get("results", {}).items():
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not name.endswith("_stderr"):
                flattened[f"{task}/{name}"] = float(value)
    return flattened
