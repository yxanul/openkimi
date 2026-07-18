"""Validate the guarded FLA KDA integration against its exact fallback.

The patched backward-intra kernel accepts its span threshold as a runtime
argument. Setting that threshold below zero forces every diagonal tile through
FLA's original scalar pairwise path. This gives the experiment an exact control
without changing source, recompiling a different kernel, or perturbing FLA's
autotuner choices.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

EXACT_FALLBACK_THRESHOLD = -1.0
GUARDED_THRESHOLD = 232.0
RELATIVE_ERROR_LIMIT = 5e-3
ABSOLUTE_ERROR_LIMIT = 5e-5
HEADS = 6
HEAD_DIM = 128


@dataclass(frozen=True)
class TensorMetrics:
    relative_l2: float
    max_absolute_error: float
    reference_l2: float
    actual_l2: float
    reference_nonfinite: int
    actual_nonfinite: int

    @property
    def passed(self) -> bool:
        return (
            self.reference_nonfinite == 0
            and self.actual_nonfinite == 0
            and (
                self.relative_l2 <= RELATIVE_ERROR_LIMIT
                or self.max_absolute_error <= ABSOLUTE_ERROR_LIMIT
            )
        )


@dataclass(frozen=True)
class KDAFixture:
    name: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    output_gradient: torch.Tensor
    use_gate_in_kernel: bool
    expected_fallback: bool


def tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> TensorMetrics:
    actual_float = actual.float()
    reference_float = reference.float()
    difference = actual_float - reference_float
    denominator = torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
    return TensorMetrics(
        relative_l2=float(torch.linalg.vector_norm(difference) / denominator),
        max_absolute_error=float(difference.abs().max()),
        reference_l2=float(torch.linalg.vector_norm(reference_float)),
        actual_l2=float(torch.linalg.vector_norm(actual_float)),
        reference_nonfinite=int((~torch.isfinite(reference_float)).sum()),
        actual_nonfinite=int((~torch.isfinite(actual_float)).sum()),
    )


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _random_bf16(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )


def _base_fixture(
    name: str,
    *,
    batch: int,
    sequence_length: int,
    seed: int,
    per_token_decay: float,
    expected_fallback: bool,
) -> KDAFixture:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    token_shape = (batch, sequence_length, HEADS, HEAD_DIM)
    q = _random_bf16(token_shape, generator=generator)
    k = _random_bf16(token_shape, generator=generator)
    v = _random_bf16(token_shape, generator=generator)
    gate = torch.full(
        token_shape,
        _inverse_softplus(per_token_decay / 16.0),
        device="cuda",
        dtype=torch.bfloat16,
    )
    beta = _random_bf16(
        (batch, sequence_length, HEADS),
        generator=generator,
    )
    a_log = torch.full(
        (HEADS,),
        math.log(16.0),
        device="cuda",
        dtype=torch.float32,
    )
    dt_bias = torch.zeros(
        HEADS * HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    output_gradient = _random_bf16(token_shape, generator=generator)
    return KDAFixture(
        name=name,
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        output_gradient=output_gradient,
        use_gate_in_kernel=True,
        expected_fallback=expected_fallback,
    )


def make_near_threshold_fixture() -> KDAFixture:
    # A per-token decay of 15.2 produces a 228-wide cumulative span across the
    # 16 rows consumed by one diagonal tile: accepted, but close to 232.
    return _base_fixture(
        "near_threshold_accepted",
        batch=2,
        sequence_length=256,
        seed=2401,
        per_token_decay=15.2,
        expected_fallback=False,
    )


def make_high_decay_fixture() -> KDAFixture:
    # A 480-wide diagonal span is well beyond the guard and must execute the
    # original pairwise scalar path.
    return _base_fixture(
        "high_decay_fallback",
        batch=2,
        sequence_length=256,
        seed=2402,
        per_token_decay=32.0,
        expected_fallback=True,
    )


def make_cancellation_fixture() -> KDAFixture:
    fixture = _base_fixture(
        "cancellation_heavy",
        batch=2,
        sequence_length=1024,
        seed=2403,
        per_token_decay=2.0,
        expected_fallback=False,
    )
    q = fixture.q.clone()
    k = fixture.k.clone()
    v = fixture.v.clone()
    output_gradient = fixture.output_gradient.clone()
    q[..., 1::2] = -q[..., 0::2]
    k[..., 1::2] = k[..., 0::2]
    token_sign = torch.where(
        torch.arange(q.shape[1], device="cuda") % 2 == 0,
        1.0,
        -1.0,
    ).to(torch.bfloat16)
    v.mul_(token_sign[None, :, None, None])
    output_gradient.mul_(-token_sign[None, :, None, None])
    return replace(
        fixture,
        q=q,
        k=k,
        v=v,
        output_gradient=output_gradient,
    )


def make_partial_chunk_fixture() -> KDAFixture:
    # Exercises a final 7-token BT=64 chunk and a partial BC=16 tile.
    return _base_fixture(
        "partial_chunk_4103",
        batch=2,
        sequence_length=4103,
        seed=2404,
        per_token_decay=4.0,
        expected_fallback=False,
    )


def make_nonmonotonic_fixture() -> KDAFixture:
    generator = torch.Generator(device="cuda").manual_seed(2405)
    shape = (2, 256, HEADS, HEAD_DIM)
    q = _random_bf16(shape, generator=generator)
    k = _random_bf16(shape, generator=generator)
    v = _random_bf16(shape, generator=generator)
    # FLA applies the chunk-local cumulative sum even when gate activation is
    # external. One positive increment in every BC=16 tile violates the
    # monotonicity condition while keeping all pairwise exponentials bounded.
    gate = torch.full(shape, -0.25, device="cuda", dtype=torch.bfloat16)
    gate[:, 8::16] = 0.5
    beta = torch.sigmoid(
        _random_bf16((2, 256, HEADS), generator=generator)
    )
    output_gradient = _random_bf16(shape, generator=generator)
    return KDAFixture(
        name="nonmonotonic_fallback",
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        a_log=torch.zeros(HEADS, device="cuda"),
        dt_bias=torch.zeros(HEADS * HEAD_DIM, device="cuda"),
        output_gradient=output_gradient,
        use_gate_in_kernel=False,
        expected_fallback=True,
    )


FIXTURE_FACTORIES = {
    "near_threshold": make_near_threshold_fixture,
    "high_decay": make_high_decay_fixture,
    "cancellation": make_cancellation_fixture,
    "partial_chunk": make_partial_chunk_fixture,
    "nonmonotonic": make_nonmonotonic_fixture,
}


def _set_guard_threshold(value: float) -> None:
    import fla.ops.kda.chunk_intra as chunk_intra

    if not hasattr(chunk_intra, "_OPENKIMI_GUARD_MAX_LOG2_SPAN"):
        raise RuntimeError(
            "the guarded FLA patch is not active; run "
            "experiments/kda_sm90/patch_fla_guarded_intra.py"
        )
    if chunk_intra._OPENKIMI_GUARD_DOT_PRECISION != "tf32x3":
        raise RuntimeError("guarded integration validation requires tf32x3")
    chunk_intra._OPENKIMI_GUARD_MAX_LOG2_SPAN = value


def _run_kda_fixture(
    fixture: KDAFixture,
    *,
    threshold: float,
) -> dict[str, torch.Tensor]:
    from fla.ops.kda import chunk_kda

    _set_guard_threshold(threshold)
    source_values = [
        fixture.q,
        fixture.k,
        fixture.v,
        fixture.gate,
        fixture.beta,
        fixture.a_log,
        fixture.dt_bias,
    ]
    inputs = [
        value.detach().clone().requires_grad_(True)
        for value in source_values
    ]
    q, k, v, gate, beta, a_log, dt_bias = inputs
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        A_log=a_log if fixture.use_gate_in_kernel else None,
        dt_bias=dt_bias if fixture.use_gate_in_kernel else None,
        scale=HEAD_DIM**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=fixture.use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=fixture.use_gate_in_kernel,
        safe_gate=False,
        output_final_state=False,
        state_v_first=True,
        disable_recompute=True,
    )
    output.backward(fixture.output_gradient)
    names = [
        "q_gradient",
        "k_gradient",
        "v_gradient",
        "gate_gradient",
        "beta_gradient",
        "a_log_gradient",
        "dt_bias_gradient",
    ]
    snapshots = {"output": output.detach()}
    for name, value in zip(names, inputs, strict=True):
        if value.grad is not None:
            snapshots[name] = value.grad.detach()
    torch.cuda.synchronize()
    return snapshots


def validate_fixture(
    fixture: KDAFixture,
    *,
    guarded_threshold: float,
) -> dict[str, Any]:
    exact = _run_kda_fixture(
        fixture,
        threshold=EXACT_FALLBACK_THRESHOLD,
    )
    guarded = _run_kda_fixture(
        fixture,
        threshold=guarded_threshold,
    )
    if exact.keys() != guarded.keys():
        raise RuntimeError("exact and guarded paths returned different gradients")
    metrics = {
        name: tensor_metrics(guarded[name], exact[name])
        for name in exact
    }
    passed = all(value.passed for value in metrics.values())
    if fixture.expected_fallback:
        passed = passed and all(
            value.max_absolute_error == 0.0
            for value in metrics.values()
        )
    return {
        "name": fixture.name,
        "shape": list(fixture.q.shape),
        "use_gate_in_kernel": fixture.use_gate_in_kernel,
        "expected_fallback": fixture.expected_fallback,
        "guarded_threshold": guarded_threshold,
        "passed": passed,
        "metrics": {
            name: {
                **asdict(value),
                "passed": value.passed,
            }
            for name, value in metrics.items()
        },
    }


def _capture_real_model_kda_inputs(
    *,
    config: Path,
    sequence_length: int,
    microbatch: int,
) -> tuple[
    float,
    torch.nn.Module,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    from k3mini.config import load_config
    from k3mini.model import K3MiniForCausalLM

    model_config, _, train_config = load_config(config)
    # Hooks must observe one original forward per mixer. At microbatch one this
    # easily fits on H100 without outer checkpoint replay.
    model_config.checkpoint_attention = False
    model_config.checkpoint_ffn = False
    _set_guard_threshold(EXACT_FALLBACK_THRESHOLD)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)
    model = K3MiniForCausalLM(model_config).cuda().train()
    kda_layers = {
        index: layer.mixer
        for index, layer in enumerate(model.layers)
        if hasattr(layer.mixer, "A_log")
    }
    captured_inputs: dict[int, torch.Tensor] = {}
    captured_output_gradients: dict[int, torch.Tensor] = {}
    handles = []

    def capture_hook(index: int):
        def hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured_inputs[index] = inputs[0].detach().clone()

            def capture_gradient(gradient: torch.Tensor) -> None:
                captured_output_gradients[index] = gradient.detach().clone()

            output.register_hook(capture_gradient)

        return hook

    for index, mixer in kda_layers.items():
        handles.append(mixer.register_forward_hook(capture_hook(index)))

    generator = torch.Generator(device="cuda").manual_seed(290719)
    input_ids = torch.randint(
        model_config.vocab_size,
        (microbatch, sequence_length),
        device="cuda",
        generator=generator,
    )
    labels = torch.randint(
        model_config.vocab_size,
        (microbatch, sequence_length),
        device="cuda",
        generator=generator,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels, is_first_microbatch=True)
    if output.loss is None:
        raise RuntimeError("real-model validation requires a training loss")
    output.loss.backward()
    for handle in handles:
        handle.remove()
    if captured_inputs.keys() != kda_layers.keys():
        raise RuntimeError(
            f"missing KDA mixer inputs: {len(captured_inputs)}/{len(kda_layers)}"
        )
    if captured_output_gradients.keys() != kda_layers.keys():
        raise RuntimeError(
            "missing KDA mixer output gradients: "
            f"{len(captured_output_gradients)}/{len(kda_layers)}"
        )
    loss = float(output.loss.detach())
    model.zero_grad(set_to_none=True)
    return loss, model, captured_inputs, captured_output_gradients


def _run_real_model_mixer(
    mixer: torch.nn.Module,
    mixer_input: torch.Tensor,
    output_gradient: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, torch.Tensor]:
    _set_guard_threshold(threshold)
    mixer.zero_grad(set_to_none=True)
    replay_input = mixer_input.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = mixer(replay_input)
    output.backward(output_gradient)
    snapshots = {
        "output": output.detach(),
        "input_gradient": replay_input.grad.detach(),
    }
    for name, parameter in mixer.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing mixer parameter gradient: {name}")
        snapshots[f"parameter.{name}"] = parameter.grad.detach()
    torch.cuda.synchronize()
    return snapshots


def validate_real_model(
    *,
    config: Path,
    sequence_length: int,
    microbatch: int,
    guarded_thresholds: list[float],
) -> dict[str, Any]:
    capture_loss, model, inputs, output_gradients = (
        _capture_real_model_kda_inputs(
            config=config,
            sequence_length=sequence_length,
            microbatch=microbatch,
        )
    )
    kda_layers = {
        index: layer.mixer
        for index, layer in enumerate(model.layers)
        if hasattr(layer.mixer, "A_log")
    }
    metrics_by_threshold: dict[float, dict[str, TensorMetrics]] = {
        threshold: {}
        for threshold in guarded_thresholds
    }
    for index, mixer in kda_layers.items():
        exact = _run_real_model_mixer(
            mixer,
            inputs[index],
            output_gradients[index],
            threshold=EXACT_FALLBACK_THRESHOLD,
        )
        for threshold in guarded_thresholds:
            guarded = _run_real_model_mixer(
                mixer,
                inputs[index],
                output_gradients[index],
                threshold=threshold,
            )
            if exact.keys() != guarded.keys():
                raise RuntimeError(
                    f"layer {index} replays returned different gradient sets"
                )
            for name in exact:
                metrics_by_threshold[threshold][
                    f"layers.{index}.mixer.{name}"
                ] = tensor_metrics(
                    guarded[name],
                    exact[name],
                )
    del model, inputs, output_gradients
    gc.collect()
    torch.cuda.empty_cache()
    threshold_results = {}
    for threshold, metrics in metrics_by_threshold.items():
        worst_name, worst_metrics = max(
            metrics.items(),
            key=lambda item: item[1].relative_l2,
        )
        failed = [
            name
            for name, value in metrics.items()
            if not value.passed
        ]
        threshold_results[str(threshold)] = {
            "maximum_relative_l2": worst_metrics.relative_l2,
            "worst_tensor": worst_name,
            "failed_tensor_count": len(failed),
            "failed_tensors": failed,
            "passed": not failed,
            "metrics": {
                name: {
                    **asdict(value),
                    "passed": value.passed,
                }
                for name, value in metrics.items()
            },
        }
    passing_thresholds = [
        threshold
        for threshold in guarded_thresholds
        if threshold_results[str(threshold)]["passed"]
    ]
    return {
        "config": str(config),
        "shape": [microbatch, sequence_length],
        "capture_loss": capture_loss,
        "capture_threshold": EXACT_FALLBACK_THRESHOLD,
        "kda_layer_count": len(kda_layers),
        "compared_tensor_count": len(next(iter(metrics_by_threshold.values()))),
        "threshold_results": threshold_results,
        "passing_thresholds": passing_thresholds,
        "selected_threshold": max(passing_thresholds)
        if passing_thresholds
        else None,
        "passed": bool(passing_thresholds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=",".join(FIXTURE_FACTORIES),
        help="comma-separated fixture names",
    )
    parser.add_argument(
        "--real-model-config",
        type=Path,
        default=Path("configs/h100-fp8-current-quack-sonic-kda-saved.json"),
    )
    parser.add_argument("--real-model-sequence-length", type=int, default=4096)
    parser.add_argument("--real-model-microbatch", type=int, default=1)
    parser.add_argument(
        "--guarded-thresholds",
        default=str(GUARDED_THRESHOLD),
        help="comma-separated thresholds; real-model mode selects the highest passing value",
    )
    parser.add_argument("--skip-real-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("guarded FLA integration validation requires SM90+")

    selected_cases = [value.strip() for value in args.cases.split(",") if value]
    unknown_cases = sorted(set(selected_cases) - FIXTURE_FACTORIES.keys())
    if unknown_cases:
        raise ValueError(f"unknown cases: {', '.join(unknown_cases)}")
    guarded_thresholds = [
        float(value)
        for value in args.guarded_thresholds.split(",")
        if value.strip()
    ]
    if not guarded_thresholds:
        raise ValueError("at least one guarded threshold is required")
    if selected_cases and len(guarded_thresholds) != 1:
        raise ValueError("fixture validation accepts exactly one guarded threshold")

    fixture_results = []
    for name in selected_cases:
        result = validate_fixture(
            FIXTURE_FACTORIES[name](),
            guarded_threshold=guarded_thresholds[0],
        )
        fixture_results.append(result)
        print(json.dumps({"fixture": result["name"], "passed": result["passed"]}))

    real_model_result = None
    if not args.skip_real_model:
        real_model_result = validate_real_model(
            config=args.real_model_config,
            sequence_length=args.real_model_sequence_length,
            microbatch=args.real_model_microbatch,
            guarded_thresholds=guarded_thresholds,
        )
        print(
            json.dumps(
                {
                    "real_model": real_model_result["config"],
                    "passed": real_model_result["passed"],
                    "selected_threshold": real_model_result[
                        "selected_threshold"
                    ],
                    "passing_thresholds": real_model_result[
                        "passing_thresholds"
                    ],
                }
            )
        )

    passed = all(result["passed"] for result in fixture_results)
    if real_model_result is not None:
        passed = passed and real_model_result["passed"]
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "exact_fallback_threshold": EXACT_FALLBACK_THRESHOLD,
        "guarded_thresholds": guarded_thresholds,
        "dot_precision": "tf32x3",
        "relative_error_limit": RELATIVE_ERROR_LIMIT,
        "absolute_error_limit": ABSOLUTE_ERROR_LIMIT,
        "fixtures": fixture_results,
        "real_model": real_model_result,
        "passed": passed,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if not passed:
        raise SystemExit("guarded FLA integration validation failed")


if __name__ == "__main__":
    main()
