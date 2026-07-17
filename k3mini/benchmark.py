from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .config import LinearPrecision, ModelConfig
from .model import K3MiniForCausalLM, LatentMoE


def _timed(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - started) * 1000 / iterations


def benchmark_full_step(
    cfg: ModelConfig,
    *,
    sequence_length: int,
    batch_size: int = 1,
    warmup: int = 2,
    iterations: int = 5,
    trace_path: str | None = None,
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = K3MiniForCausalLM(cfg).to(device).train()
    input_ids = torch.randint(cfg.vocab_size, (batch_size, sequence_length), device=device)
    labels = torch.randint(cfg.vocab_size, (batch_size, sequence_length), device=device)

    def fp8_context():
        if cfg.linear_precision is LinearPrecision.FP8_CURRENT:
            import transformer_engine.pytorch as te

            return te.autocast(enabled=True, recipe=model.fp8_recipe)
        return nullcontext()

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(input_ids, labels, is_first_microbatch=True)
        assert output.loss is not None
        output.loss.backward()

    hidden = torch.randn(
        batch_size,
        sequence_length,
        cfg.d_model,
        device=device,
        requires_grad=True,
    )

    def component_step(module: torch.nn.Module) -> None:
        module.zero_grad(set_to_none=True)
        if hidden.grad is not None:
            hidden.grad = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ), fp8_context():
            component_output = module(hidden)
            if isinstance(component_output, tuple):
                component_output = component_output[0]
            component_output.float().square().mean().backward()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    milliseconds = _timed(step, device=device, warmup=warmup, iterations=iterations)
    components: dict[str, float] = {}
    for layer in model.layers:
        if layer.mixer_kind not in components:
            components[layer.mixer_kind] = _timed(
                lambda layer=layer: component_step(layer.mixer),
                device=device,
                warmup=1,
                iterations=iterations,
            )
        if isinstance(layer.ffn, LatentMoE) and "latent_moe" not in components:
            components["latent_moe"] = _timed(
                lambda layer=layer: component_step(layer.ffn),
                device=device,
                warmup=1,
                iterations=iterations,
            )
    attnres_sources = [hidden.detach().requires_grad_(True) for _ in range(9)]

    def attnres_step() -> None:
        model.final_read.zero_grad(set_to_none=True)
        for source in attnres_sources:
            source.grad = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output, _ = model.final_read(
                attnres_sources,
                output_norm_weight=model.final_norm.weight,
                return_weights=False,
            )
            output.float().square().mean().backward()

    components["attnres_9_sources"] = _timed(attnres_step, device=device, warmup=1, iterations=iterations)
    result = {
        "device": str(device),
        "backend": model.backend.as_dict(),
        "milliseconds_per_step": milliseconds,
        "tokens_per_second": batch_size * sequence_length / (milliseconds / 1000),
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "component_forward_backward_ms": components,
    }
    if trace_path:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            step()
        output_path = Path(trace_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(output_path))
        result["trace"] = str(output_path)
    return result


def save_benchmark(result: dict[str, Any], path: str | None) -> None:
    payload = json.dumps(result, indent=2) + "\n"
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")
