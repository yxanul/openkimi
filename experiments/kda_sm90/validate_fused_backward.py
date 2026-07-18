from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.kda_sm90.fused_wy_intra_triton import (
    chunk_kda_bwd_intra_chunk_blocked,
    install_intra_chunk_experiment,
)
from experiments.kda_sm90.validate_fla_guarded_integration import (
    EXACT_FALLBACK_THRESHOLD,
    GUARDED_THRESHOLD,
    _capture_real_model_kda_inputs,
    _run_real_model_mixer,
    tensor_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/h100-fp8-current-quack-sonic-kda-saved.json"),
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("the fused backward validator requires SM90+")

    capture_loss, model, inputs, output_gradients = _capture_real_model_kda_inputs(
        config=args.config,
        sequence_length=args.sequence_length,
        microbatch=args.microbatch,
    )
    kda_layers = {
        index: layer.mixer
        for index, layer in enumerate(model.layers)
        if hasattr(layer.mixer, "A_log")
    }

    import fla.ops.kda.chunk_bwd as chunk_bwd

    original_intra = install_intra_chunk_experiment(
        "blocked",
        num_warps=args.num_warps,
    )
    chunk_bwd.chunk_kda_bwd_intra = original_intra

    metrics = {}
    for index, mixer in kda_layers.items():
        exact = _run_real_model_mixer(
            mixer,
            inputs[index],
            output_gradients[index],
            threshold=EXACT_FALLBACK_THRESHOLD,
        )
        chunk_bwd.chunk_kda_bwd_intra = chunk_kda_bwd_intra_chunk_blocked
        candidate = _run_real_model_mixer(
            mixer,
            inputs[index],
            output_gradients[index],
            threshold=GUARDED_THRESHOLD,
        )
        chunk_bwd.chunk_kda_bwd_intra = original_intra
        if exact.keys() != candidate.keys():
            raise RuntimeError(f"layer {index} returned different gradient sets")
        for name in exact:
            metrics[f"layers.{index}.mixer.{name}"] = tensor_metrics(
                candidate[name],
                exact[name],
            )

    worst_name, worst = max(
        metrics.items(),
        key=lambda item: item[1].relative_l2,
    )
    failed = [name for name, value in metrics.items() if not value.passed]
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "config": str(args.config),
        "shape": [args.microbatch, args.sequence_length],
        "capture_loss": capture_loss,
        "candidate": "blocked coarsened intra",
        "num_warps": args.num_warps,
        "kda_layer_count": len(kda_layers),
        "compared_tensor_count": len(metrics),
        "maximum_relative_l2": worst.relative_l2,
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
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    del model, inputs, output_gradients
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
