from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from k3mini.config import load_config
from k3mini.model import K3MiniForCausalLM, KimiDeltaAttention

CHUNK_SIZE = 64
BLOCK_SIZE = 16
CHANNEL_TILE = 32
RCP_LN2 = 1.0 / math.log(2.0)


def gate_tile_spans(
    raw_decay: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Return the log2 span of each FLA `[BC=16, BK=32]` gate tile."""

    batch, sequence_length, heads, head_dim = raw_decay.shape
    if sequence_length % CHUNK_SIZE:
        raise ValueError("sequence length must be divisible by the FLA chunk size")
    if CHUNK_SIZE % BLOCK_SIZE or head_dim % CHANNEL_TILE:
        raise ValueError("KDA dimensions are incompatible with the SM90 tile")
    gate_steps = (
        -a_log.float().exp().view(1, 1, heads, 1)
        * F.softplus(raw_decay.float() + dt_bias.float().view(1, 1, heads, head_dim))
        * RCP_LN2
    )
    chunks = gate_steps.view(
        batch,
        sequence_length // CHUNK_SIZE,
        CHUNK_SIZE,
        heads,
        head_dim,
    ).cumsum(dim=2)
    tiles = chunks.view(
        batch,
        sequence_length // CHUNK_SIZE,
        CHUNK_SIZE // BLOCK_SIZE,
        BLOCK_SIZE,
        heads,
        head_dim // CHANNEL_TILE,
        CHANNEL_TILE,
    )
    return (tiles[:, :, :, :1] - tiles).amax(dim=(3, 6)).flatten()


def _summary(spans: torch.Tensor, thresholds: list[float]) -> dict[str, object]:
    quantile_levels = torch.tensor(
        [0.0, 0.5, 0.9, 0.99, 0.999, 1.0],
        dtype=torch.float32,
    )
    quantiles = torch.quantile(spans.float(), quantile_levels)
    return {
        "tiles": spans.numel(),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("minimum", "p50", "p90", "p99", "p99_9", "maximum"),
                quantiles,
                strict=True,
            )
        },
        "guard_hit_rate": {
            str(threshold): float((spans <= threshold).float().mean())
            for threshold in thresholds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture real initialized-model KDA gate spans for the guarded SM90 path."
    )
    parser.add_argument(
        "--config",
        default="configs/h100-fp8-current-quack-sonic-kda-saved.json",
    )
    parser.add_argument("--warmup-forwards", type=int, default=1)
    parser.add_argument("--microbatch-sequences", type=int)
    parser.add_argument("--thresholds", default="4,8,12,16,20,24,30")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if args.warmup_forwards < 0:
        raise ValueError("warmup-forwards cannot be negative")
    if not thresholds or any(not 0.0 < value < 252.0 for value in thresholds):
        raise ValueError("thresholds must be finite values between 0 and 252")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("this profiler requires an SM90+ CUDA GPU")

    model_cfg, data_cfg, train_cfg = load_config(args.config)
    microbatch_sequences = args.microbatch_sequences or train_cfg.microbatch_sequences
    torch.manual_seed(train_cfg.seed)
    torch.cuda.manual_seed_all(train_cfg.seed)
    model = K3MiniForCausalLM(model_cfg).cuda().eval()
    input_ids = torch.randint(
        model_cfg.vocab_size,
        (microbatch_sequences, data_cfg.sequence_length),
        device="cuda",
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup_forwards):
            model(input_ids, is_first_microbatch=True)
        torch.cuda.synchronize()

    captures: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.layers):
        if not isinstance(layer.mixer, KimiDeltaAttention):
            continue
        mixer = layer.mixer

        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            layer_index: int = layer_index,
            mixer: KimiDeltaAttention = mixer,
        ) -> None:
            raw_decay = output.detach().view(
                output.shape[0],
                output.shape[1],
                mixer.n_heads,
                mixer.head_dim,
            )
            captures[layer_index] = gate_tile_spans(
                raw_decay,
                mixer.A_log,
                mixer.dt_bias,
            ).cpu()

        handles.append(mixer.f_b_proj.register_forward_hook(capture))

    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids, is_first_microbatch=False)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    expected_layers = sum(
        isinstance(layer.mixer, KimiDeltaAttention) for layer in model.layers
    )
    if len(captures) != expected_layers:
        raise RuntimeError(f"captured {len(captures)} of {expected_layers} KDA layers")
    all_spans = torch.cat([captures[index] for index in sorted(captures)])
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "config": args.config,
        "seed": train_cfg.seed,
        "microbatch_sequences": microbatch_sequences,
        "sequence_length": data_cfg.sequence_length,
        "kda_layers": expected_layers,
        "tile": {
            "chunk_size": CHUNK_SIZE,
            "block_size": BLOCK_SIZE,
            "channel_tile": CHANNEL_TILE,
        },
        "aggregate": _summary(all_spans, thresholds),
        "layers": {
            str(index): _summary(captures[index], thresholds)
            for index in sorted(captures)
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
