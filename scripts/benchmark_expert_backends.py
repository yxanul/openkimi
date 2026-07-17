from __future__ import annotations

import argparse
import json
import os
import statistics
import types

import torch
import torch.nn as nn

from k3mini.backends import resolve_backend
from k3mini.config import KernelBackend, ModelConfig
from k3mini.model import StackedRoutedExperts


class TransformerEngineExperts(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        os.environ.setdefault("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
        from transformer_engine.pytorch.ops import GroupedLinear, Sequential, SwiGLU

        self.experts = Sequential(
            GroupedLinear(
                cfg.n_routed_experts,
                cfg.latent_dim,
                2 * cfg.expert_ffn_dim,
                bias=False,
                device="cuda",
                dtype=torch.float32,
                single_grouped_weight=True,
            ),
            SwiGLU(),
            GroupedLinear(
                cfg.n_routed_experts,
                cfg.expert_ffn_dim,
                cfg.latent_dim,
                bias=False,
                device="cuda",
                dtype=torch.float32,
                single_grouped_weight=True,
            ),
        )
        self.n_experts = cfg.n_routed_experts

    def forward(
        self,
        latent: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        import transformer_engine.pytorch as te

        counts = torch.bincount(indices.flatten(), minlength=self.n_experts)
        permuted, row_id_map = te.moe_permute(
            latent,
            indices.to(torch.int32),
            num_out_tokens=indices.numel(),
            map_type="index",
        )
        expert_output = self.experts(permuted, counts, counts)
        return te.moe_unpermute(
            expert_output,
            row_id_map,
            merging_probs=weights.float(),
            map_type="index",
        )


def replace_model_experts_with_transformer_engine(model: nn.Module, cfg: ModelConfig) -> None:
    def te_forward(
        module: StackedRoutedExperts,
        latent: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return module.transformer_engine_experts(latent, indices, weights)

    for module in list(model.modules()):
        if not isinstance(module, StackedRoutedExperts):
            continue
        te_experts = TransformerEngineExperts(cfg)
        with torch.no_grad():
            te_experts.experts[0].weight.rowwise_data.view(
                cfg.n_routed_experts,
                2 * cfg.expert_ffn_dim,
                cfg.latent_dim,
            ).copy_(module.gate_up_weight)
            te_experts.experts[2].weight.rowwise_data.view(
                cfg.n_routed_experts,
                cfg.latent_dim,
                cfg.expert_ffn_dim,
            ).copy_(module.down_weight.transpose(1, 2))
        del module.gate_up_weight
        del module.down_weight
        module.transformer_engine_experts = te_experts
        module.forward = types.MethodType(te_forward, module)


def _measure(
    name: str,
    module: nn.Module,
    latent: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float | str]:
    times: list[float] = []
    for iteration in range(warmup + repeats):
        module.zero_grad(set_to_none=True)
        latent.grad = None
        weights.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(latent, indices, weights)
            loss = output.float().square().mean()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        if iteration >= warmup:
            times.append(start.elapsed_time(end))
    return {
        "provider": name,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "output_l2": float(output.float().norm()),
        "latent_grad_l2": float(latent.grad.float().norm()),
        "route_grad_l2": float(weights.grad.float().norm()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare routed-expert training backends.")
    parser.add_argument("--tokens", type=int, default=262_144)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--provider", action="append", choices=("megablocks", "transformer-engine"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    cfg = ModelConfig(kernel_backend=KernelBackend.H100)
    torch.manual_seed(1234)
    latent = torch.randn(
        args.tokens,
        cfg.latent_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    indices = torch.randint(
        cfg.n_routed_experts,
        (args.tokens, cfg.top_k),
        device="cuda",
    )
    weights = torch.rand(
        args.tokens,
        cfg.top_k,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with torch.no_grad():
        weights.div_(weights.sum(-1, keepdim=True))
    providers = args.provider or ["megablocks", "transformer-engine"]

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "tokens": args.tokens,
                "routed_rows": args.tokens * cfg.top_k,
                "experts": cfg.n_routed_experts,
                "shape": f"{cfg.latent_dim}->{2 * cfg.expert_ffn_dim}->{cfg.latent_dim}",
            }
        )
    )
    for provider in providers:
        torch.cuda.reset_peak_memory_stats()
        if provider == "megablocks":
            module = StackedRoutedExperts(
                cfg,
                resolve_backend(KernelBackend.H100),
            ).cuda()
        else:
            module = TransformerEngineExperts(cfg).cuda()
        result = _measure(
            provider,
            module,
            latent,
            indices,
            weights,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(json.dumps(result))
        del module
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
