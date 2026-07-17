from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .backends import BackendStatus, resolve_backend
from .config import KernelBackend, LossBackend, ModelConfig, RouterType


@torch.compiler.disable
def _run_external_fused_loss(
    loss_fn: nn.Module,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return loss_fn(weight, hidden.flatten(0, 1), labels.flatten())


@dataclass(slots=True)
class ModelOutput:
    loss: torch.Tensor | None
    lm_loss: torch.Tensor | None
    router_aux_loss: torch.Tensor
    router_z_loss: torch.Tensor
    logits: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouterOutput:
    indices: torch.Tensor
    weights: torch.Tensor
    auxiliary_loss: torch.Tensor
    z_loss: torch.Tensor
    load: torch.Tensor
    entropy: torch.Tensor
    max_load_violation: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * weight.float()).to(dtype)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.channels:
            raise ValueError(f"expected last dimension {self.channels}, got {x.shape[-1]}")
        xt = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return F.silu(F.conv1d(xt, self.weight, groups=self.channels).transpose(1, 2))


class HeadwiseRMSNormGate(nn.Module):
    def __init__(self, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps) * gate_logits.sigmoid()


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Exact FP32-state KDA recurrence used as the numerical oracle."""
    if not (q.shape == k.shape == v.shape == log_alpha.shape):
        raise ValueError("q, k, v, and log_alpha must have identical shapes")
    if beta.shape != q.shape[:-1]:
        raise ValueError("beta must have shape [B,T,H]")
    batch, time, heads, dim = q.shape
    state = torch.zeros(batch, heads, dim, dim, device=q.device, dtype=torch.float32)
    outputs: list[torch.Tensor] = []
    qf, kf, vf = q.float(), k.float(), v.float()
    alpha, betaf = log_alpha.float().exp(), beta.float()
    scale = dim**-0.5 if scale is None else scale
    for token_idx in range(time):
        decayed = state * alpha[:, token_idx].unsqueeze(-1)
        prediction = torch.einsum("bhd,bhdv->bhv", kf[:, token_idx], decayed)
        error = vf[:, token_idx] - prediction
        state = decayed + kf[:, token_idx].unsqueeze(-1) * (
            betaf[:, token_idx].unsqueeze(-1) * error
        ).unsqueeze(-2)
        outputs.append(torch.einsum("bhdv,bhd->bhv", state, qf[:, token_idx]) * scale)
    return torch.stack(outputs, dim=1).to(q.dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: BackendStatus) -> None:
        super().__init__()
        self.backend = backend
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.kda_head_dim
        projection_size = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.d_model, projection_size, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, projection_size, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, projection_size, bias=False)

        self._fla_conv = backend.selected is KernelBackend.H100
        if self._fla_conv:
            from fla.modules import ShortConvolution

            conv = lambda: ShortConvolution(  # noqa: E731
                hidden_size=projection_size,
                kernel_size=cfg.kda_conv_kernel,
                bias=False,
                activation="silu",
            )
            self.q_conv, self.k_conv, self.v_conv = conv(), conv(), conv()
        else:
            self.q_conv = CausalDepthwiseConv1d(projection_size, cfg.kda_conv_kernel)
            self.k_conv = CausalDepthwiseConv1d(projection_size, cfg.kda_conv_kernel)
            self.v_conv = CausalDepthwiseConv1d(projection_size, cfg.kda_conv_kernel)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1.0, 16.0)))
        self.f_a_proj = nn.Linear(cfg.d_model, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads, self.head_dim, dtype=torch.float32))
        self.beta_proj = nn.Linear(cfg.d_model, self.n_heads, bias=False)
        self.out_gate_a = nn.Linear(cfg.d_model, self.head_dim, bias=False)
        self.out_gate_b = nn.Linear(self.head_dim, projection_size, bias=False)
        if backend.selected is KernelBackend.H100:
            from fla.modules import FusedRMSNormGated

            self.out_norm_gate = FusedRMSNormGated(
                self.head_dim,
                activation="sigmoid",
                eps=cfg.rms_norm_eps,
            )
        else:
            self.out_norm_gate = HeadwiseRMSNormGate(self.head_dim, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(projection_size, cfg.d_model, bias=False)
        self.scale = self.head_dim**-0.5

    def _conv(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._fla_conv:
            output, _ = layer(x)
            return output
        return layer(x)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:2], self.n_heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self._conv(self.q_conv, self.q_proj(x)))
        k = self._split_heads(self._conv(self.k_conv, self.k_proj(x)))
        v = self._split_heads(self._conv(self.v_conv, self.v_proj(x)))
        raw_decay = self._split_heads(self.f_b_proj(self.f_a_proj(x)))
        beta_logits = self.beta_proj(x)

        if self.backend.selected is KernelBackend.H100:
            from fla.ops.kda import chunk_kda

            out, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=raw_decay,
                beta=beta_logits,
                A_log=self.A_log,
                dt_bias=self.dt_bias.reshape(-1),
                scale=self.scale,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=False,
                output_final_state=False,
                state_v_first=True,
            )
        else:
            q = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6).to(x.dtype)
            k = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6).to(x.dtype)
            log_alpha = -self.A_log.float().exp().view(1, 1, self.n_heads, 1) * F.softplus(
                raw_decay.float() + self.dt_bias.view(1, 1, self.n_heads, self.head_dim)
            )
            out = kda_recurrent_reference(q, k, v, log_alpha, beta_logits.float().sigmoid(), scale=self.scale)

        output_gate = self._split_heads(self.out_gate_b(self.out_gate_a(x)))
        out = self.out_norm_gate(out, output_gate).flatten(2)
        return self.out_proj(out)


class NoPELatentAttention(nn.Module):
    """Published ungated NoPE MLA; the unreleased K3 gate remains out of scope."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qk_head_dim = cfg.mla_qk_head_dim
        self.v_head_dim = cfg.mla_v_head_dim
        self.kv_lora_rank = cfg.mla_kv_lora_rank
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj = nn.Linear(cfg.d_model, self.kv_lora_rank, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_head_dim + self.v_head_dim),
            bias=False,
        )
        self.out_proj = nn.Linear(self.n_heads * self.v_head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.n_heads, self.qk_head_dim).transpose(1, 2)
        compressed = self.kv_a_norm(self.kv_a_proj(x))
        kv = self.kv_b_proj(compressed).view(batch, time, self.n_heads, self.qk_head_dim + self.v_head_dim)
        k, v = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)
        output = F.scaled_dot_product_attention(
            q,
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0,
            scale=self.qk_head_dim**-0.5,
        )
        return self.out_proj(output.transpose(1, 2).contiguous().flatten(2))


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _router_diagnostics(
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    experts: int,
    load: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if load is None:
        load = torch.bincount(indices.flatten(), minlength=experts).to(torch.float32)
    mean = load.mean().clamp_min(1.0)
    max_violation = (load.max() - mean) / mean
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
    return load, entropy, max_violation


def _expert_load(
    indices: torch.Tensor,
    experts: int,
    backend: BackendStatus | None,
) -> torch.Tensor:
    if backend is not None and backend.selected is KernelBackend.H100:
        from megablocks import ops as mb_ops

        return mb_ops.histogram(indices.flatten().to(torch.int32), experts).to(torch.float32)
    return torch.bincount(indices.flatten(), minlength=experts).to(torch.float32)


class SoftmaxTopKRouter(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: BackendStatus | None = None) -> None:
        super().__init__()
        self.backend = backend
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.weight = nn.Parameter(torch.empty(self.n_experts, cfg.d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("last_load", torch.zeros(self.n_experts))
        self.register_buffer("last_entropy", torch.zeros(()))
        self.register_buffer("last_max_load_violation", torch.zeros(()))
        self.register_buffer("consecutive_dead_steps", torch.zeros(self.n_experts, dtype=torch.long))

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = True) -> RouterOutput:
        logits = F.linear(x.float(), self.weight.float())
        probabilities = logits.softmax(dim=-1)
        selected_probabilities, indices = probabilities.topk(self.top_k, dim=-1, sorted=False)
        weights = selected_probabilities / selected_probabilities.sum(-1, keepdim=True).clamp_min(1e-9)
        load = _expert_load(indices, self.n_experts, self.backend)
        if collect_diagnostics:
            _, entropy, max_violation = _router_diagnostics(
                probabilities,
                indices,
                self.n_experts,
                load,
            )
        else:
            entropy = logits.new_zeros(())
            max_violation = logits.new_zeros(())
        with torch.no_grad():
            self.last_load.copy_(load)
            if collect_diagnostics:
                self.last_entropy.copy_(entropy)
                self.last_max_load_violation.copy_(max_violation)
        load_fraction = load / load.sum().clamp_min(1.0)
        auxiliary_loss = self.n_experts * torch.sum(probabilities.mean(0) * load_fraction)
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        return RouterOutput(
            indices=indices,
            weights=weights.to(x.dtype),
            auxiliary_loss=auxiliary_loss,
            z_loss=z_loss,
            load=load.detach(),
            entropy=entropy.detach(),
            max_load_violation=max_violation.detach(),
        )


class SigmoidNoAuxTopKRouter(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: BackendStatus | None = None) -> None:
        super().__init__()
        self.backend = backend
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.scale = cfg.router_scale
        self.update_rate = cfg.router_bias_update_rate
        self.n_groups = cfg.router_num_groups
        self.topk_groups = cfg.router_topk_groups
        self.experts_per_group = self.n_experts // self.n_groups
        self.weight = nn.Parameter(torch.empty(self.n_experts, cfg.d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_experts))
        self.register_buffer("pending_load", torch.zeros(self.n_experts, dtype=torch.float64))
        self.register_buffer("last_load", torch.zeros(self.n_experts, dtype=torch.float64))
        self.register_buffer("last_entropy", torch.zeros(()))
        self.register_buffer("last_max_load_violation", torch.zeros(()))
        self.register_buffer("consecutive_dead_steps", torch.zeros(self.n_experts, dtype=torch.long))

    def _group_mask(self, scores: torch.Tensor) -> torch.Tensor:
        if self.n_groups == 1:
            return scores
        grouped = scores.view(-1, self.n_groups, self.experts_per_group)
        group_scores = grouped.topk(min(2, self.experts_per_group), dim=-1).values.sum(-1)
        chosen = group_scores.topk(self.topk_groups, dim=-1, sorted=False).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool)
        mask.scatter_(1, chosen, True)
        return scores.masked_fill(~mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores), -torch.inf)

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = True) -> RouterOutput:
        logits = F.linear(x.float(), self.weight.float())
        probabilities = logits.sigmoid()
        selection = self._group_mask(probabilities + self.e_score_correction_bias.float().unsqueeze(0))
        indices = selection.topk(self.top_k, dim=-1, sorted=False).indices
        weights = probabilities.gather(1, indices)
        weights = self.scale * weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        load = _expert_load(indices, self.n_experts, self.backend)
        if collect_diagnostics:
            _, entropy, max_violation = _router_diagnostics(
                probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-9),
                indices,
                self.n_experts,
                load,
            )
        else:
            entropy = logits.new_zeros(())
            max_violation = logits.new_zeros(())
        with torch.no_grad():
            self.pending_load.add_(load.to(self.pending_load))
            if collect_diagnostics:
                self.last_entropy.copy_(entropy)
                self.last_max_load_violation.copy_(max_violation)
        zero = logits.new_zeros(())
        return RouterOutput(
            indices=indices,
            weights=weights.to(x.dtype),
            auxiliary_loss=zero,
            z_loss=zero,
            load=load.detach(),
            entropy=entropy.detach(),
            max_load_violation=max_violation.detach(),
        )

    @torch.no_grad()
    def update_bias(self) -> None:
        load = self.pending_load.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        self.last_load.copy_(load)
        if load.sum() > 0:
            self.e_score_correction_bias.add_(
                self.update_rate * torch.sign(load.mean() - load).to(self.e_score_correction_bias)
            )
        self.pending_load.zero_()


class StackedRoutedExperts(nn.Module):
    """One parameter layout shared by the reference and grouped-GEMM paths."""

    def __init__(self, cfg: ModelConfig, backend: BackendStatus) -> None:
        super().__init__()
        self.backend = backend
        shape_in = (cfg.n_routed_experts, 2 * cfg.expert_ffn_dim, cfg.latent_dim)
        shape_out = (cfg.n_routed_experts, cfg.expert_ffn_dim, cfg.latent_dim)
        self.gate_up_weight = nn.Parameter(torch.empty(shape_in))
        self.down_weight = nn.Parameter(torch.empty(shape_out))
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.sort_end_bit = max(math.ceil(math.log2(self.n_experts)), 1)
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def _reference(self, latent: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(latent)
        for expert_id in range(self.n_experts):
            token_pos, slot_pos = torch.where(indices == expert_id)
            if token_pos.numel() == 0:
                continue
            expert_input = latent.index_select(0, token_pos)
            gate_up_weight = self.gate_up_weight[expert_id].to(expert_input.dtype)
            down_weight = self.down_weight[expert_id].to(expert_input.dtype)
            gate, up = F.linear(expert_input, gate_up_weight).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            hidden = hidden * weights[token_pos, slot_pos].unsqueeze(-1)
            expert_output = F.linear(hidden, down_weight.transpose(0, 1))
            output.index_add_(0, token_pos, expert_output)
        return output

    def _grouped(self, latent: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        from megablocks import grouped_gemm_util as gg
        from megablocks import ops as mb_ops

        if gg.ops is None:
            raise RuntimeError("grouped_gemm is unavailable despite selecting the H100 backend")
        flat_experts = indices.flatten().to(torch.int32)
        with torch.no_grad():
            bin_ids, permutation = mb_ops.sort(flat_experts, self.sort_end_bit)
            token_counts = mb_ops.histogram(flat_experts, self.n_experts)
            bins = mb_ops.inclusive_cumsum(token_counts, 0)
            tokens_per_expert = token_counts.to(torch.long)
        sorted_input = mb_ops.gather(
            latent,
            permutation,
            bin_ids,
            bins,
            self.top_k,
        ).contiguous()
        gate_up = gg.ops.gmm(
            sorted_input,
            self.gate_up_weight.to(sorted_input.dtype).contiguous(),
            tokens_per_expert,
            trans_b=True,
        )
        from .cuda_kernels import fused_weighted_swiglu

        sorted_weights = weights.flatten().index_select(0, permutation)
        hidden = fused_weighted_swiglu(gate_up, sorted_weights)
        expert_output = gg.ops.gmm(
            hidden,
            self.down_weight.to(sorted_input.dtype).contiguous(),
            tokens_per_expert,
        )
        return mb_ops.scatter(
            expert_output,
            permutation,
            bin_ids,
            None,
            bins,
            self.top_k,
        )

    def forward(self, latent: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.backend.selected is KernelBackend.H100:
            return self._grouped(latent, indices, weights)
        return self._reference(latent, indices, weights)


class LatentMoE(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: BackendStatus) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.router: SoftmaxTopKRouter | SigmoidNoAuxTopKRouter
        if cfg.router_type is RouterType.SOFTMAX:
            self.router = SoftmaxTopKRouter(cfg, backend)
        else:
            self.router = SigmoidNoAuxTopKRouter(cfg, backend)
        self.down_proj = nn.Linear(cfg.d_model, cfg.latent_dim, bias=False)
        self.up_proj = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.routed_experts = StackedRoutedExperts(cfg, backend)
        self.shared_experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.shared_ffn_dim, cfg.d_model) for _ in range(cfg.n_shared_experts)]
        )

    def forward(
        self, x: torch.Tensor, *, diagnostics: bool = False
    ) -> tuple[torch.Tensor, RouterOutput]:
        flat = x.flatten(0, 1)
        routing = self.router(flat, collect_diagnostics=diagnostics)
        latent = self.down_proj(flat)
        routed = self.up_proj(
            self.routed_experts(
                latent,
                routing.indices,
                routing.weights.to(latent.dtype),
            )
        )
        shared = sum((expert(flat) for expert in self.shared_experts), torch.zeros_like(flat))
        return (routed + shared).view_as(x), routing


class BlockAttnResRead(nn.Module):
    def __init__(self, d_model: int, eps: float, backend: BackendStatus) -> None:
        super().__init__()
        self.backend = backend
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        *,
        output_norm_weight: torch.Tensor,
        return_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not sources:
            raise ValueError("Block AttnRes needs at least one source")
        if self.backend.selected is KernelBackend.H100:
            from fla.ops.attnres import fused_attnres

            result = fused_attnres(
                query=self.pseudo_query,
                residuals=sources,
                rms_weight=self.key_weight,
                output_rms_weight=output_norm_weight,
                rms_eps=self.eps,
                return_weights=return_weights,
                checkpoint_level=1,
            )
            if return_weights:
                hidden, weights = result
                return hidden, weights.detach().float().mean(dim=tuple(range(1, weights.ndim)))
            return result, None

        values = torch.stack(list(sources), dim=0)
        keys = rms_norm(values, self.key_weight, self.eps)
        logits = torch.einsum("d,sbtd->sbt", self.pseudo_query.float(), keys.float())
        weights = logits.softmax(dim=0).to(values.dtype)
        hidden = torch.einsum("sbt,sbtd->btd", weights, values)
        hidden = rms_norm(hidden, output_norm_weight, self.eps)
        mean_weights = weights.detach().float().mean(dim=(1, 2)) if return_weights else None
        return hidden, mean_weights


class BlockAttnResState:
    def __init__(self, embedding: torch.Tensor, block_size: int) -> None:
        self.completed: list[torch.Tensor] = [embedding]
        self.partial: torch.Tensor | None = None
        self.block_size = block_size
        self.in_partial = 0

    def sources(self) -> list[torch.Tensor]:
        return [*self.completed] if self.partial is None else [*self.completed, self.partial]

    def add(self, output: torch.Tensor) -> None:
        self.partial = output if self.partial is None else self.partial + output
        self.in_partial += 1
        if self.in_partial == self.block_size:
            self.completed.append(self.partial)
            self.partial = None
            self.in_partial = 0


class K3MiniBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, backend: BackendStatus) -> None:
        super().__init__()
        use_global = cfg.global_attn_every > 0 and (layer_idx + 1) % cfg.global_attn_every == 0
        self.mixer_kind = "nope_mla" if use_global else "kda"
        self.mixer = NoPELatentAttention(cfg) if use_global else KimiDeltaAttention(cfg, backend)
        self.attn_read = BlockAttnResRead(cfg.d_model, cfg.rms_norm_eps, backend)
        self.ffn_read = BlockAttnResRead(cfg.d_model, cfg.rms_norm_eps, backend)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.is_dense = layer_idx == 0
        self.ffn: SwiGLU | LatentMoE
        self.ffn = (
            SwiGLU(cfg.d_model, cfg.dense_ffn_dim, cfg.d_model) if self.is_dense else LatentMoE(cfg, backend)
        )

    def forward(
        self,
        state: BlockAttnResState,
        *,
        diagnostics: bool,
        checkpoint_sublayers: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        attn_sources = state.sources()

        def attention_step(*sources: torch.Tensor) -> torch.Tensor:
            hidden, _ = self.attn_read(
                sources,
                output_norm_weight=self.attn_norm.weight,
                return_weights=False,
            )
            return self.mixer(hidden)

        if checkpoint_sublayers:
            attn_output = checkpoint(
                attention_step,
                *attn_sources,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            attn_weights = None
        else:
            hidden, attn_weights = self.attn_read(
                attn_sources,
                output_norm_weight=self.attn_norm.weight,
                return_weights=diagnostics,
            )
            attn_output = self.mixer(hidden)
        state.add(attn_output)

        ffn_sources = state.sources()
        can_checkpoint_ffn = checkpoint_sublayers and (
            self.is_dense
            or (isinstance(self.ffn, LatentMoE) and isinstance(self.ffn.router, SoftmaxTopKRouter))
        )
        if can_checkpoint_ffn:

            def ffn_step(*sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ffn_hidden, _ = self.ffn_read(
                    sources,
                    output_norm_weight=self.ffn_norm.weight,
                    return_weights=False,
                )
                if self.is_dense:
                    result = self.ffn(ffn_hidden)
                    step_zero = ffn_hidden.new_zeros(())
                    return result, step_zero, step_zero
                result, step_routing = self.ffn(ffn_hidden, diagnostics=False)
                return result, step_routing.auxiliary_loss, step_routing.z_loss

            ffn_output, auxiliary_loss, z_loss = checkpoint(
                ffn_step,
                *ffn_sources,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            ffn_weights = None
            router_stats = {}
        else:
            hidden, ffn_weights = self.ffn_read(
                ffn_sources,
                output_norm_weight=self.ffn_norm.weight,
                return_weights=diagnostics,
            )
            if self.is_dense:
                ffn_output = self.ffn(hidden)
                zero = hidden.new_zeros(())
                auxiliary_loss, z_loss = zero, zero
                router_stats = {}
            else:
                ffn_output, routing = self.ffn(hidden, diagnostics=diagnostics)
                auxiliary_loss, z_loss = routing.auxiliary_loss, routing.z_loss
                router_stats = (
                    {
                        "load": routing.load,
                        "entropy": routing.entropy,
                        "dead_experts": int((routing.load == 0).sum().item()),
                        "max_load_violation": routing.max_load_violation,
                    }
                    if diagnostics
                    else {}
                )
        state.add(ffn_output)
        return (
            auxiliary_loss,
            z_loss,
            {
                "mixer": self.mixer_kind,
                "dense_ffn": self.is_dense,
                "attnres_attention_weights": attn_weights,
                "attnres_ffn_weights": ffn_weights,
                "router": router_stats,
            },
        )


class K3MiniForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.backend = resolve_backend(cfg.kernel_backend, cfg.loss_backend)
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [K3MiniBlock(cfg, layer_idx, self.backend) for layer_idx in range(cfg.n_layers)]
        )
        self.final_read = BlockAttnResRead(cfg.d_model, cfg.rms_norm_eps, self.backend)
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if self.backend.loss_backend is LossBackend.LIGER:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            self.loss_fn: nn.Module | None = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
                accum_dtype=torch.float32,
            )
        elif self.backend.loss_backend is LossBackend.FLA:
            from fla.modules import FusedLinearCrossEntropyLoss

            self.loss_fn = FusedLinearCrossEntropyLoss(
                ignore_index=-100,
                num_chunks=8,
                accumulate_grad_in_fp32=True,
            )
        else:
            self.loss_fn = None
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _lm_loss(
        self, hidden: torch.Tensor, labels: torch.Tensor, return_logits: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.backend.loss_backend is LossBackend.LIGER and not return_logits:
            assert self.loss_fn is not None
            return _run_external_fused_loss(
                self.loss_fn,
                self.lm_head.weight,
                hidden,
                labels,
            ), None
        if self.backend.loss_backend is LossBackend.FLA and not return_logits:
            assert self.loss_fn is not None
            return self.loss_fn(hidden, labels, self.lm_head.weight), None
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        return loss, logits if return_logits else None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_logits: bool = False,
        return_diagnostics: bool = False,
    ) -> ModelOutput:
        embedding = self.token_embedding(input_ids)
        if self.backend.selected is KernelBackend.H100:
            if not torch.is_autocast_enabled("cuda"):
                raise RuntimeError(
                    "the H100 backend requires CUDA BF16 autocast so all AttnRes "
                    "residual sources share a BF16 dtype"
                )
            embedding = embedding.to(torch.get_autocast_dtype("cuda"))
        state = BlockAttnResState(embedding, self.cfg.attnres_block_size)
        aux_losses: list[torch.Tensor] = []
        z_losses: list[torch.Tensor] = []
        layer_diagnostics: list[dict[str, Any]] = []
        use_checkpoint = (
            self.cfg.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and not return_diagnostics
        )
        for layer in self.layers:
            aux, z_loss, layer_stats = layer(
                state,
                diagnostics=return_diagnostics,
                checkpoint_sublayers=use_checkpoint,
            )
            aux_losses.append(aux)
            z_losses.append(z_loss)
            if return_diagnostics:
                layer_diagnostics.append(layer_stats)

        hidden, final_weights = self.final_read(
            state.sources(),
            output_norm_weight=self.final_norm.weight,
            return_weights=return_diagnostics,
        )
        zero = hidden.new_zeros(())
        router_aux = torch.stack(aux_losses).mean() if aux_losses else zero
        router_z = torch.stack(z_losses).mean() if z_losses else zero
        lm_loss: torch.Tensor | None = None
        logits: torch.Tensor | None = self.lm_head(hidden) if return_logits and labels is None else None
        total_loss: torch.Tensor | None = None
        if labels is not None:
            lm_loss, logits = self._lm_loss(hidden, labels, return_logits)
            total_loss = (
                lm_loss
                + self.cfg.router_aux_loss_coefficient * router_aux
                + self.cfg.router_z_loss_coefficient * router_z
            )
        return ModelOutput(
            loss=total_loss,
            lm_loss=lm_loss,
            router_aux_loss=router_aux,
            router_z_loss=router_z,
            logits=logits,
            diagnostics={
                "backend": self.backend.as_dict(),
                "layers": layer_diagnostics,
                "final_attnres_weights": final_weights,
                "attnres_sources": len(state.sources()),
            }
            if return_diagnostics
            else {},
        )

    @torch.no_grad()
    def update_router_biases(self) -> None:
        for module in self.modules():
            if isinstance(module, SigmoidNoAuxTopKRouter):
                module.update_bias()

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        routed_total = sum(
            module.gate_up_weight.numel() + module.down_weight.numel()
            for module in self.modules()
            if isinstance(module, StackedRoutedExperts)
        )
        active = total - routed_total + routed_total * self.cfg.top_k // self.cfg.n_routed_experts
        return {"total": total, "active_estimate": active}

    @torch.no_grad()
    def router_diagnostics(self) -> dict[str, Any]:
        loads: list[list[int]] = []
        entropies: list[float] = []
        max_violations: list[float] = []
        dead_streak = 0
        for module in self.modules():
            if not isinstance(module, (SoftmaxTopKRouter, SigmoidNoAuxTopKRouter)):
                continue
            load = module.last_load.float().clone()
            if dist.is_available() and dist.is_initialized() and isinstance(module, SoftmaxTopKRouter):
                dist.all_reduce(load)
            module.consecutive_dead_steps.copy_(
                torch.where(
                    load == 0,
                    module.consecutive_dead_steps + 1,
                    torch.zeros_like(module.consecutive_dead_steps),
                )
            )
            loads.append([int(value) for value in load.cpu().tolist()])
            probabilities = load / load.sum().clamp_min(1.0)
            nonzero = probabilities[probabilities > 0]
            entropies.append(float((-(nonzero * nonzero.log()).sum()).item()) if nonzero.numel() else 0.0)
            mean = load.mean().clamp_min(1.0)
            max_violations.append(float(((load.max() - mean) / mean).item()))
            dead_streak = max(dead_streak, int(module.consecutive_dead_steps.max().item()))
        return {
            "expert_loads": loads,
            "mean_load_entropy": sum(entropies) / max(1, len(entropies)),
            "dead_experts": sum(sum(value == 0 for value in layer) for layer in loads),
            "max_load_violation": max(max_violations, default=0.0),
            "max_consecutive_dead_steps": dead_streak,
        }


def estimate_parameter_counts(cfg: ModelConfig) -> dict[str, int]:
    """Instantiate on the meta device to calculate the exact configured shape cheaply."""
    with torch.device("meta"):
        model = K3MiniForCausalLM(cfg)
    return model.parameter_counts()
