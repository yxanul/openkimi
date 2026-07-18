from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def current_scaling_recipe() -> Any:
    """Construct TE's stateless per-tensor Current Scaling HYBRID recipe."""
    from transformer_engine.common.recipe import Float8CurrentScaling, Format

    return Float8CurrentScaling(
        fp8_format=Format.HYBRID,
        fp8_dpa=False,
        fp8_mha=False,
    )


class CurrentScalingLinear(nn.Module):
    """Transformer Engine Linear with an FP32 master weight."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        import transformer_engine.pytorch as te

        self.linear = te.Linear(
            input_dim,
            output_dim,
            bias=False,
            params_dtype=torch.float32,
            init_method=lambda weight: nn.init.normal_(weight, mean=0.0, std=0.02),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        return self.linear(x, is_first_microbatch=is_first_microbatch)


class CurrentScalingSwiGLU(nn.Module):
    """One FP8 gate/up GEMM, a BF16 SwiGLU activation, and one FP8 down GEMM."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = CurrentScalingLinear(input_dim, 2 * hidden_dim)
        self.down_proj = CurrentScalingLinear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        gate, up = self.gate_up_proj(
            x,
            is_first_microbatch=is_first_microbatch,
        ).chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden, is_first_microbatch=is_first_microbatch)


def te_checkpoint(function: Any, *args: torch.Tensor) -> Any:
    import transformer_engine.pytorch as te

    return te.checkpoint(
        function,
        *args,
        use_reentrant=False,
    )


def _current_scaling_quantizers(device: torch.device) -> tuple[Any, Any, Any]:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import Float8CurrentScalingQuantizer

    input_quantizer = Float8CurrentScalingQuantizer(
        tex.DType.kFloat8E4M3,
        device=device,
        rowwise=True,
        columnwise=True,
    )
    weight_quantizer = Float8CurrentScalingQuantizer(
        tex.DType.kFloat8E4M3,
        device=device,
        rowwise=True,
        columnwise=True,
    )
    grad_output_quantizer = Float8CurrentScalingQuantizer(
        tex.DType.kFloat8E5M2,
        device=device,
        rowwise=True,
        columnwise=True,
    )
    return input_quantizer, weight_quantizer, grad_output_quantizer


def resolve_fp8_lm_head_chunk_size(
    token_count: int,
    physical_vocab_size: int,
    hidden_dim: int,
    requested_chunk_size: int | None,
) -> int:
    """Resolve the configured chunk or reproduce the original Liger-style heuristic."""
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if requested_chunk_size is not None:
        if requested_chunk_size < 16 or requested_chunk_size % 16:
            raise ValueError("requested_chunk_size must be a positive multiple of 16")
        return min(requested_chunk_size, token_count)
    increase_factor = math.ceil(physical_vocab_size / hidden_dim)
    target = math.ceil(token_count / increase_factor)
    return max(16, 1 << (target - 1).bit_length())


def _import_quack_cross_entropy_fwd_out() -> Any:
    """Import QuACK CE on Torch 2.7 without enabling its unused FP4 code paths."""
    missing_fp4_dtype = not hasattr(torch, "float4_e2m1fn_x2")
    if missing_fp4_dtype:
        torch.float4_e2m1fn_x2 = object()  # type: ignore[attr-defined]
    try:
        from quack.cross_entropy import cross_entropy_fwd_out
    finally:
        if missing_fp4_dtype:
            del torch.float4_e2m1fn_x2
    return cross_entropy_fwd_out


def _fp8_linear_cross_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    logical_vocab_size: int,
    requested_chunk_size: int | None,
    ce_backend: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    import triton
    from transformer_engine.pytorch.ops import BasicLinear

    if ce_backend == "liger":
        from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
        from liger_kernel.ops.fused_linear_cross_entropy import MAX_FUSED_SIZE
        from liger_kernel.ops.utils import is_hip

        quack_cross_entropy_fwd_out = None
    elif ce_backend == "quack":
        quack_cross_entropy_fwd_out = _import_quack_cross_entropy_fwd_out()
        liger_cross_entropy_kernel = None
        MAX_FUSED_SIZE = None
        is_hip = None
    else:
        raise ValueError(f"unsupported FP8 cross-entropy backend: {ce_backend}")

    if hidden.ndim != 2 or weight.ndim != 2 or labels.ndim != 1:
        raise ValueError("FP8 fused loss expects hidden [N,H], weight [V,H], and labels [N]")
    token_count, hidden_dim = hidden.shape
    physical_vocab_size, weight_hidden_dim = weight.shape
    if hidden_dim != weight_hidden_dim:
        raise ValueError("hidden and LM-head weight dimensions do not match")
    if labels.shape[0] != token_count:
        raise ValueError("labels and hidden must contain the same number of tokens")
    if physical_vocab_size % 16 or hidden_dim % 16:
        raise ValueError("Transformer Engine FP8 GEMMs require dimensions divisible by 16")
    if not 0 < logical_vocab_size <= physical_vocab_size:
        raise ValueError("logical_vocab_size must be in (0, physical_vocab_size]")

    input_requires_grad = hidden.requires_grad
    weight_requires_grad = weight.requires_grad
    compute_gradients = input_requires_grad or weight_requires_grad
    input_quantizer, weight_quantizer, grad_output_quantizer = _current_scaling_quantizers(
        hidden.device
    )

    weight_quantizer.set_usage(
        rowwise=True,
        columnwise=input_requires_grad,
    )
    quantized_weight = weight_quantizer(weight)

    grad_hidden = torch.zeros_like(hidden) if input_requires_grad else None
    grad_weight = (
        torch.zeros_like(weight, dtype=torch.float32)
        if weight_requires_grad
        else None
    )
    loss_per_token = torch.zeros(token_count, dtype=torch.float32, device=hidden.device)

    chunk_size = resolve_fp8_lm_head_chunk_size(
        token_count,
        physical_vocab_size,
        hidden_dim,
        requested_chunk_size,
    )
    num_chunks = triton.cdiv(token_count, chunk_size)
    block_size = (
        min(MAX_FUSED_SIZE, triton.next_power_of_2(logical_vocab_size))
        if MAX_FUSED_SIZE is not None
        else None
    )

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, token_count)
        hidden_chunk = hidden[start:end]
        labels_chunk = labels[start:end].contiguous()
        unpadded_rows = end - start
        padded_rows = ((unpadded_rows + 15) // 16) * 16
        if padded_rows != unpadded_rows:
            hidden_chunk = F.pad(hidden_chunk, (0, 0, 0, padded_rows - unpadded_rows))
            labels_chunk = F.pad(
                labels_chunk,
                (0, padded_rows - unpadded_rows),
                value=-100,
            )
        logits, saved_hidden, saved_weight = BasicLinear._functional_forward(
            input=hidden_chunk,
            weight=quantized_weight,
            dtype=torch.bfloat16,
            with_quantized_compute=True,
            input_quantizer=input_quantizer,
            input_requires_grad=input_requires_grad,
            weight_requires_grad=weight_requires_grad,
        )
        logits = logits.contiguous()
        loss_chunk = torch.zeros(
            padded_rows,
            dtype=torch.float32,
            device=hidden.device,
        )
        row_count = logits.shape[0]

        if ce_backend == "liger":
            assert liger_cross_entropy_kernel is not None
            assert block_size is not None
            assert is_hip is not None
            # Liger sees only the logical vocabulary. The physical-only
            # dlogits are zeroed before the Transformer Engine backward.
            liger_cross_entropy_kernel[(row_count,)](
                X_ptr=logits,
                X_stride=logits.stride(-2),
                Y_ptr=labels_chunk,
                Y_stride=labels_chunk.stride(-1),
                weight_ptr=None,
                loss_ptr=loss_chunk,
                z_loss_ptr=None,
                loss_stride=loss_chunk.stride(-1),
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=logical_vocab_size,
                n_non_ignore=token_count,
                sum_non_ignore_weight=token_count,
                weight_sum=0.0,
                ignore_index=-100,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction="mean",
                softcap=None,
                RETURN_Z_LOSS=False,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False,
                HAS_GRADIENTS=compute_gradients,
                BLOCK_SIZE=block_size,
                num_warps=32 if not is_hip() else 16,
            )
        else:
            assert quack_cross_entropy_fwd_out is not None
            # QuACK reduces across the physical row. Negative infinity makes
            # the padded vocabulary mathematically absent from the softmax,
            # and its in-place dx output leaves those columns exactly zero.
            if logical_vocab_size < physical_vocab_size:
                logits[:, logical_vocab_size:].fill_(-torch.inf)
            quack_cross_entropy_fwd_out(
                logits,
                labels_chunk,
                None,
                loss_chunk,
                None,
                logits if compute_gradients else None,
                None,
                -100,
            )
        loss_per_token[start:end].copy_(loss_chunk[:unpadded_rows])

        if compute_gradients:
            if logical_vocab_size < physical_vocab_size:
                logits[:, logical_vocab_size:].zero_()
            chunk_grad_hidden, grad_weight = BasicLinear._functional_backward(
                grad_output=logits,
                input=saved_hidden,
                weight=saved_weight,
                input_requires_grad=input_requires_grad,
                weight_requires_grad=weight_requires_grad,
                dtype=torch.bfloat16,
                grad_weight=grad_weight,
                accumulate_into_grad_weight=chunk_id > 0,
                with_quantized_compute=True,
                input_quantizer=input_quantizer,
                weight_quantizer=weight_quantizer,
                grad_output_quantizer=grad_output_quantizer,
            )
            if grad_hidden is not None:
                grad_hidden[start:end].copy_(chunk_grad_hidden[:unpadded_rows])

    if ce_backend == "quack":
        valid_tokens = (labels != -100).sum().clamp_min(1).to(torch.float32)
        gradient_scale = valid_tokens.reciprocal()
        loss = loss_per_token.sum() * gradient_scale
    else:
        gradient_scale = None
        loss = loss_per_token.sum()
    return loss, grad_hidden, grad_weight, gradient_scale


class _CurrentScalingFusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        logical_vocab_size: int,
        requested_chunk_size: int | None,
        ce_backend: str,
    ) -> torch.Tensor:
        loss, grad_hidden, grad_weight, gradient_scale = _fp8_linear_cross_entropy_forward(
            hidden,
            weight,
            labels,
            logical_vocab_size,
            requested_chunk_size,
            ce_backend,
        )
        if grad_hidden is not None and grad_weight is not None:
            ctx.save_for_backward(grad_hidden.detach(), grad_weight.detach())
        else:
            ctx.save_for_backward()
        ctx.gradient_scale = gradient_scale
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        from liger_kernel.ops.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_backward,
        )

        grad_hidden, grad_weight = ctx.saved_tensors
        if ctx.gradient_scale is not None:
            grad_output = grad_output * ctx.gradient_scale
        grad_hidden, grad_weight, _ = fused_linear_cross_entropy_backward(
            grad_output,
            grad_hidden,
            grad_weight,
            None,
        )
        return grad_hidden, grad_weight, None, None, None, None


class CurrentScalingFusedLinearCrossEntropyLoss(nn.Module):
    """Chunked TE FP8 LM head plus an in-place fused CE provider."""

    def __init__(
        self,
        logical_vocab_size: int,
        chunk_size: int | None = None,
        ce_backend: str = "liger",
    ) -> None:
        super().__init__()
        self.logical_vocab_size = logical_vocab_size
        self.chunk_size = chunk_size
        if ce_backend not in {"liger", "quack"}:
            raise ValueError(f"unsupported FP8 cross-entropy backend: {ce_backend}")
        self.ce_backend = ce_backend

    def forward(
        self,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            weight = weight.detach()
            hidden = hidden.detach()
        return _CurrentScalingFusedLinearCrossEntropy.apply(
            hidden,
            weight,
            labels,
            self.logical_vocab_size,
            self.chunk_size,
            self.ce_backend,
        )
