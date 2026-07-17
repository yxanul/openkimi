from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_swiglu_fwd(
    gate_up,
    route_weight,
    output,
    hidden_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_dim
    row_start = row * 2 * hidden_dim
    gate = tl.load(gate_up + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up + row_start + hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(route_weight + row).to(tl.float32)
    activated = gate * tl.sigmoid(gate)
    tl.store(output + row * hidden_dim + offsets, activated * up * weight, mask=mask)


@triton.jit
def _weighted_swiglu_bwd(
    grad_output,
    gate_up,
    route_weight,
    grad_gate_up,
    grad_route_weight,
    hidden_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_dim
    row_start = row * 2 * hidden_dim
    gate = tl.load(gate_up + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up + row_start + hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_output + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(route_weight + row).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    activated = gate * sigmoid
    grad_activation = sigmoid + gate * sigmoid * (1.0 - sigmoid)
    tl.store(grad_gate_up + row_start + offsets, grad * up * weight * grad_activation, mask=mask)
    tl.store(
        grad_gate_up + row_start + hidden_dim + offsets,
        grad * activated * weight,
        mask=mask,
    )
    tl.store(grad_route_weight + row, tl.sum(grad * activated * up, axis=0))


class _WeightedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, route_weight: torch.Tensor) -> torch.Tensor:
        if not gate_up.is_cuda or not route_weight.is_cuda:
            raise ValueError("fused weighted SwiGLU requires CUDA tensors")
        if gate_up.ndim != 2 or gate_up.shape[-1] % 2:
            raise ValueError("gate_up must have shape [tokens, 2 * hidden_dim]")
        if route_weight.shape != gate_up.shape[:1]:
            raise ValueError("route_weight must have shape [tokens]")
        gate_up = gate_up.contiguous()
        route_weight = route_weight.contiguous()
        hidden_dim = gate_up.shape[-1] // 2
        output = gate_up.new_empty(gate_up.shape[0], hidden_dim)
        block_size = triton.next_power_of_2(hidden_dim)
        _weighted_swiglu_fwd[(gate_up.shape[0],)](
            gate_up,
            route_weight,
            output,
            hidden_dim=hidden_dim,
            block_size=block_size,
        )
        ctx.save_for_backward(gate_up, route_weight)
        ctx.hidden_dim = hidden_dim
        ctx.block_size = block_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up, route_weight = ctx.saved_tensors
        grad_gate_up = torch.empty_like(gate_up)
        grad_route_weight = torch.empty_like(route_weight)
        _weighted_swiglu_bwd[(gate_up.shape[0],)](
            grad_output.contiguous(),
            gate_up,
            route_weight,
            grad_gate_up,
            grad_route_weight,
            hidden_dim=ctx.hidden_dim,
            block_size=ctx.block_size,
        )
        return grad_gate_up, grad_route_weight


def fused_weighted_swiglu(
    gate_up: torch.Tensor,
    route_weight: torch.Tensor,
) -> torch.Tensor:
    return _WeightedSwiGLU.apply(gate_up, route_weight)
