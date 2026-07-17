"""Compatibility exports for the packaged implementation.

New code should import from :mod:`k3mini`.
"""

from k3mini.config import ModelConfig
from k3mini.model import (
    BlockAttnResRead,
    BlockAttnResState,
    K3MiniForCausalLM,
    KimiDeltaAttention,
    LatentMoE,
    ModelOutput,
    NoPELatentAttention,
    RMSNorm,
    SigmoidNoAuxTopKRouter,
    SoftmaxTopKRouter,
    SwiGLU,
    estimate_parameter_counts,
    kda_recurrent_reference,
)

KimiStyleMoELM = K3MiniForCausalLM
KimiDeltaAttentionReference = KimiDeltaAttention
NoAuxTopKRouter = SigmoidNoAuxTopKRouter

__all__ = [
    "BlockAttnResRead",
    "BlockAttnResState",
    "K3MiniForCausalLM",
    "KimiDeltaAttention",
    "KimiDeltaAttentionReference",
    "KimiStyleMoELM",
    "LatentMoE",
    "ModelConfig",
    "ModelOutput",
    "NoAuxTopKRouter",
    "NoPELatentAttention",
    "RMSNorm",
    "SigmoidNoAuxTopKRouter",
    "SoftmaxTopKRouter",
    "SwiGLU",
    "estimate_parameter_counts",
    "kda_recurrent_reference",
]
