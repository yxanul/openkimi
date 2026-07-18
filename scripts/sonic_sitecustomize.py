"""Import-only compatibility for QuACK/SonicMoE workers running on Torch 2.7.

QuACK 0.6.1 imports its FP4 dtype map even when only BF16 kernels are used.
Torch 2.7 does not expose ``float4_e2m1fn_x2``, so give the import path a unique
sentinel. No OpenKimi or SonicMoE BF16 kernel uses this value.
"""

from __future__ import annotations

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = object()
