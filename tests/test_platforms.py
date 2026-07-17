from __future__ import annotations

import os

import pytest
import torch

from k3mini.backends import resolve_backend
from k3mini.config import KernelBackend
from k3mini.model import K3MiniForCausalLM


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_forward_backward(tiny_model_config) -> None:
    model = K3MiniForCausalLM(tiny_model_config).to("mps")
    tokens = torch.randint(0, tiny_model_config.vocab_size, (1, 5), device="mps")
    output = model(tokens, tokens)
    assert output.loss is not None
    output.loss.backward()
    assert torch.isfinite(output.loss).item()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9
    or os.environ.get("K3MINI_RUN_GPU_TESTS") != "1",
    reason="set K3MINI_RUN_GPU_TESTS=1 on SM90+ with CUDA extras",
)
def test_h100_backend_selected_and_kda_parity(tiny_model_config) -> None:
    tiny_model_config.n_heads = 1
    tiny_model_config.d_model = 128
    tiny_model_config.kda_head_dim = 128
    tiny_model_config.mla_qk_head_dim = 128
    tiny_model_config.mla_v_head_dim = 128
    tiny_model_config.kernel_backend = KernelBackend.H100
    tiny_model_config.validate()
    status = resolve_backend(KernelBackend.H100)
    assert status.selected is KernelBackend.H100
    model = K3MiniForCausalLM(tiny_model_config).cuda()
    tokens = torch.randint(0, tiny_model_config.vocab_size, (1, 64), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens, tokens, return_diagnostics=True)
    assert output.loss is not None
    output.loss.backward()
    assert output.diagnostics["backend"]["kda"] == "fla.ops.kda.chunk_kda"
