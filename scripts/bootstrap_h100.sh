#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
core_cuda_home="${K3MINI_CORE_CUDA_HOME:-/usr/local/cuda-12.6}"
experiment_cuda_home="${K3MINI_EXPERIMENT_CUDA_HOME:-/usr/local/cuda-13.2}"
uv_bin="${UV:-${HOME}/.local/bin/uv}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This bootstrap is only supported on Linux x86-64 H100 hosts." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required; provision an NVIDIA GPU driver first." >&2
  exit 1
fi
if [[ "${K3MINI_SKIP_APT:-0}" != "1" ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run as root, or set K3MINI_SKIP_APT=1 after installing system dependencies." >&2
    exit 1
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    cuda-compiler-12-6 \
    git \
    libcudnn9-dev-cuda-12 \
    libnccl2=2.24.3-1+cuda12.6 \
    libnccl-dev=2.24.3-1+cuda12.6 \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-venv
fi

if [[ ! -x "${core_cuda_home}/bin/nvcc" ]]; then
  echo "The CUDA 12.6 compiler was not found at ${core_cuda_home}." >&2
  exit 1
fi
if [[ "${K3MINI_INSTALL_EXPERIMENTS:-1}" == "1" && ! -x "${experiment_cuda_home}/bin/nvcc" ]]; then
  echo "CUDA 13.2 was not found at ${experiment_cuda_home}; it is required by the CuTe experiments." >&2
  exit 1
fi

export CUDA_HOME="${core_cuda_home}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"

if [[ ! -x "${uv_bin}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "${repo_root}"
"${uv_bin}" sync --locked --extra cuda

if [[ "${K3MINI_INSTALL_EXPERIMENTS:-1}" == "1" ]]; then
  CUDA_HOME="${experiment_cuda_home}" UV="${uv_bin}" \
    "${repo_root}/scripts/install_sonic_isolated.sh"
fi

runner=()
if [[ "${K3MINI_INSTALL_EXPERIMENTS:-1}" == "1" ]]; then
  runner=("${repo_root}/scripts/run_with_sonic.sh")
fi
"${runner[@]}" "${repo_root}/.venv/bin/python" - <<'PY'
import torch
import transformer_engine
from fla.ops.kda import chunk_intra

assert chunk_intra.OPENKIMI_KDA_GUARDED_DIAGONAL_VERSION == 1
assert chunk_intra._OPENKIMI_GUARD_MAX_LOG2_SPAN <= 232.0
assert chunk_intra._OPENKIMI_GUARD_DOT_PRECISION == "tf32x3"

print(
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine": transformer_engine.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "openkimi_fla_patch": (
            chunk_intra.OPENKIMI_KDA_GUARDED_DIAGONAL_VERSION
        ),
        "kda_guard_span": chunk_intra._OPENKIMI_GUARD_MAX_LOG2_SPAN,
        "kda_dot_precision": chunk_intra._OPENKIMI_GUARD_DOT_PRECISION,
    }
)
PY

echo "H100 environment is ready from uv.lock."
