#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${QUACK_TARGET:-${repo_root}/.quack-packages}"
python="${PYTHON:-${repo_root}/.venv/bin/python}"
uv_bin="${UV:-uv}"

if [[ ! -x "${python}" ]]; then
  echo "Python environment not found at ${python}; run 'uv sync --locked --extra cuda' first." >&2
  exit 1
fi

"${python}" -c 'import sys; assert sys.version_info >= (3, 12), "QuACK requires Python 3.12+"'
mkdir -p "${target}"

# Keep QuACK's CUDA-13/CuTe DSL stack outside the locked training environment.
# The complete, explicitly pinned target was verified on CUDA 13.2 with Torch 2.7.
"${uv_bin}" pip install \
  --python "${python}" \
  --target "${target}" \
  --no-deps \
  quack-kernels==0.6.1 \
  apache-tvm-ffi==0.1.12 \
  cuda-bindings==12.9.2 \
  cuda-pathfinder==1.2.2 \
  cuda-python==12.9.0 \
  nvidia-cuda-nvdisasm==13.3.73 \
  nvidia-cutlass-dsl==4.6.0 \
  nvidia-cutlass-dsl-libs-base==4.6.0 \
  nvidia-cutlass-dsl-libs-core==4.6.0 \
  nvidia-cutlass-dsl-libs-cu12==4.6.0 \
  nvidia-cutlass-dsl-libs-cu13==4.6.0 \
  protobuf==6.33.6 \
  torch-c-dlpack-ext==0.1.5 \
  einops==0.8.2

echo "Installed the isolated QuACK stack at ${target}"
echo "Run commands through scripts/run_with_quack.sh"
