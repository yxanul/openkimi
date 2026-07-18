#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${QUACK_TARGET:-${repo_root}/.quack-packages}"
cuda_home_default="/usr/local/cuda-13.2"

if [[ ! -d "${target}/quack" ]]; then
  echo "QuACK target not found at ${target}; run scripts/install_quack_isolated.sh first." >&2
  exit 1
fi
if [[ ! -d "${cuda_home_default}" ]]; then
  cuda_home_default="/usr/local/cuda"
fi

export CUDA_HOME="${CUDA_HOME:-${cuda_home_default}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${target}:${target}/nvidia_cutlass_dsl/dsl_packages${PYTHONPATH:+:${PYTHONPATH}}"

exec "$@"
