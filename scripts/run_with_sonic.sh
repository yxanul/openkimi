#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${SONIC_TARGET:-${QUACK_TARGET:-${repo_root}/.quack-packages}}"

if [[ ! -d "${target}/sonicmoe" ]]; then
  echo "SonicMoE target not found at ${target}; run scripts/install_sonic_isolated.sh first." >&2
  exit 1
fi

export QUACK_TARGET="${target}"
exec "${repo_root}/scripts/run_with_quack.sh" "$@"
