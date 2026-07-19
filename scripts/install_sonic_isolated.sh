#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${SONIC_TARGET:-${QUACK_TARGET:-${repo_root}/.quack-packages}}"
python="${PYTHON:-${repo_root}/.venv/bin/python}"
uv_bin="${UV:-uv}"
sonic_revision="0349404acd7952592f73d180ff0c1510f6d112c2"
gram_newton_schulz_revision="e45d0aca7083cb275c9a303220c05c4abecd9187"

QUACK_TARGET="${target}" PYTHON="${python}" UV="${uv_bin}" \
  "${repo_root}/scripts/install_quack_isolated.sh"

"${uv_bin}" pip install \
  --python "${python}" \
  --target "${target}" \
  --no-deps \
  "git+https://github.com/Dao-AILab/sonic-moe.git@${sonic_revision}" \
  "git+https://github.com/Dao-AILab/gram-newton-schulz.git@${gram_newton_schulz_revision}"

cp "${repo_root}/scripts/sonic_sitecustomize.py" "${target}/sitecustomize.py"

echo "Installed SonicMoE ${sonic_revision}, Gram Newton-Schulz ${gram_newton_schulz_revision}, and QuACK in ${target}"
echo "Run commands through scripts/run_with_sonic.sh"
