#!/usr/bin/env bash
# Build the isolated benchmark venv. Installs sbijax (editable, from this repo)
# plus the two comparison libraries from GitHub. NOTE: the `sabc` (MLX) install
# triggers a C++ build via scikit-build-core (cmake + ninja + nanobind); Apple
# Silicon only. sbijax requires Python >= 3.12, so we pin 3.12 via uv.
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
echo "Done. Activate with: source experiments/sabc-performance/.venv/bin/activate"
