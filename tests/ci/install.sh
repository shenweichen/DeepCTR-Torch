#!/usr/bin/env bash
set -euo pipefail

python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q "numpy<2"

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  python -m pip install -q --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
else
  python -m pip install -q "torch==${TORCH_VERSION}"
fi

python -m pip install -q requests pytest pytest-cov python-coveralls pandas
python -m pip install -e .
python -m pip check
