#!/usr/bin/env bash
set -euo pipefail

export DEEPCTR_EXAMPLE_EPOCHS="${DEEPCTR_EXAMPLE_EPOCHS:-1}"

scripts=(
  "run_classification_criteo.py"
  "run_regression_movielens.py"
  "run_multitask_learning.py"
  "run_multivalue_movielens.py"
  "run_din.py"
  "run_dien.py"
)

pushd examples >/dev/null
for script in "${scripts[@]}"; do
  echo "Running example smoke test: ${script} (epochs=${DEEPCTR_EXAMPLE_EPOCHS})"
  python "${script}"
done
popd >/dev/null
