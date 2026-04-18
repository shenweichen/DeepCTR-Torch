#!/usr/bin/env bash
set -euo pipefail

pytest --cov=deepctr_torch --cov-report=xml --cov-report=term-missing:skip-covered
