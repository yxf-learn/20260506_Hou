#!/usr/bin/env bash
set -euo pipefail
python -m mtcareernet.cli.sweep \
    --space configs/sweeps/default.yaml \
    --n-trials "${N_TRIALS:-32}" \
    --budget-hours "${BUDGET_HOURS:-8.0}"
