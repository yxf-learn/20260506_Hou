#!/usr/bin/env bash
set -euo pipefail
python -m mtcareernet.cli.eval \
    --run-id "${RUN_ID:?must set RUN_ID}" \
    --split test \
    --report "results/reports/${RUN_ID}.json"
