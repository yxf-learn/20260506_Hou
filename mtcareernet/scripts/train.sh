#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
python -m mtcareernet.cli.train \
    --config-name default \
    --run-id "$RUN_ID" \
    --device "${MTC_DEVICE:-cuda:0}" \
    --seed "${MTC_SEED:-20260219}" \
    --overrides training.epochs=60 fairness.beta=1.0
