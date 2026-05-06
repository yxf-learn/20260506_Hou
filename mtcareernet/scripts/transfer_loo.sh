#!/usr/bin/env bash
set -euo pipefail
for TARGET in compu_1 compu_2 sci_eng normal_uni; do
    python -m mtcareernet.cli.train \
        --config-name default \
        --run-id "loo_${TARGET}" \
        --overrides data=empirical transfer=dann_default \
        loo_target="${TARGET}"
done
