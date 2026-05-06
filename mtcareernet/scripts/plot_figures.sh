#!/usr/bin/env bash
set -euo pipefail
python -m experiments.run_all --only exp_01_dgp1 exp_07_conformal
python tools/plot_paper_figures.py --suite results/runs --out results/figures
