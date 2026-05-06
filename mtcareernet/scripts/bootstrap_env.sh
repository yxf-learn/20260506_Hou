#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3.10}
$PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel
pip install -e ".[dev,viz]"
pre-commit install
