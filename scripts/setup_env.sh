#!/usr/bin/env bash
set -euo pipefail

# Run this from the project root (where pyproject.toml is).
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
# Install your project in editable mode; pulls deps from pyproject.toml
pip install -e .

# Optional: dev tools we’ll use soon
pip install pytest ipykernel

echo "✅ Env ready. Activate next time with: source .venv/bin/activate"
