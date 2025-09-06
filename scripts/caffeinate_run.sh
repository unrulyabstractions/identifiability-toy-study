#!/usr/bin/env bash
# Keep the Mac awake while running any command (great for long training)
# Usage: scripts/caffeinate_run.sh uv run python train.py
set -euo pipefail
caffeinate -dimsu "$@"
