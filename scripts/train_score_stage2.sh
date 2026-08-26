#!/bin/bash
# Thin wrapper: Stage-2 DeQA score training from repo root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "${ROOT}/qwen/scripts/train_score.sh" "$@"
