#!/bin/bash
# Thin wrapper: Stage-1 SFT from repo root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "${ROOT}/qwen/scripts/finetune.sh" "$@"
