#!/bin/bash
# Filename-compatible wrapper for Co-Instruct++/qwen/scripts/train_qwen_koniq.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec bash "${ROOT}/scripts/train_score_stage2_koniq.sh" "$@"
