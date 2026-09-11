#!/usr/bin/env bash
set -euo pipefail

# Public Stage-II entry point: KonIQ-only dynamic pair training from Stage I.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export STAGE2_PROTOCOL=koniq
export STAGE2_CONFIG=${STAGE2_CONFIG:-"${ROOT}/configs/stage2_score_koniq.yaml"}
exec bash "${ROOT}/qwen/scripts/train_score.sh" "$@"
