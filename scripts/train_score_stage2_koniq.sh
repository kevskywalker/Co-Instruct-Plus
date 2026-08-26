#!/bin/bash
# Thin wrapper: paper §3.2 KonIQ-only Stage-2 (online pairing on train_koniq_7k).
# Usage (from repo root):
#   export ROOT_DATA=/path/to/Co-Instruct++    # must contain koniq10k/
#   bash scripts/gen_soft_label_koniq.sh       # if train_koniq_7k.json is missing
#   bash scripts/train_score_stage2_koniq.sh 0,1,2,3
#
# MODEL defaults to the Stage-1 checkpoint (two-stage paper pipeline).
# To match the original lab run from the base Qwen3-VL weights:
#   MODEL=/path/to/Qwen3-VL-8B-Instruct bash scripts/train_score_stage2_koniq.sh 0,1,2,3
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export STAGE2_PROTOCOL=koniq
exec bash "${ROOT}/qwen/scripts/train_score.sh" "$@"
