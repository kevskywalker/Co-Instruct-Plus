#!/usr/bin/env bash
set -euo pipefail

# Public Stage-I SFT entry point. Optional positional argument: GPU list.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export STAGE1_CONFIG=${STAGE1_CONFIG:-"${ROOT}/configs/stage1_sft.yaml"}
exec bash "${ROOT}/qwen/scripts/finetune.sh" "$@"
