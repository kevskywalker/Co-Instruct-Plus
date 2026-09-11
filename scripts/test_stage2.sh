#!/usr/bin/env bash
set -euo pipefail

# Public Stage-II evaluation entry point. MODEL_PATH and ROOT_DIR are required.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "${ROOT}/scripts/eval_iqa.sh" "$@"
