#!/usr/bin/env bash
set -euo pipefail

# Evaluate either a Stage-I or Stage-II checkpoint on the public MICBench-v2 set.
# Required: MODEL_PATH, MICBENCH_ANNOTATION, MICBENCH_IMAGE_ROOT.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${ROOT}/qwen/src:${PYTHONPATH:-}"

: "${MODEL_PATH:?Set MODEL_PATH to a Stage-I or Stage-II checkpoint.}"
: "${MICBENCH_ANNOTATION:?Set MICBENCH_ANNOTATION to micbench_v2_1998.json.}"
: "${MICBENCH_IMAGE_ROOT:?Set MICBENCH_IMAGE_ROOT to extracted MICBench_test/images.}"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/results/micbench-v2/$(basename "$MODEL_PATH")"}
exec "$PYTHON_BIN" -m evaluate.micbench_v2 \
  --model-path "$MODEL_PATH" \
  --annotation "$MICBENCH_ANNOTATION" \
  --image-root "$MICBENCH_IMAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
