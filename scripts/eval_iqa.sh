#!/bin/bash
set -euo pipefail

# Scoring-only IQA eval: inference + PLCC/SRCC.
# Required env:
#   MODEL_PATH   trained checkpoint directory
#   ROOT_DIR     benchmark image/meta root (e.g. Co-Instruct++)
# Optional:
#   PREPROCESSOR_PATH, SAVE_DIR, KONIQ_IMAGES_DIR, SPAQ_IMAGES_DIR,
#   CUDA_VISIBLE_DEVICES, PYTHON_BIN, META_PATHS (space-separated)

ROOT=$(cd "$(dirname "$0")/.." && pwd)
QWEN_SRC="${ROOT}/qwen/src"

if [ -f "${ROOT}/secrets/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/secrets/.env"
    set +a
fi
if [ -f "${ROOT}/configs/env.local.sh" ]; then
    # shellcheck disable=SC1091
    source "${ROOT}/configs/env.local.sh"
fi

export PYTHONPATH="${QWEN_SRC}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=${MODEL_PATH:-${MODEL:-${OUTPUT:-}}}
PREPROCESSOR_PATH=${PREPROCESSOR_PATH:-${MODEL_NAME:-}}
SAVE_DIR=${SAVE_DIR:-${ROOT}/results/iqa_eval}
ROOT_DIR=${ROOT_DIR:-${ROOT_DATA:-}}
PYTHON_BIN=${PYTHON_BIN:-python}

if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "[error] Set MODEL_PATH to a trained checkpoint directory." >&2
    exit 1
fi
if [ -z "$ROOT_DIR" ] || [ ! -d "$ROOT_DIR" ]; then
    echo "[error] Set ROOT_DIR (benchmark root with LIVE/CSIQ/... metas and images)." >&2
    exit 1
fi

shopt -s nullglob
shards=("$MODEL_PATH"/*.safetensors)
shopt -u nullglob
if [ ${#shards[@]} -eq 0 ] && [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo "[error] No *.safetensors under MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

DEFAULT_METAS=(
    "$ROOT_DIR/LIVE/metas/test_live_203.json"
    "$ROOT_DIR/CSIQ/metas/test_csiq_174.fixed.json"
    "$ROOT_DIR/KADID10K/metas/test_kadid_2k.json"
    "$ROOT_DIR/BID/metas/test_bid_117.soft.json"
    "$ROOT_DIR/CLIVE/metas/test_clive_234.json"
    "$ROOT_DIR/KONIQ/metas/test_koniq_2k.json"
    "$ROOT_DIR/TID2013/metas/test_tid2013_3k.json"
    "$ROOT_DIR/SPAQ/metas/test_spaq_2k.json"
    "$ROOT_DIR/AGIQA3K/metas/test_agiqa_3k.json"
    "$ROOT_DIR/PIPAL/metas/test_pipal_1k_new.json"
)

if [ -n "${META_PATHS:-}" ]; then
    # shellcheck disable=SC2206
    METAS=($META_PATHS)
else
    METAS=()
    for m in "${DEFAULT_METAS[@]}"; do
        if [ -f "$m" ]; then
            METAS+=("$m")
        fi
    done
fi

if [ ${#METAS[@]} -eq 0 ]; then
    echo "[error] No meta JSON found under ROOT_DIR=$ROOT_DIR" >&2
    exit 1
fi

echo "[eval] model=${MODEL_PATH}"
echo "[eval] root=${ROOT_DIR}"
echo "[eval] save_dir=${SAVE_DIR}"
echo "[eval] n_metas=${#METAS[@]}"

EVAL_ARGS=(
    --level-names Excellent Good Fair Poor Bad
    --model-path "$MODEL_PATH"
    --save-dir "$SAVE_DIR"
    --root-dir "$ROOT_DIR"
    --batch-size 1
    --meta-paths "${METAS[@]}"
)
if [ -n "${PREPROCESSOR_PATH}" ] && [ -d "${PREPROCESSOR_PATH}" ]; then
    EVAL_ARGS+=(--preprocessor-path "$PREPROCESSOR_PATH")
fi
if [ -n "${KONIQ_IMAGES_DIR:-}" ]; then
    EVAL_ARGS+=(--koniq-images-dir "$KONIQ_IMAGES_DIR")
fi
if [ -n "${SPAQ_IMAGES_DIR:-}" ]; then
    EVAL_ARGS+=(--spaq-images-dir "$SPAQ_IMAGES_DIR")
fi

"$PYTHON_BIN" "${QWEN_SRC}/evaluate/iqa_eval_qwen.py" "${EVAL_ARGS[@]}"

PRED_PATHS=()
GT_PATHS=()
for m in "${METAS[@]}"; do
    base=$(basename "$m")
    pred="${SAVE_DIR}/${base}"
    if [ -f "$pred" ]; then
        PRED_PATHS+=("$pred")
        GT_PATHS+=("$m")
    fi
done

"$PYTHON_BIN" "${QWEN_SRC}/evaluate/cal_plcc_srcc.py" \
    --level_names Excellent Good Fair Poor Bad \
    --pred_paths "${PRED_PATHS[@]}" \
    --gt_paths "${GT_PATHS[@]}"

echo "[eval] done -> ${SAVE_DIR}"
