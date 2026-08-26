#!/bin/bash
set -euo pipefail

# Stage 2: score / DeQA pair training on Stage-1 SFT weights.
# Usage (from repo root):
#   export ROOT_DATA=/path/to/pair-images   # required
#   bash scripts/train_score_stage2.sh 0,1,2,3
#   bash scripts/train_score_stage2_koniq.sh 0,1,2,3   # paper KonIQ-only
#
# MODEL defaults to checkpoints/stage1_qsit_sft after Stage 1.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
QWEN_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PKG_ROOT=$(cd "${QWEN_ROOT}/.." && pwd)
# shellcheck disable=SC1091
source "${PKG_ROOT}/scripts/lib.sh"
coinstruct_load_env

export PYTHONPATH="${QWEN_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export COINSTRUCT_SEED=${COINSTRUCT_SEED:-42}

cd "${QWEN_ROOT}"

MODEL=${MODEL:-${OUTPUT_DIR:-${PKG_ROOT}/checkpoints/stage1_qsit_sft}}
MODEL=${MODEL%/}
PYTHON_BIN=${PYTHON_BIN:-python}

# STAGE2_PROTOCOL=koniq overrides env.local PAIRS_JSON/OUTPUT so the 185K
# prebuilt-pair defaults cannot leak into the paper KonIQ-only launcher.
if [ "${STAGE2_PROTOCOL:-}" = "koniq" ]; then
    if [ -z "${ROOT_DATA:-}" ]; then
        echo "[error] ROOT_DATA is not set (KonIQ image root, e.g. Co-Instruct++)." >&2
        exit 1
    fi
    CONFIG_PATH=${STAGE2_CONFIG:-${PKG_ROOT}/configs/stage2_score_koniq.yaml}
    PAIRS_JSON=${KONIQ_META:-${ROOT_DATA%/}/koniq10k/metas/train_koniq_7k.json}
    OUTPUT=${KONIQ_OUTPUT:-${PKG_ROOT}/checkpoints/stage2_score_koniq}
    KONIQ_IMAGES_DIR=${KONIQ_IMAGES_DIR:-${ROOT_DATA%/}/koniq10k/512x384}
    export KONIQ_IMAGES_DIR
    SKIP_ANNOTATION_ENSURE=${SKIP_ANNOTATION_ENSURE:-1}
else
    PAIRS_JSON=${PAIRS_JSON:-${PKG_ROOT}/pairs/pairs_180k.json}
    OUTPUT=${OUTPUT:-${PKG_ROOT}/checkpoints/stage2_score}
    CONFIG_PATH=${STAGE2_CONFIG:-${PKG_ROOT}/configs/stage2_score.yaml}
fi
OUTPUT=${OUTPUT%/}

yaml_value() {
    "$PYTHON_BIN" "${PKG_ROOT}/scripts/yaml_value.py" "$CONFIG_PATH" "$1" "$2"
}

if [ -n "${LEVEL_NAMES:-}" ]; then
    read -r -a LEVEL_NAMES <<< "${LEVEL_NAMES}"
else
    mapfile -t LEVEL_NAMES < <("$PYTHON_BIN" "${PKG_ROOT}/scripts/yaml_value.py" "$CONFIG_PATH" level_names --list)
fi
if [ "${#LEVEL_NAMES[@]}" -eq 0 ]; then
    echo "[error] level_names must contain at least one label." >&2
    exit 1
fi

DS_CONFIG=${DS_CONFIG:-$(yaml_value deepspeed "${PKG_ROOT}/configs/deepspeed/zero3.json")}
if [ "${DS_CONFIG#/}" = "${DS_CONFIG}" ]; then
    DS_CONFIG="${PKG_ROOT}/${DS_CONFIG}"
fi

if [ -z "${ROOT_DATA:-}" ]; then
    echo "[error] ROOT_DATA is not set (Stage-2 pair image root)." >&2
    echo "[error] Example: export ROOT_DATA=/path/to/Co-Instruct++" >&2
    echo "[error] Optional: put lasting overrides in configs/env.local.sh (see configs/env.example)." >&2
    exit 1
fi
export ROOT_DATA

if [ "${SKIP_ANNOTATION_ENSURE:-0}" != "1" ]; then
    coinstruct_ensure_annotations
    PAIRS_JSON=${PAIRS_JSON:-${PKG_ROOT}/pairs/pairs_180k.json}
fi

if [ ! -f "$DS_CONFIG" ]; then
    echo "[error] DeepSpeed config not found: $DS_CONFIG" >&2
    exit 1
fi

GPUS=${1:-0,1,2,3}
ENV_NAME=${ENV_NAME:-train}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}
SEED=${COINSTRUCT_SEED:-42}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-$(yaml_value num_train_epochs 1)}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$(yaml_value per_device_train_batch_size 4)}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$(yaml_value per_device_eval_batch_size 4)}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$(yaml_value gradient_accumulation_steps 1)}
LEARNING_RATE=${LEARNING_RATE:-$(yaml_value learning_rate 2e-5)}
MERGER_LR=${MERGER_LR:-$(yaml_value merger_lr 2e-5)}
VISION_LR=${VISION_LR:-$(yaml_value vision_lr 2e-6)}
WEIGHT_DECAY=${WEIGHT_DECAY:-$(yaml_value weight_decay 0.0)}
WARMUP_RATIO=${WARMUP_RATIO:-$(yaml_value warmup_ratio 0.03)}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-$(yaml_value lr_scheduler_type cosine)}
LOGGING_STEPS=${LOGGING_STEPS:-$(yaml_value logging_steps 1)}
SAVE_STRATEGY=${SAVE_STRATEGY:-$(yaml_value save_strategy steps)}
SAVE_STEPS=${SAVE_STEPS:-$(yaml_value save_steps 4000)}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-$(yaml_value save_total_limit 1)}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-$(yaml_value dataloader_num_workers 4)}
IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-$(yaml_value image_min_pixels 196608)}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-$(yaml_value image_max_pixels 1310720)}
MASTER_PORT=${MASTER_PORT:-$(yaml_value master_port 6689)}
LEVEL_PREFIX=${LEVEL_PREFIX:-$(yaml_value level_prefix "The quality of the image is")}
SOFTKL_LOSS=${SOFTKL_LOSS:-$(yaml_value softkl_loss true)}
WEIGHT_RANK=${WEIGHT_RANK:-$(yaml_value weight_rank 1.0)}
WEIGHT_SOFTKL=${WEIGHT_SOFTKL:-$(yaml_value weight_softkl 1.0)}
WEIGHT_NEXT_TOKEN=${WEIGHT_NEXT_TOKEN:-$(yaml_value weight_next_token 0.05)}
CONTINUOUS_RATING_LOSS=${CONTINUOUS_RATING_LOSS:-$(yaml_value continuous_rating_loss true)}
CLOSESET_RATING_LOSS=${CLOSESET_RATING_LOSS:-$(yaml_value closeset_rating_loss true)}
USE_FIX_STD=${USE_FIX_STD:-$(yaml_value use_fix_std true)}
DETACH_PRED_STD=${DETACH_PRED_STD:-$(yaml_value detach_pred_std true)}
AUTO_RESUME=${AUTO_RESUME:-$(yaml_value auto_resume true)}
PREBUILT_PAIRS=${PREBUILT_PAIRS:-$(yaml_value prebuilt_pairs true)}

if [ ! -f "$MODEL/config.json" ]; then
    echo "[error] Missing config.json under MODEL: $MODEL" >&2
    echo "[error] Run Stage 1 first, or export MODEL=/path/to/stage1_checkpoint." >&2
    exit 1
fi
if [ ! -f "$PAIRS_JSON" ]; then
    echo "[error] Missing Stage-2 data: $PAIRS_JSON" >&2
    if [ "${STAGE2_PROTOCOL:-}" = "koniq" ]; then
        echo "[error] Expected KonIQ train meta (export KONIQ_META or ROOT_DATA/koniq10k/metas/train_koniq_7k.json)." >&2
        echo "[error] Build it with: bash scripts/gen_soft_label_koniq.sh" >&2
    else
        echo "[error] See docs/DATA.md (pairs/pairs_180k.json)." >&2
    fi
    exit 1
fi
if [ ! -d "$ROOT_DATA" ]; then
    echo "[error] ROOT_DATA is not a directory: $ROOT_DATA" >&2
    exit 1
fi

mkdir -p "$OUTPUT"

if [ "${STAGE2_PROTOCOL:-}" = "koniq" ]; then
    "$PYTHON_BIN" - <<PY
import json
import os
import sys

root = os.environ["ROOT_DATA"]
meta = "${PAIRS_JSON}"
with open(meta, encoding="utf-8") as handle:
    samples = json.load(handle)
missing = []
for sample in samples:
    path = os.path.join(root, sample["image"])
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        missing.append(sample["image"])
if missing:
    print(f"[error] missing koniq images: {len(missing)}", file=sys.stderr)
    print("[error] sample missing entries:", missing[:20], file=sys.stderr)
    raise SystemExit(1)
print(f"[info] koniq training images verified: {len(samples)}")
PY
fi

echo "[train_score] env=${ENV_NAME} gpus=${GPUS} seed=${SEED} protocol=${STAGE2_PROTOCOL:-pairs_180k}"
echo "[train_score] model=${MODEL}"
echo "[train_score] pairs=${PAIRS_JSON}"
echo "[train_score] image_folder=${ROOT_DATA}"
echo "[train_score] output=${OUTPUT}"

conda run --no-capture-output -n "${ENV_NAME}" deepspeed --include "localhost:${GPUS}" --master_port "${MASTER_PORT:-6689}" src/train/train_deqa.py \
    --deepspeed "${DS_CONFIG}" \
    --model_id "${MODEL}" \
    --dataset_type pair \
    --level_prefix "${LEVEL_PREFIX}" \
    --level_names "${LEVEL_NAMES[@]}" \
    --softkl_loss "${SOFTKL_LOSS}" \
    --weight_rank "${WEIGHT_RANK}" \
    --weight_softkl "${WEIGHT_SOFTKL}" \
    --weight_next_token "${WEIGHT_NEXT_TOKEN}" \
    --continuous_rating_loss "${CONTINUOUS_RATING_LOSS}" \
    --closeset_rating_loss "${CLOSESET_RATING_LOSS}" \
    --use_fix_std "${USE_FIX_STD}" \
    --detach_pred_std "${DETACH_PRED_STD}" \
    --auto_resume "${AUTO_RESUME}" \
    --data_paths "${PAIRS_JSON}" \
    --data_weights 1 \
    --image_folder "${ROOT_DATA}/" \
    --output_dir "${OUTPUT}" \
    --image_min_pixels 196608 \
    --image_max_pixels 1310720 \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --use_liger_kernel False \
    --disable_flash_attn2 True \
    --bf16 True \
    --fp16 False \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --eval_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --merger_lr "${MERGER_LR}" \
    --vision_lr "${VISION_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --prebuilt_pairs "${PREBUILT_PAIRS}" \
    --seed "${SEED}" \
    ${TRAIN_EXTRA_ARGS}
