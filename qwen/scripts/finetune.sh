#!/bin/bash
set -euo pipefail

# Stage 1: SFT on qsit_multi (M2C + T2C mixed).
# Usage (from repo root):
#   export IMAGE_FOLDER=/path/to/Co-Instruct-plus   # required: parent of data/
#   bash scripts/finetune_stage1.sh 0,1,2,3
#
# Optional: configs/env.local.sh (auto-sourced) for durable path overrides.

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

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}
DATA_PATH=${DATA_PATH:-${PKG_ROOT}/configs/qsit_multi.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-${PKG_ROOT}/checkpoints/stage1_qsit_sft}
OUTPUT_DIR=${OUTPUT_DIR%/}
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_PATH=${STAGE1_CONFIG:-${PKG_ROOT}/configs/stage1_sft.yaml}

yaml_value() {
    "$PYTHON_BIN" "${PKG_ROOT}/scripts/yaml_value.py" "$CONFIG_PATH" "$1" "$2"
}

if [ -z "${IMAGE_FOLDER:-}" ]; then
    echo "[error] IMAGE_FOLDER is not set (Stage-1 image root; must contain data/)." >&2
    echo "[error] Example: export IMAGE_FOLDER=/path/to/Co-Instruct-plus" >&2
    echo "[error] Optional: put lasting overrides in configs/env.local.sh (see configs/env.example)." >&2
    exit 1
fi

coinstruct_ensure_annotations

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$(yaml_value global_batch_size 128)}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-$(yaml_value batch_per_device 4)}
GPUS=${1:-0,1,2,3}
NUM_DEVICES=$(echo "$GPUS" | awk -F',' '{print NF}')
if [ "$NUM_DEVICES" -le 0 ] || [ "$BATCH_PER_DEVICE" -le 0 ] || [ "$GLOBAL_BATCH_SIZE" -le 0 ]; then
    echo "[error] global_batch_size, batch_per_device, and GPU count must be positive." >&2
    exit 1
fi
if [ $((GLOBAL_BATCH_SIZE % (BATCH_PER_DEVICE * NUM_DEVICES))) -ne 0 ]; then
    echo "[error] global_batch_size must be divisible by batch_per_device * num_devices." >&2
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
ENV_NAME=${ENV_NAME:-train}
DS_CONFIG=${DS_CONFIG:-$(yaml_value deepspeed "${PKG_ROOT}/configs/deepspeed/zero2.json")}
if [ "${DS_CONFIG#/}" = "${DS_CONFIG}" ]; then
    DS_CONFIG="${PKG_ROOT}/${DS_CONFIG}"
fi
if [ ! -f "$DS_CONFIG" ]; then
    echo "[error] DeepSpeed config not found: $DS_CONFIG" >&2
    exit 1
fi
SEED=${COINSTRUCT_SEED:-42}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-$(yaml_value num_train_epochs 1)}
LEARNING_RATE=${LEARNING_RATE:-$(yaml_value learning_rate 1e-5)}
MERGER_LR=${MERGER_LR:-$(yaml_value merger_lr 1e-5)}
VISION_LR=${VISION_LR:-$(yaml_value vision_lr 2e-6)}
WEIGHT_DECAY=${WEIGHT_DECAY:-$(yaml_value weight_decay 0.1)}
WARMUP_RATIO=${WARMUP_RATIO:-$(yaml_value warmup_ratio 0.03)}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-$(yaml_value lr_scheduler_type cosine)}
LOGGING_STEPS=${LOGGING_STEPS:-$(yaml_value logging_steps 1)}
SAVE_STRATEGY=${SAVE_STRATEGY:-$(yaml_value save_strategy steps)}
SAVE_STEPS=${SAVE_STEPS:-$(yaml_value save_steps 1000)}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-$(yaml_value save_total_limit 1)}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-$(yaml_value dataloader_num_workers 8)}
IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-$(yaml_value image_min_pixels 524288)}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-$(yaml_value image_max_pixels 1310720)}
MASTER_PORT=${MASTER_PORT:-$(yaml_value master_port 6690)}

if ! coinstruct_model_ok "$MODEL_NAME"; then
    echo "[error] MODEL_NAME must be a local checkpoint dir (with config.json) or a Hub id like Qwen/Qwen3-VL-8B-Instruct." >&2
    echo "[error] Got: $MODEL_NAME" >&2
    exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
    echo "[error] Missing data config: $DATA_PATH" >&2
    exit 1
fi
if [ ! -d "$IMAGE_FOLDER/data" ]; then
    echo "[error] IMAGE_FOLDER must contain a data/ subdir (got: $IMAGE_FOLDER)" >&2
    echo "[error] Expected images like: $IMAGE_FOLDER/data/<filename>" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[finetune stage1] env=${ENV_NAME} gpus=${GPUS} num_devices=${NUM_DEVICES} seed=${SEED}"
echo "[finetune stage1] model=${MODEL_NAME}"
echo "[finetune stage1] data=${DATA_PATH}"
echo "[finetune stage1] annotations=${ANNOTATIONS_DIR}"
echo "[finetune stage1] image_folder=${IMAGE_FOLDER}"
echo "[finetune stage1] output=${OUTPUT_DIR}"
echo "[finetune stage1] batch_per_device=${BATCH_PER_DEVICE} grad_accum=${GRAD_ACCUM_STEPS} global_batch=$((BATCH_PER_DEVICE * NUM_DEVICES * GRAD_ACCUM_STEPS))"

conda run --no-capture-output -n "${ENV_NAME}" deepspeed --include "localhost:${GPUS}" --master_port "${MASTER_PORT:-6690}" src/train/train_sft.py \
    --use_liger_kernel False \
    --deepspeed "${DS_CONFIG}" \
    --model_id "${MODEL_NAME}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --image_min_pixels "${IMAGE_MIN_PIXELS}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --learning_rate "${LEARNING_RATE}" \
    --merger_lr "${MERGER_LR}" \
    --vision_lr "${VISION_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps "${LOGGING_STEPS}" \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --seed "${SEED}"
