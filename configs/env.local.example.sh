#!/bin/bash
# Machine-local example (lab). Copy once — launchers auto-source env.local.sh:
#   cp configs/env.local.example.sh configs/env.local.sh
# Then: bash scripts/finetune_stage1.sh 0,1,2,3

export MODEL_NAME=/home/zhw/IQA/Model/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct
export IMAGE_FOLDER=/mnt/nvme/zhw/Co-Instruct-plus
export ANNOTATIONS_DIR=/mnt/nvme/zhw/Co-Instruct-Plus/training_jsons
export DATA_PATH=/mnt/nvme/zhw/Co-Instruct-Plus/configs/qsit_multi.yaml
export OUTPUT_DIR=/mnt/nvme/zhw/Co-Instruct-Plus/checkpoints/stage1_qsit_sft

export MODEL=/mnt/nvme/zhw/Co-Instruct-Plus/checkpoints/stage1_qsit_sft
export ROOT_DATA=/mnt/nvme/zhw/Co-Instruct++
export PAIRS_JSON=/mnt/nvme/zhw/Co-Instruct-Plus/pairs/pairs_180k.json
export OUTPUT=/mnt/nvme/zhw/Co-Instruct-Plus/checkpoints/stage2_score

# KonIQ-only Stage 2 (scripts/train_score_stage2_koniq.sh ignores PAIRS_JSON/OUTPUT above)
# export KONIQ_META=/mnt/nvme/zhw/Co-Instruct++/koniq10k/metas/train_koniq_7k.json
# export KONIQ_OUTPUT=/mnt/nvme/zhw/Co-Instruct-Plus/checkpoints/stage2_score_koniq

export ENV_NAME=train
export COINSTRUCT_SEED=42
export PYTHONUNBUFFERED=1
