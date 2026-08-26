#!/bin/bash
set -euo pipefail

# Release gate: fail if machine-specific paths appear as *defaults* in public launchers/docs.
# Remap tables in path_utils / sanitize_annotations are allowed (they strip legacy prefixes).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "[check] scanning launchers/docs for machine-specific absolute path defaults..."
PATTERN='/home/zhw|/mnt/nvme|/home/yuhan'
EXCLUDE=(
  --glob '!qwen/src/dataset/path_utils.py'
  --glob '!scripts/sanitize_annotations.py'
  --glob '!scripts/check_release.sh'
  --glob '!docs/RELEASE_CHECKLIST.md'
  --glob '!configs/env.local.sh'
  --glob '!configs/env.local.example.sh'
)

SCAN_PATHS=(
  README.md LICENSE MODEL_CARD.md CITATION.cff
  configs/env.example configs/qsit_multi.yaml configs/stage1_sft.yaml configs/stage2_score.yaml configs/stage2_score_koniq.yaml
  configs/soft_labels_koniq.json
  configs/deepspeed
  qwen/scripts/finetune.sh qwen/scripts/train_score.sh qwen/scripts/train_qwen_koniq.sh
  scripts/finetune_stage1.sh scripts/train_score_stage2.sh scripts/train_score_stage2_koniq.sh scripts/eval_iqa.sh
  scripts/gen_soft_label_koniq.sh scripts/gen_soft_label.py
  scripts/download_annotations.sh scripts/lib.sh
  docs/DATA.md docs/TRAIN.md docs/EVAL.md
  pyproject.toml requirements.txt
  qwen/src
)

missing=()
for path in "${SCAN_PATHS[@]}"; do
  if [ ! -e "$path" ]; then
    missing+=("$path")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "[error] Missing release scan paths:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

HITS=$(rg -n "$PATTERN" "${EXCLUDE[@]}" "${SCAN_PATHS[@]}" || true)
if [ -n "$HITS" ]; then
    echo "[error] Found machine-specific paths in public defaults/docs:" >&2
    echo "$HITS" >&2
    exit 1
fi
echo "[check] path scan OK"

echo "[check] scanning sanitized annotations for absolute machine paths..."
ANNOTATION_HITS=$(rg -n '/home/|/mnt/' data/annotations --glob '*.json' 2>/dev/null || true)
if [ -n "$ANNOTATION_HITS" ]; then
    echo "[error] Found absolute machine paths in data/annotations:" >&2
    echo "$ANNOTATION_HITS" >&2
    exit 1
fi
echo "[check] annotation path scan OK"

echo "[check] scanning for files > 95MB that should not be committed..."
LARGE=$(find . -type f \
    -not -path './.git/*' \
    -not -path './training_jsons/*' \
    -not -path './pairs/*' \
    -not -path './data/annotations/*' \
    -not -path './checkpoints/*' \
    -not -path './.hf_annotations/*' \
    -size +95M 2>/dev/null || true)
if [ -n "$LARGE" ]; then
    echo "[error] Oversized files outside gitignored data dirs:" >&2
    echo "$LARGE" >&2
    exit 1
fi
echo "[check] size scan OK"
echo "[check] release gate passed"
