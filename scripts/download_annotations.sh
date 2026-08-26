#!/bin/bash
set -euo pipefail

# Download Stage-1/2 annotation JSON into data/annotations and pairs/.
# Set HF_DATASET_REPO to your Hugging Face dataset id after upload, e.g.:
#   export HF_DATASET_REPO=your-org/co-instruct-plus-annotations
#
# Until the public dataset is published, this script can also symlink/copy from
# a local lab directory:
#   LOCAL_ANNOTATIONS_DIR=/path/to/training_jsons LOCAL_PAIRS_DIR=/path/to/pairs \
#     bash scripts/download_annotations.sh

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "${ROOT}/secrets/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/secrets/.env"
    set +a
fi
OUT_ANN=${ANNOTATIONS_OUT:-${ROOT}/data/annotations}
OUT_PAIRS=${PAIRS_OUT:-${ROOT}/pairs}
mkdir -p "$OUT_ANN" "$OUT_PAIRS"

if [ -n "${HF_DATASET_REPO:-}" ]; then
    if ! command -v huggingface-cli >/dev/null 2>&1 && ! command -v hf >/dev/null 2>&1; then
        echo "[error] Install huggingface_hub CLI: pip install -U huggingface_hub" >&2
        exit 1
    fi
    echo "[download] fetching ${HF_DATASET_REPO} ..."
    if command -v hf >/dev/null 2>&1; then
        hf download "${HF_DATASET_REPO}" --repo-type dataset --local-dir "${ROOT}/.hf_annotations"
    else
        huggingface-cli download "${HF_DATASET_REPO}" --repo-type dataset --local-dir "${ROOT}/.hf_annotations"
    fi
    # Prefer nested annotations/ + pairs/ layout if present
    if [ -d "${ROOT}/.hf_annotations/annotations" ]; then
        cp -n "${ROOT}/.hf_annotations/annotations/"*.json "$OUT_ANN/" 2>/dev/null || true
    else
        cp -n "${ROOT}/.hf_annotations/"*.json "$OUT_ANN/" 2>/dev/null || true
    fi
    if [ -f "${ROOT}/.hf_annotations/pairs/pairs_180k.json" ]; then
        cp -n "${ROOT}/.hf_annotations/pairs/pairs_180k.json" "$OUT_PAIRS/"
    elif [ -f "${ROOT}/.hf_annotations/pairs_180k.json" ]; then
        cp -n "${ROOT}/.hf_annotations/pairs_180k.json" "$OUT_PAIRS/"
    fi
elif [ -n "${LOCAL_ANNOTATIONS_DIR:-}" ]; then
    echo "[download] copying local annotations from ${LOCAL_ANNOTATIONS_DIR}"
    cp -n "${LOCAL_ANNOTATIONS_DIR}/"*.json "$OUT_ANN/" 2>/dev/null || true
    if [ -n "${LOCAL_PAIRS_DIR:-}" ] && [ -f "${LOCAL_PAIRS_DIR}/pairs_180k.json" ]; then
        cp -n "${LOCAL_PAIRS_DIR}/pairs_180k.json" "$OUT_PAIRS/"
    elif [ -f "${ROOT}/pairs/pairs_180k.json" ]; then
        echo "[download] pairs already present"
    fi
else
    # Fallback: use in-repo training_jsons / pairs if present (lab checkout)
    if ls "${ROOT}/training_jsons/"*.json >/dev/null 2>&1; then
        echo "[download] using existing training_jsons/ (lab layout)"
        export ANNOTATIONS_DIR="${ROOT}/training_jsons"
        echo "ANNOTATIONS_DIR=${ANNOTATIONS_DIR}"
        if [ -f "${ROOT}/pairs/pairs_180k.json" ]; then
            echo "PAIRS_JSON=${ROOT}/pairs/pairs_180k.json"
        fi
        exit 0
    fi
    cat >&2 <<'EOF'
[error] No annotation source configured.

Options:
  1) export HF_DATASET_REPO=org/co-instruct-plus-annotations
  2) LOCAL_ANNOTATIONS_DIR=... LOCAL_PAIRS_DIR=... bash scripts/download_annotations.sh
  3) Place JSON under training_jsons/ and pairs/pairs_180k.json

See docs/DATA.md
EOF
    exit 1
fi

echo "[download] annotations -> ${OUT_ANN}"
echo "[download] pairs       -> ${OUT_PAIRS}"
echo "export ANNOTATIONS_DIR=${OUT_ANN}"
echo "export PAIRS_JSON=${OUT_PAIRS}/pairs_180k.json"
echo "export DATA_PATH=${ROOT}/configs/qsit_multi.yaml"
