#!/bin/bash
# Build KonIQ-only DeQA soft labels (train_koniq_7k.json) from MOS + split.
# Usage (from repo root):
#   export ROOT_DATA=/path/to/pair-or-benchmark-images
#   bash scripts/gen_soft_label_koniq.sh
#   bash scripts/gen_soft_label_koniq.sh --overwrite-test   # also replace test_koniq_2k.json
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PKG_ROOT="$ROOT"
# shellcheck disable=SC1091
source "${PKG_ROOT}/scripts/lib.sh"
coinstruct_load_env

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_PATH=${SOFT_LABEL_CONFIG:-${PKG_ROOT}/configs/soft_labels_koniq.json}

if [ -z "${ROOT_DATA:-}" ]; then
    echo "[error] ROOT_DATA is not set (KonIQ MOS/split/image root)." >&2
    echo "[error] Example: export ROOT_DATA=/path/to/Co-Instruct++" >&2
    exit 1
fi
ROOT_DATA=${ROOT_DATA%/}

if [ ! -d "$ROOT_DATA" ]; then
    echo "[error] ROOT_DATA is not a directory: $ROOT_DATA" >&2
    exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[error] Missing soft-label config: $CONFIG_PATH" >&2
    exit 1
fi

MOS_JSON=${KONIQ_MOS_JSON:-${ROOT_DATA}/KONIQ/metas/mos.json}
SPLIT_JSON=${KONIQ_SPLIT_JSON:-${ROOT_DATA}/KONIQ/metas/split.json}
IMAGES_DIR=${KONIQ_IMAGES_DIR:-${ROOT_DATA}/koniq10k/512x384}

if [ ! -f "$MOS_JSON" ]; then
    echo "[error] Missing MOS JSON: $MOS_JSON" >&2
    echo "[error] Place official KonIQ mos.json under ROOT_DATA/KONIQ/metas/." >&2
    exit 1
fi
if [ ! -f "$SPLIT_JSON" ]; then
    echo "[error] Missing split JSON: $SPLIT_JSON" >&2
    echo "[error] Place official KonIQ split.json under ROOT_DATA/KONIQ/metas/." >&2
    exit 1
fi
if [ ! -d "$IMAGES_DIR" ]; then
    echo "[error] Missing KonIQ images dir: $IMAGES_DIR" >&2
    echo "[error] Expected ROOT_DATA/koniq10k/512x384/ (or set KONIQ_IMAGES_DIR)." >&2
    exit 1
fi

EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

echo "[gen_soft_label] root_data=${ROOT_DATA}"
echo "[gen_soft_label] config=${CONFIG_PATH}"
echo "[gen_soft_label] mos=${MOS_JSON}"
echo "[gen_soft_label] split=${SPLIT_JSON}"
echo "[gen_soft_label] images=${IMAGES_DIR}"

"$PYTHON_BIN" "${PKG_ROOT}/scripts/gen_soft_label.py" \
    --config "$CONFIG_PATH" \
    --root-data "$ROOT_DATA" \
    --datasets koniq \
    "${EXTRA_ARGS[@]}"

echo "[gen_soft_label] train meta: ${ROOT_DATA}/koniq10k/metas/train_koniq_7k.json"
if [[ " ${*:-} " == *" --overwrite-test "* ]]; then
    echo "[gen_soft_label] test meta overwritten: ${ROOT_DATA}/KONIQ/metas/test_koniq_2k.json"
else
    echo "[gen_soft_label] test meta: ${ROOT_DATA}/KONIQ/metas/test_koniq_2k.generated.json"
    echo "[gen_soft_label] eval still uses KONIQ/metas/test_koniq_2k.json unless --overwrite-test"
fi
