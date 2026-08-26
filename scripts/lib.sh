#!/bin/bash
# Shared helpers for train/eval launchers. Requires PKG_ROOT to be set.

coinstruct_load_secrets() {
    local secrets_env="${PKG_ROOT}/secrets/.env"
    if [ -f "$secrets_env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$secrets_env"
        set +a
    fi
}

coinstruct_load_env() {
    coinstruct_load_secrets
    if [ -f "${PKG_ROOT}/configs/env.local.sh" ]; then
        # shellcheck disable=SC1091
        source "${PKG_ROOT}/configs/env.local.sh"
    fi
}

coinstruct_has_json() {
    local dir="$1"
    [ -d "$dir" ] && compgen -G "${dir}/*.json" >/dev/null
}

# Resolve ANNOTATIONS_DIR / ensure pairs exist. Downloads if needed.
coinstruct_ensure_annotations() {
    if [ -z "${ANNOTATIONS_DIR:-}" ]; then
        if coinstruct_has_json "${PKG_ROOT}/data/annotations"; then
            ANNOTATIONS_DIR="${PKG_ROOT}/data/annotations"
        elif coinstruct_has_json "${PKG_ROOT}/training_jsons"; then
            ANNOTATIONS_DIR="${PKG_ROOT}/training_jsons"
        fi
    fi

    local need_pairs=0
    if [ ! -f "${PAIRS_JSON:-${PKG_ROOT}/pairs/pairs_180k.json}" ]; then
        need_pairs=1
    fi

    if ! coinstruct_has_json "${ANNOTATIONS_DIR:-/nonexistent}" || [ "$need_pairs" -eq 1 ]; then
        echo "[info] annotations/pairs missing; running scripts/download_annotations.sh ..."
        bash "${PKG_ROOT}/scripts/download_annotations.sh"
        if [ -z "${ANNOTATIONS_DIR:-}" ] || ! coinstruct_has_json "${ANNOTATIONS_DIR}"; then
            if coinstruct_has_json "${PKG_ROOT}/data/annotations"; then
                ANNOTATIONS_DIR="${PKG_ROOT}/data/annotations"
            elif coinstruct_has_json "${PKG_ROOT}/training_jsons"; then
                ANNOTATIONS_DIR="${PKG_ROOT}/training_jsons"
            fi
        fi
    fi

    if ! coinstruct_has_json "${ANNOTATIONS_DIR:-/nonexistent}"; then
        echo "[error] No Stage-1 annotation JSON found." >&2
        echo "[error] Place JSON under data/annotations/ or training_jsons/, or set HF_DATASET_REPO / ANNOTATIONS_DIR." >&2
        echo "[error] See docs/DATA.md" >&2
        return 1
    fi
    export ANNOTATIONS_DIR
    PAIRS_JSON=${PAIRS_JSON:-${PKG_ROOT}/pairs/pairs_180k.json}
    export PAIRS_JSON
}

# True if MODEL_NAME is a local directory with config.json, or a Hub id (Org/Name).
coinstruct_model_ok() {
    local m="$1"
    if [ -d "$m" ]; then
        [ -f "$m/config.json" ]
    else
        # Hugging Face repo id or cache path handled by transformers
        [[ "$m" == */* ]]
    fi
}
