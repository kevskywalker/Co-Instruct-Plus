# Data preparation

## Directory roles

| Variable | Meaning | Default |
|----------|---------|---------|
| `IMAGE_FOLDER` | Stage-1 image root. Paths like `./data/foo.jpg` → `$IMAGE_FOLDER/data/foo.jpg`. | **required** |
| `ANNOTATIONS_DIR` | Stage-1 JSON lists (`m2c_*.json`, `t2c_*.json`, …). | `data/annotations/` or `training_jsons/` |
| `DATA_PATH` | YAML mix config. | [`configs/qsit_multi.yaml`](../configs/qsit_multi.yaml) |
| `ROOT_DATA` | Stage-2 pair / eval image root. | **required for Stage 2 / eval** |
| `PAIRS_JSON` | Path to `pairs_180k.json`. | `pairs/pairs_180k.json` |
| `KONIQ_META` | KonIQ-only Stage-2 image list. | `$ROOT_DATA/koniq10k/metas/train_koniq_7k.json` |
| `KONIQ_OUTPUT` | KonIQ-only Stage-2 checkpoint dir. | `checkpoints/stage2_score_koniq/` |
| `KONIQ_IMAGES_DIR` / `SPAQ_IMAGES_DIR` | Optional eval remaps; KonIQ-only training also defaults `KONIQ_IMAGES_DIR` to `$ROOT_DATA/koniq10k/512x384`. | unset |

## Expected layout

```text
$IMAGE_FOLDER/
  data/                  # Stage-1 relative images (./data/...)
$ANNOTATIONS_DIR/        # data/annotations/ or training_jsons/
  m2c_*.json
  t2c_*.json
  coinstruct_562k_t2c.json
$ROOT_DATA/
  KONIQ/metas/mos.json                 # official MOS + std (not in git)
  KONIQ/metas/split.json               # official train/test split (not in git)
  KONIQ/metas/test_koniq_2k.json       # eval meta (do not overwrite by default)
  koniq10k/512x384/...                 # as referenced by pairs_180k.json / train_koniq_7k.json
  koniq10k/metas/train_koniq_7k.json   # KonIQ-only Stage 2 (generated)
pairs/
  pairs_180k.json
```

## KonIQ-only Stage 2 meta (generate locally)

Do **not** commit MOS, split, or images. From MOS + split under `$ROOT_DATA`, build the DeQA Gaussian `level_probs` list that `train_score_stage2_koniq.sh` reads. This is DeQA's Gaussian pdf + binary fallback, **not** paper Eq. (9) triangle interpolation. `PairDataset` rebuilds pair prompts; it only consumes `image`, `gt_score`, `std`, and `level_probs`.

```bash
export ROOT_DATA=/path/to/pair-or-benchmark-images
bash scripts/gen_soft_label_koniq.sh
```

Required inputs:

- `$ROOT_DATA/KONIQ/metas/mos.json`
- `$ROOT_DATA/KONIQ/metas/split.json`
- `$ROOT_DATA/koniq10k/512x384/` (images; JSON stores relative `koniq10k/512x384/<file>`)

Default outputs:

- `$ROOT_DATA/koniq10k/metas/train_koniq_7k.json` (~7046 train images)
- `$ROOT_DATA/KONIQ/metas/test_koniq_2k.generated.json` (~2010 test images)

Eval continues to use the existing `KONIQ/metas/test_koniq_2k.json`. Pass `--overwrite-test` only if you intend to replace that file.

Then train: `bash scripts/train_score_stage2_koniq.sh 0,1,2,3` (see [TRAIN.md](TRAIN.md)).

## Annotations (lab vs upload)

Large JSON files are **not** stored in git.

| Copy | Path style | Use |
|------|------------|-----|
| `training_jsons/` | May include absolute lab paths | Local training only; **do not upload** |
| `data/annotations/` | Relative (`data/...`) | **Public / HF upload**; default for outsiders |
| `pairs/pairs_180k.json` | Relative (`koniq10k/...`) | Upload as-is |

Lab checkouts can keep using `training_jsons/` via `ANNOTATIONS_DIR` in `configs/env.local.sh`. Runtime remapping in `path_utils.py` still resolves old absolute prefixes when that copy is used.

Training launchers prefer `data/annotations/` if present, else `training_jsons/`, else `scripts/download_annotations.sh`.

```bash
# Outsiders (after you publish):
export HF_DATASET_REPO=your-org/co-instruct-plus-annotations
bash scripts/download_annotations.sh
```

## Sanitize for upload (maintainers)

Keep `training_jsons/` untouched on the lab machine. Write a portable copy:

```bash
python scripts/sanitize_annotations.py \
  --input-dir training_jsons \
  --output-dir data/annotations \
  --image-root "$IMAGE_FOLDER" \
  --check-exists --sample 200
```

Before upload, confirm no absolute roots remain:

```bash
rg '/home/|/mnt/' data/annotations/*.json && echo FAIL || echo OK
```

Upload **`data/annotations/*.json`** and **`pairs/pairs_180k.json`** to Hugging Face / Zenodo — not `training_jsons/`.

## Third-party image / model licenses

You must obtain and comply with licenses for:

- **Qwen3-VL** base weights (Alibaba / model card terms)
- Source IQA / instruction corpora used to build Co-Instruct-style images and captions (e.g. SPAQ, KonIQ, KADIS, and related datasets)

This repository distributes **code** under Apache-2.0; it does not grant rights to third-party datasets or base models.
