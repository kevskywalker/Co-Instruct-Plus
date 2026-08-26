# Training

## Environment

```bash
pip install -r requirements.txt
# Optional: pip install -e .
```

Transformers 4.57.0 or newer is required for the Qwen3-VL model classes used
by the training and evaluation entry points.

Set only the paths that are not in-repo:

```bash
export IMAGE_FOLDER=/path/to/Co-Instruct-plus   # Stage 1 (parent of data/)
export ROOT_DATA=/path/to/pair-images           # Stage 2 / eval
```

Optional durable overrides: `cp configs/env.local.example.sh configs/env.local.sh` (auto-sourced by launchers).

Annotations default to `data/annotations/` or `training_jsons/`; if missing, training runs `scripts/download_annotations.sh` automatically. See [DATA.md](DATA.md).

## Stage 1 — SFT (`qsit_multi`)

```bash
bash scripts/finetune_stage1.sh 0,1,2,3
```

- Entry: `qwen/src/train/train_sft.py`
- Data config: `configs/qsit_multi.yaml`; hyperparameters: `configs/stage1_sft.yaml`
- DeepSpeed: ZeRO-2 by default
- Model: `$MODEL_NAME` (default `Qwen/Qwen3-VL-8B-Instruct`)
- Output: `$OUTPUT_DIR` (default `checkpoints/stage1_qsit_sft/`)
- Seed: `$COINSTRUCT_SEED` (default 42)

## Stage 2 — Score / DeQA

Default (185K prebuilt pairs from five IQA datasets):

```bash
bash scripts/train_score_stage2.sh 0,1,2,3
```

- Entry: `qwen/src/train/train_deqa.py`
- Model: `$MODEL` (default Stage-1 output dir)
- Data: `$PAIRS_JSON` (default `pairs/pairs_180k.json`)
- Images: `$ROOT_DATA`
- Hyperparameters: `configs/stage2_score.yaml`
- Output: `$OUTPUT` (default `checkpoints/stage2_score/`)

KonIQ-only paper protocol (online pairing on the official ~7046-image train split):

```bash
bash scripts/gen_soft_label_koniq.sh            # skip if train_koniq_7k.json already exists
bash scripts/train_score_stage2_koniq.sh 0,1,2,3
```

- Data: `$KONIQ_META` (default `$ROOT_DATA/koniq10k/metas/train_koniq_7k.json`)
- Generate that JSON from MOS + split via [`scripts/gen_soft_label_koniq.sh`](../scripts/gen_soft_label_koniq.sh) (DeQA Gaussian `level_probs`, not paper Eq. 9). See [DATA.md](DATA.md).
- `prebuilt_pairs: false` — `PairDataset` samples a partner image each step
- Hyperparameters: `configs/stage2_score_koniq.yaml`
- Output: `$KONIQ_OUTPUT` (default `checkpoints/stage2_score_koniq/`)
- `MODEL` still defaults to the Stage-1 checkpoint. To start from base Qwen3-VL (original lab `train_qwen_koniq.sh`): `MODEL=/path/to/Qwen3-VL-8B-Instruct`

## Hyperparameters

Canonical knobs live in:

- [`configs/stage1_sft.yaml`](../configs/stage1_sft.yaml)
- [`configs/stage2_score.yaml`](../configs/stage2_score.yaml)
- [`configs/stage2_score_koniq.yaml`](../configs/stage2_score_koniq.yaml) (KonIQ-only Stage 2)

Launchers read the stage YAML files. Override individual values with the
corresponding environment variable; command-line launcher arguments take
precedence over both environment variables and YAML values.

## Reproducibility tips

- Record `git rev-parse HEAD` and the full env exports with each run.
- Prefer a fixed `COINSTRUCT_SEED`.
- Disable FlashAttention in scripts for portability (`--disable_flash_attn2 True`).
