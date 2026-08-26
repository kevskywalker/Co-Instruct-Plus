# Co-Instruct-Plus

Two-stage training for Qwen3-VL quality understanding:

1. **Stage 1 — SFT** on mixed M2C + T2C instruction data (`qsit_multi`)
2. **Stage 2 — Score** DeQA-style pair training on the Stage-1 checkpoint

This repository is packaged for reproducible open-source release (code under Apache-2.0). Large annotation JSON files and images are **not** shipped in git — see [docs/DATA.md](docs/DATA.md).

## Quickstart

```bash
git clone <this-repo> Co-Instruct-Plus
cd Co-Instruct-Plus

pip install -r requirements.txt   # install a matching PyTorch build first

# Stage 1 — only the image root is required
export IMAGE_FOLDER=/path/to/Co-Instruct-plus   # must contain data/
bash scripts/finetune_stage1.sh 0,1,2,3

# Stage 2 — pair image root + Stage-1 checkpoint (default path used if present)
export ROOT_DATA=/path/to/pair-images
bash scripts/train_score_stage2.sh 0,1,2,3
# Optional paper §3.2 KonIQ-only Stage 2:
# bash scripts/gen_soft_label_koniq.sh          # if train_koniq_7k.json is missing
# bash scripts/train_score_stage2_koniq.sh 0,1,2,3

# Eval
export MODEL_PATH=checkpoints/stage2_score
export ROOT_DIR=$ROOT_DATA
bash scripts/eval_iqa.sh
```

Defaults (override with `export` or optional `configs/env.local.sh`):

| Item | Default |
|------|---------|
| Base model | `Qwen/Qwen3-VL-8B-Instruct` (Hub) |
| Annotations | `data/annotations/` (portable) or `training_jsons/` (lab) |
| Stage-1 out | `checkpoints/stage1_qsit_sft/` |
| Stage-2 out | `checkpoints/stage2_score/` |

**Lab vs upload:** keep absolute-path JSON in `training_jsons/` locally; upload only `data/annotations/` + `pairs/pairs_180k.json` (relative paths). See [docs/DATA.md](docs/DATA.md).

Lab shortcut: `cp configs/env.local.example.sh configs/env.local.sh` once; launchers auto-load it.

More detail: [docs/TRAIN.md](docs/TRAIN.md), [docs/EVAL.md](docs/EVAL.md), [docs/DATA.md](docs/DATA.md).

## Layout

```text
Co-Instruct-Plus/
├── configs/                 # yaml + deepspeed + env.local.example.sh
├── scripts/                 # download / train / eval / release checks
├── qwen/src/                # training + eval code
├── data/annotations/        # downloaded JSON (gitignored)
├── pairs/                   # pairs_180k.json (gitignored)
├── docs/
└── checkpoints/             # train outputs (gitignored)
```

## Requirements

- NVIDIA GPU(s), CUDA-compatible PyTorch
- Conda env name defaults to `train` (`ENV_NAME`) for DeepSpeed launchers
- Base weights: Hub id above, or a local `MODEL_NAME` directory

## Paper pipeline

**Supported:** `scripts/finetune_stage1.sh`, `scripts/train_score_stage2.sh`, `scripts/gen_soft_label_koniq.sh`, `scripts/train_score_stage2_koniq.sh`, `scripts/eval_iqa.sh`
(Stage-1 SFT + Stage-2 DeQA score on 185K pairs or KonIQ-only online pairing + IQA eval).

## Citation

```bibtex
@misc{co-instruct-plus,
  title        = {Co-Instruct-Plus},
  author       = {Co-Instruct-Plus Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/}},
  note         = {Update with paper venue / DOI when available}
}
```

Also see [`CITATION.cff`](CITATION.cff).

## License

Code: [Apache License 2.0](LICENSE). Third-party models and datasets remain under their own terms ([docs/DATA.md](docs/DATA.md)).

## Maintainer notes

- Release gate: `bash scripts/check_release.sh` and [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md)
