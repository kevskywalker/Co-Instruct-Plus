# Evaluation

Scoring-only benchmark: single-image quality assessment + SRCC / PLCC.

## Prerequisites

- Trained checkpoint (`MODEL_PATH`) with `*.safetensors`
- Benchmark root (`ROOT_DIR`) containing metas such as `KONIQ/metas/test_koniq_2k.json` and images
- Optional remaps: `KONIQ_IMAGES_DIR`, `SPAQ_IMAGES_DIR`
- Optional preprocessor: `PREPROCESSOR_PATH` (base Qwen3-VL folder if checkpoint config is incomplete)

## Run

```bash
export MODEL_PATH=checkpoints/stage2_score
export ROOT_DIR=/path/to/benchmark-or-pair-images
export SAVE_DIR=$PWD/results/stage2_score_eval
# optional: export KONIQ_IMAGES_DIR=$ROOT_DIR/koniq10k/512x384

bash scripts/eval_iqa.sh
```

If `configs/env.local.sh` exists it is auto-sourced (`MODEL` / `ROOT_DATA` map to `MODEL_PATH` / `ROOT_DIR`).

KonIQ-only smoke:

```bash
META_PATHS="$ROOT_DIR/KONIQ/metas/test_koniq_2k.json" bash scripts/eval_iqa.sh
```

## Outputs

- Predictions: `$SAVE_DIR/<meta_basename>.json` (JSONL)
- Metrics: printed SRCC / PLCC per dataset via `qwen/src/evaluate/cal_plcc_srcc.py`

## Implementation

- Inference: [`qwen/src/evaluate/iqa_eval_qwen.py`](../qwen/src/evaluate/iqa_eval_qwen.py)
- Metrics: [`qwen/src/evaluate/cal_plcc_srcc.py`](../qwen/src/evaluate/cal_plcc_srcc.py)
