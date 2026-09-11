# Co-Instruct-Plus

Two-stage training and evaluation code for visual quality understanding built
on `Qwen/Qwen3-VL-8B-Instruct`.

## Installation

```bash
git clone https://github.com/kevskywalker/Co-Instruct-Plus.git
cd Co-Instruct-Plus
pip install -r requirements.txt
```

Use a CUDA-compatible PyTorch build and a conda environment containing
DeepSpeed. The launchers use the `train` environment by default; override it
with `ENV_NAME` when needed.

## Data

All release data is hosted at
[`yluo016/co-instruct-plus`](https://huggingface.co/datasets/yluo016/co-instruct-plus).
Download the Stage-I annotations/images and Stage-II KonIQ images/metadata from
that dataset, then set:

```bash
export IMAGE_FOLDER=/path/to/stage1-images   # contains data/
export ROOT_DATA=/path/to/koniq-and-iqa-data
```

## Stage I

```bash
# Train
bash scripts/train_stage1.sh 0,1,2,3

# Test
MODEL_PATH=checkpoints/stage1_qsit_sft \
  ROOT_DIR="$ROOT_DATA" bash scripts/test_stage1.sh
```

## Stage II

Stage II starts from the Stage-I checkpoint and uses the KonIQ training split.

```bash
# Create KonIQ training metadata once if it is absent
bash scripts/gen_soft_label_koniq.sh

# Train
STAGE2_MODEL=checkpoints/stage1_qsit_sft \
  bash scripts/train_stage2.sh 0,1,2,3

# Test
MODEL_PATH=checkpoints/stage2_score_koniq \
  ROOT_DIR="$ROOT_DATA" bash scripts/test_stage2.sh
```

Checkpoints, datasets, predictions, and logs are local outputs and are not
part of this Git repository.

## MICBench-v2

Scalar-IQA testing remains available through `test_stage1.sh` and
`test_stage2.sh`. To test either checkpoint on MICBench-v2, download and
extract `micbench/micbench-test.zip` from the release dataset, then run:

```bash
export MICBENCH_ANNOTATION=/path/to/micbench_v2_1998.json
export MICBENCH_IMAGE_ROOT=/path/to/MICBench_test/images

MODEL_PATH=checkpoints/stage1_qsit_sft bash scripts/test_micbench_v2.sh

# Or evaluate the Stage-II checkpoint.
MODEL_PATH=checkpoints/stage2_score_koniq bash scripts/test_micbench_v2.sh
```

## License

Code is released under [Apache-2.0](LICENSE). Models and datasets retain their
respective licenses.
