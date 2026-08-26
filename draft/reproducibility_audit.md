# Co-Instruct++ 可复现性审计与无改代码执行计划

- 审计日期：2026-08-24
- 性质：**只读审查 + 后续执行计划**。本文不训练、不评测、不改任何源码。
- 约束：不修改 `/home/zhw/IQA/test/{qbench,MICBench,2AFC-LMM}` 的 canonical 脚本；评测一律冻结到 `/mnt/nvme/zhw/eval_runs/<YYYYMMDD>_<exp_name>/`。
- 相关前序文档：`draft/review.md`（2026-08-18，论文 vs 默认代码一致性审查）。本文在其结论上给出**不改代码**的复现路径。
- 密钥：本机存在 `secrets/.env`（启动器会自动 source）。下文只写**是否需要**密钥，**绝不抄写 token 值**。

---

## 1. 执行摘要

**结论：不能按论文字面协议完整复现。本机最多做到「部分复现」。**

仓库已经具备两阶段训练代码、Stage I 标注 JSON、185K pairs、KonIQ-only Stage II 入口、scalar IQA 评测脚本，以及实验室里的图像根目录和若干候选 checkpoint。但论文 `draft/main.pdf` 所声称的实验协议（Co-Instruct-359K、KonIQ 8K pairs、2 epoch / GBS 192、Eq. (9)–(17)、Q-Bench / 2AFC-LMM / MICBench_v2 主表、Table 8 消融、人工标注细节）与**默认发布入口**不一致；论文表格所用的精确 checkpoint hash、MICBench_v2 的 3000 题冻结集、数据构造脚本和 proprietary API 基线均未随本仓库发布。不改代码的前提下，可以：(a) 用现有脚本复现**代码协议**的训练/IQA 分数；(b) 冻结实验室已有评测脚本去**重跑** Q-Bench / MICBench / 2AFC；(c) 从本地或 Hub 拉取候选权重做 eval-only。无法保证得到 Table 3–8 的印刷数字，也无法复现人工研究与数据构造过程。

| 项目 | 裁决 |
|------|------|
| 完整复现论文（训练+数据+全部表格+图+消融+人工） | **否** |
| 不改代码，复现「当前仓库默认协议」 | **部分可行**（缺 Stage 1/2 成品权重、SPAQ 图、公开数据集托管、论文主评测入口） |
| 不改代码，eval-only 对齐印刷数字 | **不确定 / 大概率否**（候选 ckpt 与表格对不上；MICBench 规模 1998≠3000） |
| 从零训练对齐论文 §6.1 | **否**（超参/数据量/损失与论文不同；改 YAML/环境变量也改不了损失公式与 KL 实现） |
| 公开第三方按 README 一键复现 | **否**（GitHub 仍是占位；HF 数据集几乎空；图像不随 git） |

---

## 2. 论文身份

| 项 | 内容 |
|----|------|
| 标题 | Co-Instruct++: Generalizable Image Quality Assessment Via Open-Ended Visual Comparison |
| 作者 | Hanwei Zhu, Yuhan Luo, Baoliang Chen, Xi Zhang, Shiqi Wang, Yuming Fang, Weisi Lin |
| 投稿 | IEEE Transactions on Pattern Analysis and Machine Intelligence（页眉 `SUBMITTED TO IEEE TPAMI`） |
| 本地 PDF | `/mnt/nvme/zhw/Co-Instruct-Plus/draft/main.pdf`（14 页；pdfTeX 时间戳 `2026-08-02`）。**无 LaTeX 源。** |
| 相关 PDF | `draft/You 等 - 2025 - ... Score Distribution_副本.pdf`（DeQA-Score，Stage II 方法来源，不是本文） |
| 一致性审查 | `draft/review.md` |
| arXiv / HF Papers | **未检索到 Co-Instruct++ 的 arXiv ID**（截至 2026-08-24）。前作 ECCV 2024 Oral：*Towards Open-ended Visual Quality Comparison*，[arXiv:2402.16641](https://arxiv.org/abs/2402.16641) |
| 承诺代码仓 | 论文写 `https://github.com/h4nwei/Co-Instruct-Plus`。远程目前几乎只有「dataset will be publicly available soon」。本地 git：`origin=kevskywalker/Co-Instruct-Plus`，`upstream=h4nwei/Co-Instruct-Plus`；**工作区几乎全部未提交**（仅 `README.md` 的占位 commit `81d8811`） |
| 模型名 | Co-Instruct-Assistant（Stage I）、Co-Instruct-Rater（Stage II）；语料名 Co-Instruct-359K；基准 MICBench_v2 |
| 基座 | 论文写 Qwen3-VL-8B；代码默认 `Qwen/Qwen3-VL-8B-Instruct` |
| Hub 权重 | `yluo016/Co-instruct__`（公开，无 model card，内含多个子目录，**未标明哪一个对应论文表格**） |
| Hub 数据 | `yluo016/co-instruct-plus-score` 目前几乎空（仅 `.gitattributes`）。`scripts/download_annotations.sh` 的 `HF_DATASET_REPO` **尚未填真实 ID** |
| 代码协议说明 | `hf_staging/co-instruct-plus/README.md` 明确写「描述的是 **code protocol，不是旧论文计数**」 |

### 论文需要复现的实验清单

摘自 `draft/main.pdf`：

1. **训练配方（§6.1）**：Qwen3-VL-8B 全参微调；动态分辨率 `min=256×32×32`、`max=1280×32×32`；LR `2e-5`；**2 epochs**；**global batch 192**；4× NVIDIA H20。
2. **Stage I 数据（Table 1）**：Merge2Compare + Teach2Compare，共 **351,211** 条（论文对外称 Co-Instruct-359K）；2–6 图；coarse/fine × reasoning/question。
3. **Stage II 数据（§3.2）**：从 KonIQ-10K **训练集采 8K image pairs**；响应为两档绝对质量 +「第二张相对第一张」的相对标签。
4. **主结果**
   - Table 3：Q-BenchPAIR-A1（Assistant **81.94%**）
   - Table 5：Q-BenchSINGLE-A1（Assistant **83.88%**）
   - Table 6：2AFC-LMM（Assistant 平均 **α=0.767, ρ=0.794**）
   - Table 4 + Fig. 5：MICBench_v2（Assistant **84.86%**；3–6 图；coarse/fine）
   - Table 7：scalar IQA，Rater 六库 PLCC/SRCC（KonIQ **0.962/0.950**，KADID **0.744/0.746**，PIPAL **0.557/0.512**，LIVE-Wild **0.910/0.901**，AGIQA-3K **0.812/0.749**，CSIQ **0.758/0.707**，平均 **0.791/0.760**）
5. **消融 Table 8**：Stage I 的 M2C / T2C 开闭 × Stage II KonIQ 开闭，共多行。
6. **对比模型**：开源一串 + proprietary（Qwen3-VL-Plus teacher、Claude-Sonnet-4.5、GPT-5.4、Gemini-2.5-Pro）。
7. **定性图**：Fig. 1–4 为方法示意图 / MICBench 题卡；Fig. 5 为 MICBench 雷达图（可由数字重绘）。
8. **人工研究**：MICBench_v2「comprehensive subjective testing + expert consistency checking」（§5.2）。无 observer 数、界面、伦理说明。

---

## 3. 已经在位的东西

### 3.1 代码与配置（仓库内）

| 路径 | 作用 |
|------|------|
| `scripts/finetune_stage1.sh` → `qwen/scripts/finetune.sh` | Stage I SFT |
| `scripts/train_score_stage2.sh` → `qwen/scripts/train_score.sh` | 默认 Stage II：**185K prebuilt pairs** |
| `scripts/train_score_stage2_koniq.sh` | 论文 §3.2 口径的 **KonIQ-only 在线配对**（不是 8K 冻结 JSON） |
| `scripts/gen_soft_label_koniq.sh` + `scripts/gen_soft_label.py` | 从 MOS+split 生成 DeQA Gaussian `level_probs`（**不是**论文 Eq. (9)） |
| `scripts/eval_iqa.sh` | scalar IQA 推理 + SRCC/PLCC |
| `configs/stage1_sft.yaml` / `stage2_score.yaml` / `stage2_score_koniq.yaml` / `qsit_multi.yaml` | 超参与数据混合 |
| `qwen/src/` | 训练、DeQA 损失、IQA eval |
| `tests/test_soft_labels.py`, `tests/test_portability.py` | 少量单测 |
| `docs/{DATA,TRAIN,EVAL,RELEASE_CHECKLIST}.md` | 使用说明 |

默认训练旋钮（与论文对照见 Gap 8）：

- Stage I：1 epoch，GBS 128，LLM/merger LR `1e-5`，vision `2e-6`，min pixels **524288**，seed 42，ZeRO-2，全参非 LoRA。
- Stage II：1 epoch，4 卡时 GBS **16**，LLM LR `2e-5`，vision `2e-6`，min pixels **196608**（launcher 第 190 行还硬编码 `196608`），`weight_rank=1.0`，`weight_softkl=1.0`，`weight_next_token=0.05`，`use_fix_std=true`。

### 3.2 本机数据

| 资源 | 路径 | 状态 |
|------|------|------|
| Stage I 便携 JSON | `data/annotations/*.json` | 9 个文件齐全；计数见下 |
| Stage I 实验室 JSON | `training_jsons/*.json` | 同上内容，可能含绝对路径；**不要上传** |
| Stage I 图像 | `/mnt/nvme/zhw/Co-Instruct-plus/data/` | **398,689** 个文件 |
| pairs | `pairs/pairs_180k.json` | **185,000** 对：KonIQ 65k / KADID 65k / LIVE 30k / CSIQ 15k / BID 10k |
| IQA 评测根 | `/mnt/nvme/zhw/Co-Instruct++/` | LIVE/CSIQ/KADID/BID/CLIVE/KONIQ/TID2013/AGIQA3K/PIPAL 的 metas **基本齐** |
| KonIQ 训练图 | `.../koniq10k/512x384/` | **10,373** 张；`train_koniq_7k.json` **7,046** 条；`test_koniq_2k.json` **2,010** 条 |
| MOS/split | `.../KONIQ/metas/{mos,split}.json` | 有 |
| 上传预打包 | `/mnt/nvme/zhw/hf_staging/co-instruct-plus/` | annotations + data + 多数 IQA 库；README 声明 **不含 pairs_180k、不含 SPAQ 图** |
| 基座权重 | `/home/zhw/IQA/Model/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct` | 本地 4 shard，约 17G |

**Stage I JSON 实数 vs Table 1：**

| 文件 | 条数 | 论文 Table 1 |
|------|------|----------------|
| `m2c_coarse_general_textonly` | 28,538 | 28,538 |
| `m2c_coarse_mcq_textonly` | 57,118 | 57,118 |
| `m2c_fine_general_textonly` | 28,511 | 28,511 |
| `m2c_fine_mcq_textonly` | 57,044 | 57,044 |
| `t2c_coarse_general` | 29,991 | 30,000（少 9） |
| `t2c_coarse_mcq` | 59,980 | 60,000（少 20） |
| `t2c_fine_general` | 30,000 | 30,000 |
| `t2c_fine_mcq` | 60,000 | 60,000 |
| 八文件合计 | **351,182** | **351,211**（少 29） |
| `coinstruct_562k_t2c` | **561,644** | 论文未列 |
| 默认 YAML 全量 | **912,826** | — |

M2C 四个文件**没有 `image` 字段**，是纯文本 description 比较。T2C 有 `image` 列表。

### 3.3 Checkpoint（注意：论文成品不在本仓库 `checkpoints/`）

`Co-Instruct-Plus/checkpoints/stage1_qsit_sft/` **只有 TensorBoard `runs/`，没有 `*.safetensors`**。不存在 `checkpoints/stage2_score/`。

实验室候选权重（**未与论文表格官方绑定**）：

| 位置 | 架构 | 备注 |
|------|------|------|
| `/mnt/nvme/zhw/Co-Instruct++/checkpoints/hf_Co-instruct__/t2c_from_8b` | Qwen3-VL | 更像 Stage I Assistant；Q-Bench single local-gt **82.47%**（论文 83.88%）；MICBench 1998 题 **83.18%**（论文 84.86%） |
| `.../koniq_t2cm2c` 及 `_stripped` | Qwen3-VL | LIVE-Wild PLCC/SRCC **0.910/0.901** 与 Table 7 **完全一致**；但 KADID 为 0.727/0.729（论文 0.744/0.746）；缺 KonIQ/PIPAL/CSIQ 本地 summary |
| `.../t2c_koniq` | Qwen3-VL | KADID **0.747/0.747** 接近 Table 7；LIVE-Wild 0.916/0.902 |
| `.../qwen3vl_koniq_pairalgo_base` | Qwen3-VL | `trainer_state` epoch=1, **441 steps** ≈ 7046/16，符合 **KonIQ-only 1 epoch GBS16**，不符合论文 8K/2ep/GBS192 |
| `.../qwen3vl_koniq_stage2_t2c` | Qwen3-VL | 6181 steps |
| `.../qwen35_9b_qsit_sft(_hfkeys)` | **Qwen3.5-9B** | **不是论文基座**。已有 frozen run：`eval_runs/20260817_qwen35_qsit_sft_nothink/` |
| Hub `yluo016/Co-instruct__` | 多子目录 | 含 `koniq_t2c`、`koniq_t2cm2c`、`t2c_koniq`、`t2c_from_8b`、`qwen3vl_mix180k_pairalgo`（checkpoint-15417 ≈ 185k/12）、`test_fft/checkpoint-7132` 等 |

本地 IQA 分数摘录（均只有 CLIVE/KADID/AGIQA 三库，**不是 Table 7 六库全集**）：见 `/home/zhw/IQA/test/scoring_eval/*/plcc_srcc_summary.txt`。

### 3.4 评测资产（在仓库外，必须冻结拷贝再用）

| 套件 | Canonical 路径 | 本机数据 |
|------|----------------|----------|
| Q-Bench | `/home/zhw/IQA/test/qbench/` | `llvisionqa_test(1).json`、`q-bench2-a1-test-with-answer.jsonl` |
| MICBench | `/home/zhw/IQA/test/MICBench/` | `micbench_test_with_answer.json` **2000** 条；实际跑过 **1998** |
| MICBench 5–6 图 / 构造 | `/home/zhw/IQA/test/MICBench++/`、`micbench_ori/` | 多份 JSON，**没有一份被仓库标明为论文 MICBench_v2 的 3000 题冻结集** |
| 2AFC-LMM | `/home/zhw/IQA/test/2AFC-LMM/` | `data/*.json` + `results/` 下已有多模型结果 |
| 已冻结 run | `/mnt/nvme/zhw/eval_runs/20260817_qwen35_qsit_sft_nothink/` | **Qwen3.5-9B，不是论文模型** |

论文 Table 4 中 Qwen3-VL-Plus overall **82.21%** 与 `/home/zhw/IQA/test/MICBench/eval_outputs/canonical_rerun/top5_detailed_table.md` **完全一致**，说明至少 proprietary 基线可能用过这套 1998 题协议；但同表最高本地模型 checkpoint-7132 为 **83.03%**，不是论文 Assistant 的 **84.86%**。因此 **不能把现有 MICBench 结果直接当成 Table 4 的复现**。

### 3.5 环境

| 项 | 本机 |
|----|------|
| GPU | 4× NVIDIA H20 96GB（与论文「4 NVIDIA H20」一致） |
| 训练 conda | `ENV_NAME=train` → `/mnt/nvme/zhw/train`：torch **2.8.0+cu128**，transformers **5.3.0**，deepspeed **0.17.5** |
| 评测 venv | `/mnt/nvme/zhw/test/bin/python`：torch **2.11.0+cu128**，transformers **5.12.1**（`eval_runs/20260817_.../env.txt`） |
| `requirements.txt` | 下限 pin（`transformers>=4.57.0`），**无精确锁文件 / Docker** |
| `configs/env.local.sh` | 已指向本机 `MODEL_NAME`、`IMAGE_FOLDER`、`ROOT_DATA`、`ANNOTATIONS_DIR` |

### 3.6 密钥

- `secrets/.env`：Hugging Face token（启动器、`download_annotations.sh` 会 source）。**本地下载已缓存的 Qwen3-VL 与本地 JSON 不强制需要**；拉 Hub 私有/限流资源、上传 annotations 需要。
- 当前 `.env` **没有** OpenAI / Anthropic / Gemini / DashScope key。复现 Table 3–6 的 GPT-5.4、Claude-Sonnet-4.5、Gemini、Qwen3-VL-Plus **需要另外的 API 密钥**，且版本会漂。
- 不要把 `secrets/.env` 提交进 git（已在 `.gitignore`）。

---

## 4. Gap 清单

编号独立于 `draft/review.md` 的 C1–C18，但交叉引用。

### Gap 1 — 论文协议 ≠ 默认代码协议（总 blocker）

- **论文需要**：351,211 Stage I + KonIQ 8K pairs Stage II；损失 Eq. (9)–(17)；相对标签「图2相对图1」；closed-set 五档 KL；预测方差进 ranking；2 epoch / GBS 192 / min_pixels 256×32×32。
- **仓库现有**：默认 912,826 + 185K 五库 pairs；DeQA Gaussian `level_probs`（大量 3–5 个非零项，部分和≠1）；相对句是「Image 1 relative to Image 2」；full-vocab KL；`use_fix_std=true`；有效损失 `L_fd + 0.05*(CE+KL)`；1 epoch；Stage I/II/eval 三套像素预算。细节见 `draft/review.md` C1–C10、C4–C8。
- **严重级别**：blocker
- **不改代码如何填**：做不到「让现有二进制实现变成论文公式」。只能二选一：
  1. **复现代码协议**（推荐作为可执行目标）：承认印刷数字来自另一套未冻结实验，用 README/hf_staging 的 code protocol 出一套**新**分数。
  2. **复现论文协议**：必须改代码 + 重训（超出本文范围）。环境变量/YAML **改不了** `softkl_loss` 的 softmax 范围、`level_probs` 公式、relative 方向。
- **依赖**：决定「论文数字绑定哪次 run」之前，后面所有 eval 都无法声称「复现论文」。

### Gap 2 — 没有与表格绑定的 Assistant / Rater 成品权重

- **论文需要**：产生 Table 3–8 的最终 checkpoint（论文只说 “final checkpoint”）。
- **仓库现有**：`checkpoints/stage1_qsit_sft/` 无权重；无 `stage2_score`。实验室/Hub 有多份变体，**无 card、无 hash、无 run manifest** 绑定 Table 3–8。
- **严重级别**：blocker（eval-only 印刷数字）
- **不改代码如何填**：
  ```bash
  # 1) 列出 Hub 子目录（需 HF token 若限流）
  hf download yluo016/Co-instruct__ --local-dir /mnt/nvme/zhw/Co-Instruct++/checkpoints/hf_Co-instruct__
  # 2) 对每个候选做 sha256，写入 eval_runs 的 notes.md
  sha256sum .../t2c_from_8b/model.safetensors .../koniq_t2cm2c/model.safetensors .../t2c_koniq/model.safetensors
  # 3) 用同一套冻结评测脚本跑 Table 3/4/5/7，看谁最接近印刷值
  ```
  最接近线索：`koniq_t2cm2c` 的 LIVE-Wild 与 Table 7 一致；`t2c_from_8b` 的 Q-Bench/MICBench 接近但低 1–2 个点。**没有一个已核验为 Table 7 六库全中。**
- **依赖**：Gap 6（冻结评测）、Gap 5（六库 metas/图像）。

### Gap 3 — Stage I 默认多了 561,644 条 legacy Co-Instruct

- **论文需要**：Table 1 的八个 JSON，351,211 条。
- **仓库现有**：`configs/qsit_multi.yaml` 第 20–21 行强制 `coinstruct_562k_t2c.json`。八文件实数 351,182（少 29）。
- **严重级别**：blocker（若目标是论文 359K）；对 code protocol 则是 already-in-place
- **不改代码如何填**：
  - **不要**改 `qwen/src`。用户可**另存一份 YAML**（新文件，不算改训练实现），只列八个 json，然后：
    ```bash
    export DATA_PATH=/mnt/nvme/zhw/Co-Instruct-Plus/configs/qsit_multi_table1_only.yaml  # 需你自己新建
    export OUTPUT_DIR=/mnt/nvme/zhw/Co-Instruct-Plus/checkpoints/stage1_table1_only
    bash scripts/finetune_stage1.sh 0,1,2,3
    ```
  - 少 29 条无法从公开信息补回（可能是清洗删除）。记录 `351,182 vs 351,211` 即可。
  - **禁止**把 `coinstruct_562k_t2c.json` 从目录删掉却仍指向原 YAML——launcher 会找不到文件直接失败。
- **依赖**：Gap 10 图像；Gap 12 环境。

### Gap 4 — Stage II「8K KonIQ pairs」不存在；默认是 185K 或 7,046 在线对

- **论文需要**：冻结 8K pair JSON。
- **仓库现有**：`pairs_180k.json`（185k，五库）；`train_score_stage2_koniq.sh` 读 `train_koniq_7k.json`（7046 图，`prebuilt_pairs: false`，每 epoch 动态配对 ≈7046 对）。全盘搜索无 `pairs_8k*`。
- **严重级别**：blocker（论文 §3.2）；important（若接受 KonIQ-only 在线配对为「最接近论文」）
- **不改代码如何填**：
  ```bash
  export ROOT_DATA=/mnt/nvme/zhw/Co-Instruct++
  # 若 train_koniq_7k.json 已在（本机有 7046 条），不必重跑
  bash scripts/gen_soft_label_koniq.sh          # 不要加 --overwrite-test
  export MODEL=/path/to/stage1_ckpt             # 必须是 Stage I 产出
  export KONIQ_OUTPUT=/mnt/nvme/zhw/Co-Instruct-Plus/checkpoints/stage2_score_koniq
  bash scripts/train_score_stage2_koniq.sh 0,1,2,3
  ```
  无法在不写新采样脚本的情况下凭空得到「论文 8K 冻结对」。185K 协议：
  ```bash
  export PAIRS_JSON=/mnt/nvme/zhw/Co-Instruct-Plus/pairs/pairs_180k.json
  bash scripts/train_score_stage2.sh 0,1,2,3
  ```
  注意：185K 含 KADID/LIVE/CSIQ 训练集时，Table 7 对应测试集是 **同库 held-out**，不是论文暗示的纯 zero-shot。
- **依赖**：Gap 2（Stage I ckpt）、Gap 5（KonIQ 图）。

### Gap 5 — 评测/训练图像缺口

- **论文需要**：Table 7 六库图像 + Stage I `data/` + Stage II KonIQ。
- **仓库现有**：
  - Stage I 图：本机齐。
  - KonIQ：`koniq10k/512x384` 齐；**`KONIQ/images/` 为 0 文件**，eval JSON 常写 `KONIQ/images/...`，必须 `export KONIQ_IMAGES_DIR=$ROOT_DATA/koniq10k/512x384`。
  - **SPAQ/images 几乎空**（仅 README.json）。hf_staging 同样缺 SPAQ 图。2AFC-LMM 的 SPAQ 子集与论文 Table 6 需要官方 SPAQ。
  - LIVE/CSIQ/KADID/BID/CLIVE/PIPAL/AGIQA/TID2013：本机 `Co-Instruct++` 下基本有。
- **严重级别**：important（Table 6 SPAQ、部分 2AFC）；scalar Table 7 不含 SPAQ，KonIQ remap 即可
- **不改代码如何填**：
  1. 按各数据集官方许可下载（KonIQ-10K、SPAQ、LIVE Challenge、KADID-10K、CSIQ、PIPAL、AGIQA-3K…），放到 `$ROOT_DATA/<NAME>/`。
  2. SPAQ：官方发布页 / 论文引用的 Fang et al. CVPR 2020；下载后 `export SPAQ_IMAGES_DIR=/path/to/SPAQ/images`。
  3. 不要覆盖已有 `test_*.json`。`gen_soft_label_koniq.sh` 默认写 `test_koniq_2k.generated.json`。
- **依赖**：无。许可证由用户自行遵守（`docs/DATA.md`）。

### Gap 6 — 仓库没有论文主评测入口（Q-Bench / 2AFC / MICBench_v2）

- **论文需要**：Table 3–6、Fig. 5 的脚本、prompt、解码、解析、Thurstone 聚合。
- **仓库现有**：只有 `scripts/eval_iqa.sh`（Table 7 风格）。`eval_iqa.sh` 默认扫 **10** 个 meta（LIVE、CSIQ、KADID、BID、CLIVE、KonIQ、TID2013、SPAQ、AGIQA3K、PIPAL），论文 Table 7 只有 **6** 个（无 BID/TID2013/SPAQ，LIVE-Wild=CLIVE）。
- **严重级别**：blocker（主表）；eval_iqa 对 Table 7 为 important
- **不改代码如何填**：见第 6 节冻结拷贝。用环境变量限制 IQA 集合：
  ```bash
  export META_PATHS="\
  $ROOT_DIR/KONIQ/metas/test_koniq_2k.json \
  $ROOT_DIR/KADID10K/metas/test_kadid_2k.json \
  $ROOT_DIR/PIPAL/metas/test_pipal_1k_new.json \
  $ROOT_DIR/CLIVE/metas/test_clive_234.json \
  $ROOT_DIR/AGIQA3K/metas/test_agiqa_3k.json \
  $ROOT_DIR/CSIQ/metas/test_csiq_174.fixed.json"
  bash scripts/eval_iqa.sh
  ```
  注意：本机同时存在 `test_clive_234.json` 与 `.cleaned.json`、`test_csiq_174.json` 与 `.fixed.json`。**必须在 notes.md 写死用哪一份**；现有 scoring_eval 多用 `.cleaned.json`，与默认 `eval_iqa.sh` 列表不完全相同。
- **依赖**：Gap 2、Gap 5、第 6 节。

### Gap 7 — MICBench_v2（3000 MCQ）未冻结、未发布

- **论文需要**：3,000 题；3–6 图；coarse/fine；human-validated；与训练图 no-overlap。
- **仓库现有**：无。实验室 `micbench_test_with_answer.json` 为 **2000**（跑 1998）。`MICBench++` / `micbench_ori` 有 5–6 图分片，有过 **4000** 题合并实验（`checkpoint-7132_merged_review1_predictions_full4000_summary.json` overall **78.35%**），对不上 84.86%。无 overlap 检查脚本在本仓库。
- **严重级别**：blocker（Table 4 / Fig. 5）
- **不改代码如何填**：
  1. 在作者侧选定**唯一** JSON（3000 或实际用于 Table 4 的那份），复制进 `eval_runs/.../data/`，算 sha256。
  2. 在选定文件上冻结 MICBench 脚本后重评（第 6 节）。
  3. 人工投票原始记录、伦理批件：**无法用公开数据重建**。
- **依赖**：Gap 2；作者确认「哪份 JSON 进了论文」。

### Gap 8 — §6.1 超参无法用现有 YAML 对齐（且部分 flag 无效）

- **论文需要**：2 epoch，GBS 192，单一 LR 2e-5，min_pixels 256×32×32。
- **仓库现有**：见 §3.1。可用环境变量覆盖 epoch/batch/LR/pixels（Stage I 的 `NUM_TRAIN_EPOCHS`、`GLOBAL_BATCH_SIZE`、`LEARNING_RATE`、`IMAGE_MIN_PIXELS`）。
- **严重级别**：important
- **不改代码如何填**：
  ```bash
  # 仅「表面上」靠近 §6.1；损失公式仍不同
  export NUM_TRAIN_EPOCHS=2
  export GLOBAL_BATCH_SIZE=192   # 必须能被 4 * BATCH_PER_DEVICE 整除；4 卡 batch_per_device=4 时 accum=12
  export LEARNING_RATE=2e-5 MERGER_LR=2e-5
  export IMAGE_MIN_PIXELS=262144
  export OUTPUT_DIR=.../checkpoints/stage1_paperish_hparams
  bash scripts/finetune_stage1.sh 0,1,2,3
  ```
  **Stage II 陷阱**：`qwen/scripts/train_score.sh` 读取了 `IMAGE_MIN_PIXELS`，但调用处硬编码 `--image_min_pixels 196608`。**不改该脚本则 Stage II 像素覆盖无效。** 这是无改代码路径的硬限制（对应 review I2）。
  `level_prefix` YAML 同样不生效（I3）。`max_seq_length` 截断被注释（I4）。
- **依赖**：Gap 1。

### Gap 9 — 数据构造管线缺失

- **论文需要**：Q-Pathway 19K ID；Q-Align 分数与 0.5 阈值分组；Qwen3-VL-Plus teacher 的 revision/prompt/温度；Teach2Compare 69K 来源与 KADIS 合成参数；pairs 的 `level_probs` 生成（Eq. 8–9）。
- **仓库现有**：只有最终 JSON。`gen_soft_label.py` 是 **DeQA pdf**，`density_type: pdf`（`configs/soft_labels_koniq.json`）。无 Q-Align、无 teacher 调用脚本。
- **严重级别**：blocker（从零重建语料）；对「用现成 JSON 训练」为 nice-to-have
- **不改代码如何填**：使用已有 `data/annotations/` + `pairs_180k.json` + `Co-Instruct-plus/data/`。不要试图用 API 重跑 Qwen3-VL-Plus teacher——即使用密钥，prompt 未公开，输出不可复现。`level_probs` 不要按 Eq. (9) 手写覆盖进 JSON，否则与「代码协议」和旧结果都对不上。
- **依赖**：无。

### Gap 10 — 公开数据托管未完成

- **论文需要**：第三方可下载的 annotations + 模型。
- **仓库现有**：git 忽略大 JSON；`download_annotations.sh` 要 `HF_DATASET_REPO`；Hub dataset 空；GitHub 占位。本机 lab 布局可训练。
- **严重级别**：blocker（对外复现）；对本机为已解决
- **不改代码如何填**（维护者，本机）：
  ```bash
  # 已有相对路径副本
  ls data/annotations/*.json pairs/pairs_180k.json
  # 上传（需要 HF write token，不要把 token 写进文档）
  # 先 rg 确认无绝对路径
  rg '/home/|/mnt/' data/annotations/*.json && echo FAIL || echo OK
  # 然后 hf upload（具体 repo id 需你指定；不要用空的 co-instruct-plus-score 除非先清空策略）
  ```
  外人：在 `HF_DATASET_REPO` 发布前，只能走实验室路径 `IMAGE_FOLDER` + 已有 JSON。
- **依赖**：密钥（HF write）。

### Gap 11 — Table 8 消融没有独立配置/权重清单

- **论文需要**：M2C-only / T2C-only / 有无 Stage II 等独立模型。
- **仓库现有**：目录名暗示 `m2c_koniq`、`t2c_koniq`、`koniq_t2c`、`koniq_t2cm2c`、`qwen3vl_stage2_t2c_from_m2c`，但无 YAML 说明各自 Stage I 是否含 562k、Stage II 是否 KonIQ-only。评分三库数字与 Table 8 部分接近、部分矛盾。
- **严重级别**：important
- **不改代码如何填**：对每个子目录跑同一套冻结 eval，填对照表；用「最接近印刷值」做假说，而不是声称已复现。从零消融需多次训练（Phase 4）。可用不同 `DATA_PATH` YAML 组合八文件子集，**仍改变不了 M2C text-only 的事实**（review C3）。
- **依赖**：Gap 2、6。

### Gap 12 — 环境未冻结；训练/评测 Python 栈不一致

- **论文需要**：可重复环境。
- **仓库现有**：无 `environment.yml` / `uv.lock` / Docker。训练 `train` env 与评测 `/mnt/nvme/zhw/test` 的 torch/transformers **主版本都不同**。
- **严重级别**：important
- **不改代码如何填**：每个 `eval_runs` 写 `env.txt`（照 `20260817_.../env.txt` 格式：`PY_BIN`、torch、transformers、CUDA、`nvidia-smi`、`git rev-parse`——注意本仓库几乎无有效 commit）。训练继续用 `conda run -n train`。不要混用两套 Python 比分数。
- **依赖**：无。

### Gap 13 — 推理与训练预处理不一致；eval 不输出方差

- **论文需要**：Eq. (13) 的 μ̂、σ̂²；与训练相同的 processor 预算。
- **仓库现有**：`iqa_eval_qwen.py` 读了 min/max pixels 但 `processor()` 未传入；只存五档 logits。`cal_plcc_srcc.py` 只出期望分数 + 四参数 logistic PLCC。
- **严重级别**：important（与印刷 IQA 协议是否一致未知）；nice-to-have（方差）
- **不改代码如何填**：保持 `eval_iqa.sh` 默认，在 notes.md 记录「eval 走 processor 默认 resize」。无法在不改代码时让 eval 写出方差。
- **依赖**：无。

### Gap 14 — Proprietary 与 teacher 基线

- **论文需要**：Qwen3-VL-Plus、GPT-5.4、Claude-Sonnet-4.5、Gemini-2.5-Pro 的固定版本与解码。
- **仓库现有**：无脚本。实验室 `/home/zhw/IQA/test/` 有 API runner。密钥不在 `secrets/.env`。
- **严重级别**：important（完整 Table 3–6）；对本模型列可跳过
- **不改代码如何填**：把实验室 API 脚本**复制**进 `eval_runs/.../scripts/`（不要改 canonical）。在 notes 记录调用日期与模型版本字符串。API 漂移导致**不能保证**再得到 82.21 / 82.01 等。
- **依赖**：新的 API 密钥（不要写入本 markdown）。

### Gap 15 — 人工研究、定性图、论文内部矛盾

- **论文需要**：MICBench 主观实验；Fig. 1–5；§3.2 KonIQ-only vs §4.3.2 多数据集 ranking。
- **仓库现有**：无主观原始数据。Fig. 5 可从 MICBench 分项精度重绘。方法图不能「训练出来」。
- **严重级别**：人工研究 = blocker（无法复现过程）；图 = nice-to-have
- **不改代码如何填**：Fig. 5 在得到分项数字后用任何绘图工具重画。人工实验只能归档已有标注 JSON，不能重做同一批 observer。
- **依赖**：Gap 7。

### Gap 16 — Resume 可能污染「1 epoch」

- **论文需要**：干净的最终 ckpt。
- **仓库现有**：Stage I 见 `checkpoint-*` 就 resume；Stage II `auto_resume: true`。
- **严重级别**：important
- **不改代码如何填**：**每次实验用全新 `OUTPUT_DIR`/`OUTPUT`/`KONIQ_OUTPUT`**，不要指向已有 `checkpoints/stage1_qsit_sft`。
- **依赖**：无。

---

## 5. 分阶段复现计划（不改训练/评测源码）

日期前缀按执行当天填写；下文用 `20260824` 作占位。成功标准写的是「对齐哪张表」，不是「感觉差不多」。

### Phase 0 — 清单与环境冻结

**动作**

1. 复制本文件日期、`hostname`、`nvidia-smi`。
2. 记录两套 Python：
   ```bash
   conda run -n train python -c "import torch,transformers,deepspeed; print(torch.__version__, transformers.__version__, deepspeed.__version__)"
   /mnt/nvme/zhw/test/bin/python -c "import torch,transformers; print(torch.__version__, transformers.__version__)"
   ```
3. 对关键数据做 sha256：`data/annotations/*.json`、`pairs/pairs_180k.json`、`train_koniq_7k.json`、选定的 test meta。
4. 确认 `configs/env.local.sh` 路径仍正确；**不要把该文件提交 git**。
5. 决定绑定目标：A=代码协议 或 B=论文印刷数字（B 在无改代码下大概率失败，见 Gap 1）。

**GPU/时间**：0。  
**成功标准**：得到一份 `env.txt` 草稿；sha256 表；书面选择 A/B。

### Phase 1 — 取得数据与权重（不下大模型，除非缺文件）

**动作**

1. 基座：已有则 `MODEL_NAME=/home/zhw/IQA/Model/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct`。缺失时才 `hf download Qwen/Qwen3-VL-8B-Instruct`（需 HF token）。
2. 图像：`IMAGE_FOLDER=/mnt/nvme/zhw/Co-Instruct-plus`，`ROOT_DATA=/mnt/nvme/zhw/Co-Instruct++`。补 SPAQ（Gap 5）。
3. 权重：不要用空的 `Co-Instruct-Plus/checkpoints/stage1_qsit_sft`。从 `hf_Co-instruct__/` 或 Hub 选候选（Gap 2）。
4. 不要跑 `download_annotations.sh` 去覆盖已有 JSON（Hub dataset 是空的，会失败或空转）。

**GPU/时间**：下载 SPAQ/权重为磁盘时间。  
**成功标准**：`ls $MODEL_NAME/config.json`；`test -f $ROOT_DATA/koniq10k/metas/train_koniq_7k.json`；SPAQ 图数量非零（若要跑 Table 6）。

### Phase 2 — Eval-only 尝试对齐印刷数字（优先 frozen eval_runs）

**动作**（示例：Rater 六库 IQA）

1. 建 `eval_runs/20260824_coinstruct_rater_table7_<ckptname>/`（结构见第 6 节）。
2. 把 `Co-Instruct-Plus/scripts/eval_iqa.sh` 与 `qwen/src/evaluate/*.py` **复制**进该 run 的 `scripts/`（评测实现来自本仓库，不是 canonical IQA 三套；仍建议冻结，避免以后仓库改动）。
3. `run.sh` 使用 Gap 6 的六路径 `META_PATHS` + `KONIQ_IMAGES_DIR`。
4. 对 `t2c_from_8b`、`koniq_t2cm2c`、`t2c_koniq` 各跑一次，**新 SAVE_DIR，不覆盖** `/home/zhw/IQA/test/scoring_eval/`。
5. Q-Bench / MICBench / 2AFC：冻结 canonical 拷贝后，对同一 ckpt 各建独立 `eval_runs`（或同一 run 下分 `run_qbench.sh` / `run_micbench.sh` / `run_2afc.sh`）。参考已有 `eval_runs/20260817_qwen35_qsit_sft_nothink/{run.sh,run_qbench.sh,run_micbench.sh}` 的布局，但 **MODEL 换成 Qwen3-VL 候选，不要用 qwen35_9b**。

**GPU/时间（4×H20，粗估）**

| 任务 | 粗估 |
|------|------|
| 六库 scalar IQA（batch=1） | 每模型数小时级 |
| Q-Bench single+pair | 数小时 |
| MICBench ~2k | 数小时（20260817 日志约 1.9h / 1998 题 / 单卡） |
| 2AFC-LMM 全表 | 视 pair 数，可达一天级 |

**成功标准**

- Table 7：六库 PLCC/SRCC 与印刷值差进入你预设容差（建议先看能否 **三位小数内**命中 KonIQ 0.962/0.950 与 LIVE-Wild 0.910/0.901）。
- Table 3/5：overall 命中 81.94 / 83.88。
- Table 4：overall 命中 84.86 **且** 题量为论文的 3000（若仍是 1998，即使分数接近也不能声称复现 Table 4）。
- Table 6：六数据集 α/ρ 命中。

若全部落空：停止「印刷数字」路线，改 Phase 3 的代码协议。

### Phase 3 — 从零训练（代码协议，可选 paper-ish 超参）

**动作**

1. **新输出目录**（Gap 16）：
   ```bash
   export IMAGE_FOLDER=/mnt/nvme/zhw/Co-Instruct-plus
   export ANNOTATIONS_DIR=/mnt/nvme/zhw/Co-Instruct-Plus/data/annotations
   export OUTPUT_DIR=/mnt/nvme/zhw/Co-Instruct-Plus/checkpoints/stage1_qsit_sft_$(date +%Y%m%d)
   export COINSTRUCT_SEED=42
   bash scripts/finetune_stage1.sh 0,1,2,3
   ```
   默认将训练 **912,826** 条、1 epoch、GBS 128。步数约 `912826/128 ≈ 7131`。
2. Stage II 选一条，不要混：
   - 论文最接近：`train_score_stage2_koniq.sh`（~441 step / 4 卡 GBS16）。
   - 默认仓库：`train_score_stage2.sh`（185k pairs）。
3. 可选：Gap 3 的 Table1-only YAML；Gap 8 的 epoch/GBS 覆盖。须在 `notes.md` 写清，**不要**事后称作「论文官方 run」。

**GPU/时间（4×H20，极粗）**

- Stage I 全量 912k 全参 8B + 多图：按 7131 step、ZeRO-2，可能 **1–3 天**（视图像 token 与 I/O）。
- Stage I 仅 351k：约 40% 墙钟。
- Stage II KonIQ-only：441 step，大约 **数小时**。
- Stage II 185k：`185000/16 ≈ 11563` step，大约 **1 天量级**。

**成功标准**：新 ckpt 上 Phase 2 的同一冻结脚本；报告的是 **code protocol 分数**，并与 Table 7 并排对比（允许不同）。

### Phase 4 — 消融、图、人工

**动作**

- Table 8：对已有 `m2c_*` / `t2c_*` / `koniq_t2cm2c` 做 eval-only 对照；缺的组合才从零训（乘以 Phase 3 成本）。
- Fig. 5：用 MICBench 分项精度重绘；Fig. 1–4 不「复现」。
- 人工研究：归档现有 MICBench JSON 的 sha256；标记为不可复现过程。

**成功标准**：一张「候选 ckpt × 指标」对照表，明确哪些行是印刷值、哪些是新测值。

---

## 6. 评测冻结拷贝规则（必须遵守）

**禁止**为一次实验直接改：

- `/home/zhw/IQA/test/qbench`
- `/home/zhw/IQA/test/MICBench`
- `/home/zhw/IQA/test/2AFC-LMM`

模板（每个实验独立目录，禁止覆盖旧 `eval_runs` 或 `eval_outputs/<name>/`）：

```text
/mnt/nvme/zhw/eval_runs/20260824_<exp_name>/
  notes.md          # 对比哪次 run、ckpt 路径与 sha256、题集 sha256、相对论文哪张表
  env.txt           # PY_BIN, CUDA, torch, transformers, hostname, nvidia-smi
  run.sh            # 唯一入口
  run_qbench.sh     # 如需要
  run_micbench.sh
  run_2afc.sh
  run_iqa.sh
  scripts/          # 从 canonical 或本仓库复制的 .py/.sh（可含 nothink 等补丁，只存在于此）
  scripts.sha256
  outputs/
  logs/
```

操作顺序：

```bash
EXP=/mnt/nvme/zhw/eval_runs/20260824_coinstruct_assistant_t2c_from_8b
mkdir -p "$EXP"/{scripts,outputs,logs}
# 例：拷贝 Q-Bench / MICBench（按实际入口文件增补）
cp -a /home/zhw/IQA/test/qbench/eval_qbench_single.py "$EXP/scripts/qbench/"
cp -a /home/zhw/IQA/test/qbench/eval_qbench_pair.py   "$EXP/scripts/qbench/"
cp -a /home/zhw/IQA/test/MICBench/eval_micbench.py \
      /home/zhw/IQA/test/MICBench/micbench_runners.py \
      /home/zhw/IQA/test/MICBench/micbench_standard.py \
      "$EXP/scripts/micbench/"
# 2AFC
cp -a /home/zhw/IQA/test/2AFC-LMM/main.py \
      /home/zhw/IQA/test/2AFC-LMM/params.py \
      /home/zhw/IQA/test/2AFC-LMM/cal_acc_consistency.py \
      /home/zhw/IQA/test/2AFC-LMM/correlation.py \
      "$EXP/scripts/2afc/"
# IQA（来自本仓库，同样冻结）
cp -a /mnt/nvme/zhw/Co-Instruct-Plus/qwen/src/evaluate/*.py "$EXP/scripts/iqa/"
cp -a /mnt/nvme/zhw/Co-Instruct-Plus/scripts/eval_iqa.sh "$EXP/scripts/iqa/"
(cd "$EXP" && find scripts -type f -exec sha256sum {} + > scripts.sha256)
```

`run.sh` 只调用 **`$EXP/scripts`**，用绝对路径指向数据和 `MODEL`。可参考：

`/mnt/nvme/zhw/eval_runs/20260817_qwen35_qsit_sft_nothink/notes.md`

若必须改 thinking / max_new_tokens：只改拷贝，**改完后把 canonical 仓库恢复原样**（该 frozen run 的 notes 已示范过这一纪律）。

IQA 与论文 Table 7 对齐时，在 `notes.md` 写死：

- meta 文件名（`.json` vs `.cleaned.json` vs `.fixed.json`）
- `KONIQ_IMAGES_DIR` / `SPAQ_IMAGES_DIR`
- `PREPROCESSOR_PATH`（若 ckpt 缺 preprocessor，指向基座目录）
- `level_names=Excellent Good Fair Poor Bad`（代码顺序，不是论文小写 bad→excellent）

---

## 7. 风险、未知、即使不改代码也无法复现的部分

1. **损失与软标签公式**：Eq. (9) 三角插值、closed-set KL、Eq. (15) 预测方差 ranking、Eq. (12)(17) 的 λ/γ 展开，均与默认实现不同。YAML 不能修复。
2. **相对标签方向**（论文图2相对图1 vs 代码图1相对图2）：翻转 JSON 的 `comparison` 字段会改变监督，等于新实验，且仍与现有 ckpt 不兼容。
3. **M2C 无视觉输入**：论文把 M2C 写成 multi-image visual comparison；JSON 是 text-only。不改数据就无法「补上图像」。
4. **8K 冻结 pairs、少掉的 29 条 Table 1、Q-Pathway ID、teacher 轨迹**：公开来源重建不了同一集合。
5. **MICBench_v2 3000 + 主观实验过程**。
6. **Table 3–8 与 commit/JSON/ckpt 的三元绑定**：仓库没有 manifest；Hub 无 model card。
7. **API 基线版本漂**；`secrets/.env` 无这些 key。
8. **eval 预处理 ≠ 训练**：即使权重对，IQA 数字仍可能和训练时协议不同。
9. **Stage II 像素环境变量无效**（硬编码 196608）。
10. **Qwen3.5-9B 实验**（`qwen35_9b_qsit_sft`）容易被误当成论文模型；论文是 Qwen3-VL-8B。
11. **185K Stage II** 使 Table 7 的 KADID/CSIQ/LIVE 解释从 zero-shot 变成同库 held-out（若该 ckpt 确由 185K 训出）。
12. **公开 GitHub/HF 数据集尚未构成可引用的外部复现包**。
13. **本仓库 git 几乎未跟踪代码**：外人无法 `git checkout <sha>` 对齐你此刻的工作树。执行前应先在内部打 tag 或把当时文件树记入 `eval_runs/scripts`。

---

## 8. 建议的立即下一步（Top 5）

1. **书面选定绑定目标**：印刷论文 vs 代码协议。在 `draft/review.md` 的方案 A/B 上勾一次。无改代码只能忠实做代码协议，印刷表当「待对齐假说」。
2. **给 3 个最像的 ckpt 做 eval-only Table 7**（`t2c_from_8b`、`koniq_t2cm2c`、`t2c_koniq`），新建 `eval_runs/20260824_table7_*`，六库 `META_PATHS` + `KONIQ_IMAGES_DIR`，不覆盖旧 scoring_eval。先看能否命中 KonIQ 0.962/0.950。
3. **确认 MICBench 题集**：找到或由作者指定 Table 4 的 3000 JSON；在冻结目录计算 sha256。在此之前不要把 1998 题的 83.x% 写成 84.86% 的复现。
4. **补 SPAQ 图像 + 记录 HF/GitHub 发布缺口**：`HF_DATASET_REPO` 仍不能用；`yluo016/co-instruct-plus-score` 是空仓。本机训练不依赖它，对外复现依赖它。
5. **Phase 3 若要训**：新 `OUTPUT_DIR`，`conda run -n train`，先跑 KonIQ-only Stage II（成本低）做通路，再决定是否烧全量 912k Stage I。不要 resume 进现有空的 `stage1_qsit_sft`。

---

## 附录 A — 论文数字 vs 本机已有结果（摘）

| 指标 | 论文 | 本机最接近线索 | 是否可称为复现 |
|------|------|----------------|----------------|
| Q-BenchPAIR overall | 81.94% | `t2c_from_8b` pair all 80.8% | 否 |
| Q-BenchSINGLE overall | 83.88% | `t2c_from_8b` single local-gt 82.47% | 否 |
| MICBench_v2 overall | 84.86% (3000) | `t2c_from_8b` 83.18% / 7132 83.03%（**1998 题**）；Plus 82.21% 与论文 Plus **一致** | 题集不同，否 |
| LIVE-Wild PLCC/SRCC | 0.910/0.901 | `koniq_t2cm2c` **0.910/0.901** | 单库吻合，六库未全测 |
| KADID PLCC/SRCC | 0.744/0.746 | `t2c_koniq` 0.747/0.747 | 接近，ckpt 可能不是同一个 |
| AGIQA SRCC | 0.749 | `koniq_t2cm2c` 0.749 | 单库吻合 |
| KonIQ 0.962/0.950 | — | 本地 scoring_eval **没有** KonIQ/PIPAL/CSIQ summary | 未知 |
| 基座 | Qwen3-VL-8B-Instruct | 实验室另有 Qwen3.5-9B SFT | 不是同一模型 |

## 附录 B — 本机一键路径速查（执行时用）

```bash
# 根
REPO=/mnt/nvme/zhw/Co-Instruct-Plus
export IMAGE_FOLDER=/mnt/nvme/zhw/Co-Instruct-plus
export ROOT_DATA=/mnt/nvme/zhw/Co-Instruct++
export ANNOTATIONS_DIR=$REPO/data/annotations
export MODEL_NAME=/home/zhw/IQA/Model/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct
export KONIQ_IMAGES_DIR=$ROOT_DATA/koniq10k/512x384
export ENV_NAME=train
export COINSTRUCT_SEED=42
# 候选 Rater/Assistant（尚未官方绑定论文）
# ASSISTANT=/mnt/nvme/zhw/Co-Instruct++/checkpoints/hf_Co-instruct__/t2c_from_8b
# RATER=/mnt/nvme/zhw/Co-Instruct++/checkpoints/hf_Co-instruct__/koniq_t2cm2c
```

密钥：Hub 下载/上传需要 `secrets/.env` 中的 HF token；Qwen3-VL-Plus / GPT / Claude / Gemini **需要额外 API key，当前该文件未包含**。

## 附录 C — 与 `draft/review.md` 的关系

`review.md` 回答「论文是否描述了默认代码」。本文回答「不改代码能否复现论文」。两者一致处：C1/C2/C4/C5/C7/C9 仍是硬差异。本文额外盘点了本机 ckpt、Hub、eval_runs、SPAQ 缺口和冻结评测纪律。
