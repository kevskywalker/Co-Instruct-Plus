# Co-Instruct++ 论文与训练代码一致性审查

审查对象：

- 论文：`draft/main.pdf`（14 页；`draft/` 下未发现 LaTeX 源文件）
- 默认训练入口：`scripts/finetune_stage1.sh`、`scripts/train_score_stage2.sh`
- 实际启动器：`qwen/scripts/finetune.sh`、`qwen/scripts/train_score.sh`
- 训练、数据和损失实现：`qwen/src/`
- 默认配置与当前随仓库提供的数据：`configs/`、`data/annotations/`、`pairs/pairs_180k.json`

审查日期：2026-08-18。

## 结论

论文与当前仓库默认实现**并非完全一致**。差异不是仅限于文档遗漏，而是涉及训练语料规模和来源、Stage II 标签语义、soft label 定义、总损失、排序方差、训练超参数和图像预算等核心实验条件。按当前默认入口训练得到的模型，不能直接视为论文所述 “Co-Instruct-359K、KonIQ 8K Stage II、2 epochs、global batch 192” 配置的复现。

最严重的四项是：

- [ ] 论文的 Stage I 为 351,211 条，默认代码实际加载 912,826 条，其中额外加入未在论文数据表中披露的 561,644 条 `coinstruct_562k_t2c`。
- [ ] 论文的 Stage II 为 KonIQ-10K 训练集 8K pairs，默认代码实际使用来自 5 个 IQA 数据集的 185,000 pairs。
- [ ] 论文定义“第二张图相对第一张图”的标签，代码训练的是“第一张图相对第二张图”的关系句。
- [ ] 论文的 ordinal soft label 和总损失公式不等于代码实际执行的 full-vocabulary KL、固定方差 fidelity ranking 和 `rank + 0.05 × (CE + KL)`。

在投稿或公开模型前，应先确定哪一个才是产生论文表格结果的真实协议：

- [ ] 如果论文结果来自当前仓库默认配置，应优先修改论文并重新命名/统计训练数据。
- [ ] 如果论文结果来自 359K + KonIQ-8K 的私有旧协议，应冻结并发布该协议的 JSON 清单、配置、脚本和 checkpoint hash，而不能将当前默认入口称为论文复现代码。



## 审查口径

本报告把以下三类问题分开：

- **明确不一致**：论文有确定陈述，默认代码或随仓库数据给出不同事实。
- **实现缺陷/隐患**：即使修改论文文字，代码本身仍可能不符合其目标数学定义。
- **无法核验**：论文陈述涉及离线数据构造、人工标注或未发布的评测代码，仓库内没有足够证据。

配置优先级按实际入口判断：启动脚本 CLI / 环境变量覆盖 YAML，YAML 又覆盖 dataclass 默认值。不能用 `params.py` 中未被默认脚本采用的值代表论文实验。

---



## 一、明确不一致



### C1. Stage I 默认训练量为 912,826，而不是 351,211

- [x] 已处理

**严重级别：Critical**

论文证据：

- 摘要、§1 和结论将语料称为 **Co-Instruct-359K**。
- §3.1 Table 1 声明 Stage I 共 **351,211** 条。

代码和数据证据：

- `configs/qsit_multi.yaml:3-21` 以 `sampling_strategy: "all"` 加载 9 个 JSON。
- `qwen/src/dataset/sft_dataset.py:85-113` 会把这些 JSON 全量直接拼接。
- Table 1 对应的 8 个 M2C/T2C 文件当前实数为 **351,182**，比表中少 29 条：
  - M2C coarse reasoning：28,538
  - M2C coarse question：57,118
  - M2C fine reasoning：28,511
  - M2C fine question：57,044
  - T2C coarse reasoning：29,991（论文 30,000）
  - T2C coarse question：59,980（论文 60,000）
  - T2C fine reasoning：30,000
  - T2C fine question：60,000
- 默认 YAML 还额外加入 `coinstruct_562k_t2c.json`，实际 **561,644** 条。
- 因而默认 Stage I 总量是 `351,182 + 561,644 = 912,826`。

额外语料也不是 Table 1 所述 2–6 图数据的同义副本。其图像数分布为：

- 1 图：200,209
- 2 图：222,509
- 3 图：77,648
- 4 图：61,278

这些记录的图像路径数与 `<image>` token 数是对齐的；不存在“多图路径只使用第一张”的问题。

**影响：**

- “359K” 名称、数据贡献归因和消融解释均不适用于默认训练。
- 额外 561K 语料可能是主要性能来源之一，Table 8 未对其做控制。

**修改建议：**

- [ ] 若论文结果使用当前默认配置：把 Stage I 改为 912,826，增加 legacy Co-Instruct 子集的来源、图像数分布和消融，并相应重命名数据集。
- [ ] 若论文结果确实只用 Table 1 的八个文件：删除默认 YAML 中的 `coinstruct_562k_t2c.json`，发布论文一致的 frozen YAML；同时说明当前文件比 Table 1 少 29 条的原因。



### C2. Stage II 是 185K 五数据集 pairs，而不是 KonIQ-10K 8K pairs

- [x] 已处理

**严重级别：Critical**

论文证据：

- §3.2（PDF p.5）明确写道：从 KonIQ-10K training split 采样 **8K image pairs**。
- Table 8 将 Stage II 标为 **KonIQ-10K**。

代码和数据证据：

- `qwen/scripts/train_score.sh:26,105-108,138` 默认加载 `pairs/pairs_180k.json`。
- `configs/stage2_score.yaml:33` 设置 `prebuilt_pairs: true`。
- `qwen/src/dataset/pair_dataset.py:650-678` 因而选择 `PairDatasetPrebuilt`。
- 文件名虽为 `pairs_180k.json`，实际包含 **185,000** 对：
  - KonIQ-10K：65,000
  - KADID-10K：65,000
  - LIVE：30,000
  - CSIQ：15,000
  - BID：10,000
- 仓库现有可选 KonIQ-only 入口 `scripts/train_score_stage2_koniq.sh`（配置 `configs/stage2_score_koniq.yaml`，对应 lab 脚本 `Co-Instruct++/qwen/scripts/train_qwen_koniq.sh`）。它加载 `$ROOT_DATA/koniq10k/metas/train_koniq_7k.json`（**7,046 张图**），且 `prebuilt_pairs: false`，每个 epoch 在线配对约 7,046 对，**不是**论文写的冻结 8K pair JSON。该入口不是默认 Stage II。

**影响：**

- §3.2、Table 8 和默认代码直接冲突。
- Table 7 又在 KADID-10K、LIVE-Wild 和 CSIQ 上报告结果。即使训练和测试图像严格分离，这也是“同数据库 held-out generalization”，不能不加限定地解释为对这些数据库的纯 cross-dataset zero-shot generalization。
- 论文 §3.2 的 “KonIQ-only” 与 §4.3.2 的 “heterogeneous IQA datasets / dataset-wise ranking” 在论文内部也互相矛盾；当前代码更接近后者。

**修改建议：**

- [ ] 首选：确认论文表格究竟来自 185K prebuilt、KonIQ 7k 在线配对，还是未发布的冻结 8K pairs；若是 7k 在线配对，把 §3.2 的 “8K image pairs” 改成 “KonIQ-10K train split（7,046 images）with online pairing”，并让论文默认 launcher 指向 `train_score_stage2_koniq.sh`。
- [ ] 如果 185K 多数据集才是真实实验：重写 §3.2、Table 8 和实验结论；逐数据集报告 train/test split，避免把同数据库测试描述成未见域泛化。



### C3. Merge2Compare 默认训练数据是 text-only，不是视觉多图输入

- [ ] 已处理

**严重级别：High**

论文证据：

- §3.1 将 Merge2Compare 描述为从图像 pair/group 构造比较指令。
- §4.2 说 Stage I 通过 multi-image input 学习视觉比较。

代码和数据证据：

- 四个默认 M2C 文件名均带 `_textonly`。
- 共 171,211 条 M2C 记录没有 `image` 字段，也没有 `<image>` token；prompt 中只有 `Image 1 description: ...` 等文字描述。
- `qwen/src/dataset/sft_dataset.py:118-183` 对无 `image`/`video` 字段的记录走纯文本分支。

**影响：**

- M2C 实际训练语言空间中的“描述合并与比较”，不直接训练对对应图像的视觉比较。
- 论文把 M2C 与 T2C 一并表述为 multi-image visual supervision 会夸大实际视觉监督量。

**修改建议：**

- [ ] 若 text-only 是设计本意：论文应明确 M2C 是 description-conditioned textual comparison supervision，只有 T2C 和额外 Co-Instruct 数据提供视觉输入。
- [ ] 若希望符合现有论文：在 M2C JSON 中加入真实图像路径和对应 `<image>` token，并重新训练。



### C4. Stage II 相对标签的参照方向相反

- [x] 已处理

**严重级别：Critical**

论文证据：

- Table 2：relative-quality label 描述 **the second image with respect to the first**。
- Eq. (3)：`Δ_(2→1) = μ2 - μ1`。
- Eq. (5) 根据该差值输出 inferior → superior。

代码证据：

- `qwen/src/dataset/pair_dataset.py:300-313` 使用 `score_a - score_b`。
- `qwen/src/dataset/pair_dataset.py:535-559` 输出：
`Image 1 is {inferior/worse/similar/better/superior} to Image 2.`
- 当前 prebuilt pairs 中的 `comparison` 字段也遵循 Image 1 相对 Image 2 的方向。

**影响：**

- 词汇本身没有错，但论文定义的随机变量、响应模板和实现的语义方向相反。
- 复现者按 Eq. (3) 生成标签时，会把非 similar 样本训练成相反的关系词。

**修改建议：**

- [ ] 如果保留现有模型和数据，统一把论文改为 `Δ_(1→2)=μ1-μ2`，并将 Table 2 改成 Image 1 with respect to Image 2。
- [ ] 如果保留论文公式，则需翻转数据中所有非 similar comparison，并修改 response 模板后重新训练。



### C5. Ordinal soft target 不符合 Eq. (8)–(9)

- [ ] 已处理

**严重级别：Critical**

论文证据：

- Eq. (8) 把 MOS 线性映射到 `[1,5]`。
- Eq. (9) 使用三角插值，概率质量只分配给最近的一个或两个 anchor。

代码和数据证据：

- `PairDatasetPrebuilt` 从 JSON 直接读取预计算的 `level_probs`：`pair_dataset.py:517-586`。
- 仓库没有生成这些 `level_probs` 的脚本。
- 当前 185K pairs 共 370K 个分布，其非零项个数为：
  - 1 项：90
  - 2 项：74,541
  - 3 项：79,841
  - 4 项：127,707
  - 5 项：87,821
- 因此多数 target 并非 Eq. (9) 所述“最多两个非零项”。
- 直接按 KonIQ 的 dataset min/max 重算 Eq. (8)–(9)，与 JSON target 的平均 L1 差约为 0.45，不是浮点误差。

更严重的是，210,559 / 370,000 个 `level_probs` 的和偏离 1 超过 `1e-6`；全体均值约 1.00776，最大约 1.07022。`modeling_deqa_qwen.py:221-227` 未重新归一化便传给 `F.kl_div`。

**影响：**

- 论文中的 `q` 是合法概率分布，当前一部分 target 不是归一化分布。
- 实际 soft target 更像未公开的 DeQA/Gaussian 分桶结果，而不是论文的新三角插值公式。

**修改建议：**

- [ ] 若 Eq. (9) 是目标实现：在 dataset 中根据 MOS 在线计算，或重新生成 pairs；加入 `sum(q)=1`、非负性和最多两个非零项的测试。
- [ ] 若当前 JSON 是真实实验：论文应给出其准确生成公式并删除 Eq. (9) 的“两最近 level”陈述；训练前仍应归一化 target，并评估修复是否改变结果。



### C6. Ordinal KL 使用全词表 softmax，而论文写 closed-set 五档 softmax

- [x] 已处理

**严重级别：High**

论文证据：

- Eq. (10) 定义仅在五个 absolute-quality tokens 上归一化的 closed-set softmax。
- Eq. (11) 在该五档分布上计算 KL。

代码证据：

- `modeling_deqa_qwen.py:221-227` 先对**完整词表** logits 做 softmax，再把 target 填到五个 level token ID，计算 KL。
- `closeset_rating_loss: true` 只在 `get_score()` 的分数提取中生效（`modeling_deqa_qwen.py:432-441`），不改变 `softkl_loss()`。

**影响：**

- 实际 KL 不只约束五档之间的相对分布，还惩罚落在五档词之外的全部概率质量。
- Eq. (10) 不能准确解释训练梯度。

**修改建议：**

- [ ] 代码对齐论文：先切出五个 level logits，再做 softmax 和 KL。
- [x] 或论文对齐代码：明确使用 full-vocabulary KL、target support 限于五个 level tokens；不要称其为 Eq. (10) 的 closed-set KL。



### C7. Stage II 实际总损失与 Eq. (12)、Eq. (17) 不同

- [ ] 已处理

**严重级别：Critical**

论文写法：

- `L_rater = L_ce + λ L_ord`
- `L = L_rater + γ L_fd`

默认代码实际执行路径：

1. `forward_single()` 计算 `CE + weight_softkl × KL`（`modeling_deqa_qwen.py:343-382`）。
2. `get_score()` 把该组合整体作为 `loss_next_token` 返回（`modeling_deqa_qwen.py:396-452`）。
3. `forward_pair()` 再乘 `weight_next_token`，并加 ranking loss（`modeling_deqa_qwen.py:507-568`）。
4. 默认配置为：
  - `weight_rank = 1.0`
  - `weight_softkl = 1.0`
  - `weight_next_token = 0.05`
  - `weight_in_level = None`
  - 所有 185K 样本均为 `task_type="score"`，所以 description 分支为 0。

因此默认有效目标是：

```text
L_actual = 1.0 * L_fd + 0.05 * L_CE + 0.05 * L_KL
```

而不是从论文文字自然得到的 `L_CE + λL_ord + γL_fd`。特别地，`weight_softkl=1.0` 是括号内权重，KL 最终仍被外层 0.05 缩放。

**修改建议：**

- [ ] 在论文中给出实际展开式和所有系数，不要只报告未赋值的 λ、γ。
- [ ] 更稳妥的代码修改是让模型分别返回 CE、KL、rank，再在一处显式组合，避免 `loss_next_token` 实际已经包含 KL 的误导性命名。



### C8. 排序概率没有使用论文 Eq. (15) 的预测方差

- [x] 已处理

**严重级别：High**

论文证据：

- Eq. (13) 计算预测均值与预测方差。
- Eq. (15) 用 `sqrt(σ̂_A² + σ̂_B²)` 归一化预测均值差。

代码证据：

- `configs/stage2_score.yaml:31-32` 设置 `use_fix_std: true`、`detach_pred_std: true`。
- `modeling_deqa_qwen.py:242-260` 在 `use_fix_std` 为 true 时直接使用固定分母形式，完全不读取预测 std。
- 因而 `detach_pred_std` 在默认路径下也没有作用。

ground-truth 比较概率仍使用标注 std（`modeling_deqa_qwen.py:261-268`），不一致只发生在 predicted probability。

**修改建议：**

- [ ] 若论文 Eq. (15) 是真实方法：设置 `use_fix_std: false`，决定是否 detach 后重新训练。
- [ ] 若固定方差是实际实验：重写 Eq. (15) 为固定方差版本，并删除“预测不确定性参与 ranking”的表述。预测方差仍可作为输出统计量，但不能声称它参与默认训练。



### C9. §6.1 的 epoch、batch size 和 learning rate 均与默认训练不同

- [x] 已处理

**严重级别：Critical**

论文 §6.1：

- learning rate：`2 × 10^-5`
- epochs：2
- global batch size：192
- all parameters updated

默认代码：

**Stage I**

- 1 epoch：`configs/stage1_sft.yaml:2`
- global batch 128：每卡 4，默认 4 卡，gradient accumulation 8；见 `qwen/scripts/finetune.sh:44-56`
- LLM / merger LR `1e-5`，vision LR `2e-6`
- weight decay 0.1

**Stage II**

- 1 epoch：`configs/stage2_score.yaml:2`
- 默认 4 卡时 global batch 16：每卡 4，gradient accumulation 1
- LLM / merger LR `2e-5`，vision LR `2e-6`
- weight decay 0

“所有参数更新”与代码一致：两阶段均显式 `freeze_vision_tower=False`、`freeze_llm=False`、`freeze_merger=False`，默认也未启用 LoRA。

**修改建议：**

- [ ] §6.1 必须拆成 Stage I 和 Stage II 两套配置，报告主干、vision、merger 的独立 LR，以及 warmup 0.03、cosine scheduler、weight decay、bf16、ZeRO stage、seed。
- [ ] 若论文的 2 epoch / GBS 192 才是真实运行，应提供覆盖 YAML 的 frozen run script 或训练日志；当前仓库没有证据支持这些覆盖值。



### C10. 图像最小像素预算与论文不同，且三处协议不同

- [x] 已处理

**严重级别：High**

论文 §6.1：

- `min_pixels = 256 × 32 × 32 = 262,144`
- `max_pixels = 1280 × 32 × 32 = 1,310,720`

默认代码：

- Stage I：`524,288 = 512 × 32 × 32`；`configs/stage1_sft.yaml:16-17`
- Stage II：`196,608 = 192 × 32 × 32`；`configs/stage2_score.yaml:17-18`
- scalar IQA eval 的 argparse 默认：`524,288`；`iqa_eval_qwen.py:241-242`

另外，Stage II launcher 虽在 `train_score.sh:85-86` 读取 YAML/env 的像素值，却在 `train_score.sh:142-143` 硬编码 `196608` 和 `1310720`，使这些覆盖变量不生效。

评测脚本在 `iqa_eval_qwen.py:101-102` 读取 min/max 后，`processor()` 调用（lines 175-181）没有传入这些预算，也没有走训练时的 `get_image_info`；实际使用 processor 默认 resize。

**修改建议：**

- [ ] 选定一套协议并统一 Stage I、Stage II 和 eval。
- [ ] 修复 Stage II launcher，传 `${IMAGE_MIN_PIXELS}` / `${IMAGE_MAX_PIXELS}`。
- [ ] eval 使用与训练相同的 `process_vision_info` 路径或显式传预算。
- [ ] 论文分别报告两个阶段和推理的真实值，不能用单一 256–1280 范围概括当前实现。



### C11. “每张图前有序数短语的 image–text interleaving”未实现

- [x] 已处理

**严重级别：Medium**

论文 §4.1 声称每张图前明确加入 “the first image”“the second image”等 ordinal phrase。

实际数据：

- T2C 典型输入为连续的 `<image>\n<image>\n...`，然后问题用 first/second 指代，并非每个 image token 前有 ordinal phrase。
- M2C 是 `Image 1 description: ...` 的纯文本。
- Stage II 是先给随机问题，再写 `Image 1: <image>`、`Image 2: <image>`，只有这一阶段接近所述格式。
- 额外 `coinstruct_562k_t2c` 又有其自身 1–4 图模板。

**修改建议：**

- [ ] 论文改为“通过 prompt 中的顺序引用保持图像身份”，并按阶段说明格式。
- [ ] 或统一数据 formatter，使每张视觉 token 均有显式 ordinal prefix，再重新训练。



### C12. “vision abstractor”不是仓库中的自定义架构

- [ ] 已处理

**严重级别：Medium**

论文 §4.1 写使用 vision abstractor 作为 connector。代码直接加载官方 Qwen3-VL，并训练其原生 `visual`、`visual.merger` 和可能存在的 `deepstack_merger_list`：

- `qwen/src/model/load_model.py:70-86`
- `qwen/src/train/train_sft.py:44-58`
- 对 `qwen3_vl` 没有自定义 monkey patch：`qwen/src/model/load_model.py:43`

**修改建议：**

- [ ] 改成“使用 Qwen3-VL 原生 vision encoder 与 patch merger/connector”，除非另有未发布的 abstractor 实现。
- [ ] 不要把 backbone 的原生 merger 写成本文新增模块。



### C13. Stage II response 不是“两个 absolute token + 一个 relative token”的严格三 token 结构

- [ ] 已处理

**严重级别：Medium**

代码只断言五个 absolute level 各自带前导空格时恰为一个 token（`train_deqa.py:112-121`）。Relative label 没有对应的 token ID 配置或单-token 断言，而是固定自然语言关系短语的一部分：

```text
The quality of Image 1 is {level_a},
the quality of Image 2 is {level_b},
Image 1 is {relation_text} Image 2.
```

其中 relation 及 “to/than” 等文本均由普通 next-token CE 监督。

**修改建议：**

- [ ] 将论文中的 “relative-quality token” 改为 “relative-quality word/phrase supervised by autoregressive CE”。
- [ ] 如果确实需要一个专门位置，代码应显式配置五个 relative token ID 并检查 tokenization。



### C14. Absolute level 词表的顺序和大小写未按论文呈现

- [x] 已处理

**严重级别：Low**

- 论文：`{bad, poor, fair, good, excellent}`，低到高、小写。
- 配置：`[Excellent, Good, Fair, Poor, Bad]`，高到低、首字母大写。
- 代码用权重 `[5,4,3,2,1]` 与该倒序配置配套，因此连续得分方向本身是正确的。

**影响：**

这主要是复现描述问题，但大小写可能对应不同 tokenizer token，不能把两者视为字面相同的 closed vocabulary。

**修改建议：**

- [ ] 论文按代码写出真实 token 字符串及顺序：`[Excellent, Good, Fair, Poor, Bad]`，对应 anchor `[5,4,3,2,1]`。



### C15. “training-time conversion”只对 hard label 成立，soft label 是离线预计算

- [x] 已处理

**严重级别：Medium**

- Hard absolute label：`PairDatasetPrebuilt.__getitem__` 会按当前 pairs 中每个 dataset 的 min/max 在线计算，见 `pair_dataset.py:505-533`。
- Relative label：优先直接读取 JSON 的 `comparison`，只有缺失时才在线计算，见 lines 526-530。
- Soft target：直接读取 JSON 的 `level_probs`，见 lines 520-522。

论文 §3.2.1–3.2.2 将这些过程统一称为 during training，不够准确。

**修改建议：**

- [ ] 明确区分：hard verbal label 在线生成；comparison 和 soft distribution 在默认 prebuilt 协议中离线存储。
- [ ] 发布 pairs 生成脚本，避免核心公式只存在于论文而不在数据构造代码中。



### C16. 推理代码没有输出 Eq. (13) 所述预测方差

- [ ] 已处理

**严重级别：Low**

训练中的 `get_score()` 会内部计算 std，但公开 scalar eval：

- 只保存五个 level logits/probabilities：`iqa_eval_qwen.py:164-215`
- `cal_plcc_srcc.py:67-76` 只计算期望分数
- 不保存或报告预测方差

**修改建议：**

- [ ] 若论文要把不确定性作为模型输出能力，应在 eval JSON 中保存 `variance` 并提供校准实验。
- [ ] 否则把 Eq. (13) 后“quantifies predictive uncertainty”的陈述降为数学上可计算，而非当前评测实际使用的输出。



### C17. 论文六个 scalar IQA benchmark 与默认 eval 集合不完全一致

- [ ] 已处理

**严重级别：Medium**

论文 Table 7：KonIQ-10K、KADID-10K、PIPAL、LIVE-Wild、AGIQA-3K、CSIQ。

`scripts/eval_iqa.sh:48-59` 默认扫描 10 个 meta，包括 LIVE、CSIQ、KADID、BID、CLIVE、KonIQ、TID2013、SPAQ、AGIQA3K、PIPAL。存在即运行，不固定为论文六个集合。

**修改建议：**

- [ ] 提供 `eval_paper_table7.sh`，显式固定六个 meta 路径、split 文件 hash 和预处理参数。
- [ ] 当前通用 eval 脚本可保留，但不要作为 Table 7 的唯一复现入口。



### C18. 论文所述 Assistant 主评测在仓库中没有复现入口

- [ ] 已处理

**严重级别：High**

论文 Table 3–6 / Fig. 5 使用：

- Q-BenchSINGLE-A1
- Q-BenchPAIR-A1
- 2AFC-LMM
- MICBench_v2

当前仓库公开入口只覆盖 scalar IQA 的 `scripts/eval_iqa.sh`；没有上述四类评测的脚本、prompt 固件和 MICBench_v2 数据。

**修改建议：**

- [ ] 发布 frozen 评测目录和 exact prompts，包括 decoding、max tokens、答案解析、Thurstone 聚合和 API 模型版本。
- [ ] 对每个论文表格提供一个不可覆盖旧结果的独立 run manifest。

---



## 二、代码实现缺陷或复现隐患

以下问题不一定都是论文文字冲突，但会影响“代码实现了论文公式”的可信度。

### I1. `level_probs` 未归一化却作为 KL target

- [ ] 已处理

见 C5。建议在 dataset/collator 处检查：

```python
assert torch.all(level_probs >= 0)
level_probs = level_probs / level_probs.sum(dim=-1, keepdim=True)
```

但不能在不重新验证结果的情况下只做静默修复；该修改会改变训练目标。

### I2. Stage II 像素配置读取后被硬编码覆盖

- [ ] 已处理

`train_score.sh:85-86` 读取变量，lines 142-143 却使用字面量。环境变量和 YAML 看似可覆盖，实际无效。应改为变量引用并加 launcher 测试。

### I3. `level_prefix` 是无效配置

- [ ] 已处理

`train_deqa.py:123-131` 将 `level_prefix_ids` 写入 `DeQAConfig`，但 `modeling_deqa_qwen.py` 从未使用它；pair response 也硬编码另一种措辞。修改 `stage2_score.yaml:23` 不会改变训练文本或定位逻辑。

建议删除该无效参数，或让 dataset template 和 level position matching 真正引用它。

### I4. `max_seq_length` 没有执行

- [ ] 已处理

- `params.py:120-126` 定义默认 32768。
- `sft_dataset.py:317-318` 的截断调用被注释。
- Pair dataset 也不截断。

论文没有报告 max length，代码参数又不生效。长多图样本可能超过上下文或造成不可预期的显存波动。建议实现 image-aware truncation，且不得切断视觉 token span 或 assistant level token。

### I5. `lazy_preprocess` 没有消费者

- [ ] 已处理

脚本传入 `--lazy_preprocess True`，但 dataset 不读取该字段。当前 dataset 本身在 `__getitem__` 中处理样本，行为上近似 lazy，但该 flag 不控制任何分支。建议删除或真正实现。

### I6. 自动 resume 可能使“1 epoch”不等于全新 1 epoch 实验

- [ ] 已处理

- Stage I 只要输出目录存在 `checkpoint-*` 就 resume：`train_sft.py:228-231`。
- Stage II 默认 `auto_resume: true`：`train_deqa.py:348-372`。

复现实验应使用新的输出目录并记录 resume checkpoint，避免旧状态污染。

### I7. 训练与 scalar eval 的 prompt 和预处理没有统一固件

- [ ] 已处理

Stage II 训练是随机的双图问题模板，scalar eval 是单图：

```text
How would you rate the quality of this image?
The quality of the image is
```

单图推理本身合理，但论文必须报告该模板；同时应让图像预处理和 level token 配置来自同一份 frozen config。

### I8. 缺少数据构造和 split 防泄漏测试

- [ ] 已处理

仓库提供最终 annotation/pairs，但没有：

- Q-Align 0.5 分组实现
- Qwen3-VL-Plus teacher generation 脚本和 prompt 版本
- `level_probs` 生成实现
- 8K pair 采样实现
- MICBench 与训练图像 no-overlap 检查
- Table 7 train/test ID disjoint 检查

这些不是普通文档缺失，而是论文核心方法和实验有效性无法独立核验。

---



## 三、与代码一致的主要部分

为避免只列问题，以下陈述有明确实现依据：

- [ ] 两阶段总体流程一致：Stage I SFT 产生 assistant checkpoint，Stage II 默认从该目录初始化 rater。
- [ ] 基座为 `Qwen/Qwen3-VL-8B-Instruct`；论文应补全 `Instruct` 变体名称。
- [ ] Stage I assistant 使用标准 autoregressive next-token CE，prompt token 被 `IGNORE_INDEX` mask。
- [ ] 默认是 full-parameter fine-tuning，不是 LoRA；LLM、vision、merger 均未冻结。
- [ ] 动态分辨率方向一致，且 max pixel budget 均为 `1280 × 32 × 32`。
- [ ] Hard absolute quality label 基本按每个 dataset 的 min/max 均匀五分桶。
- [ ] Relative label 的 ±σ、±2σ 五档阈值结构与 Eq. (5) 同类，仅参照方向相反；在恰好 `-σ` 的边界上代码与论文区间闭合方式也有细小差异。
- [ ] Ground-truth pair probability 使用均值差及两个标注方差之和。
- [ ] Fidelity loss 的数学形式与 Eq. (16) 同类。
- [ ] Closed-set 五档 softmax用于连续分数提取，`[Excellent, Good, Fair, Poor, Bad]` 对应 `[5,4,3,2,1]`。
- [ ] Scalar IQA 评分通过五档 logits 的 closed-set softmax期望得到，随后用四参数 logistic fitting 计算 PLCC，并计算 SRCC。
- [ ] 默认 seed 为 42，scheduler 为 cosine，warmup ratio 为 0.03，精度为 bf16。

---



## 四、论文内部需要单独修正的问题

这些问题即使不参考代码，也需要作者统一口径。

- [ ] **KonIQ-only 与 multi-dataset ranking 冲突。** §3.2 写 Stage II 只有 KonIQ 8K，§4.3.2 却以多个 heterogeneous IQA datasets 为前提解释 dataset-wise ranking。
- [ ] **Fig. 3 的 “9K coarse + 60K fine” 与 Table 1 的 Teach2Compare 计数口径不清。** 应说明是图像组、reasoning items、QA items，还是筛选后的源图数。
- [ ] **§6.1 用一套超参概括两个顺序训练阶段。** 即使真实实验恰好同参，也应分别报告。
- [ ] **Eq. (11) 名称与形式。** 公式是 `KL(q || p)`，应明确是 KL/soft-label cross entropy，而不仅称 ordinal loss。
- [ ] **MICBench_v2 主观实验缺少复现细节。** 需要 observer 数量、招募/筛选、界面、投票规则、一致性阈值、模糊题过滤率和伦理说明。

---



## 五、无法从当前仓库核验的论文陈述

以下项目不应直接判为“代码错误”，但目前没有可审计实现或工件：

- [ ] Q-Pathway 19K source pool 的精确 ID 清单。
- [ ] Q-Align 模型版本、推理 prompt、分数归一化和 `0.5` coarse/fine 阈值代码。
- [ ] Teach2Compare 69K 图像来源的完整清单和 KADIS 合成参数。
- [ ] Qwen3-VL-Plus teacher 的 exact model revision、system prompt、temperature 和后处理。
- [ ] MICBench_v2 的 3,000 MCQ 文件、human validation 原始投票和 no-overlap 检查。
- [ ] Table 3–6、Fig. 5 的评测代码和原始预测。
- [ ] GPT-5.4、Claude-Sonnet-4.5 等 API 基线的固定版本、日期和解码设置。
- [ ] Table 8 每个消融模型的独立训练配置与 checkpoint。
- [ ] “4 NVIDIA H20 GPUs”只能由运行记录证明，代码仅默认使用四个 GPU ID，不能证明硬件型号。
- [ ] 论文所有结果是否来自当前 commit、当前 JSON 和当前默认配置；仓库没有 run manifest、日志或 checkpoint hash 将三者绑定。

---



## 六、建议的修订方案



### 方案 A：以当前仓库默认实现为准修改论文

至少需要：

- [ ] 将数据集改写为 Stage I 912,826 + Stage II 185,000，并披露额外 561,644 条语料。
- [ ] 将 Stage II 改为 KonIQ/KADID/LIVE/CSIQ/BID 多数据集训练，并重新解释 Table 7 和 Table 8。
- [ ] 把 relative label 改为 Image 1 relative to Image 2。
- [ ] 用真实 `level_probs` 生成公式替换 Eq. (9)，修复并披露非归一化问题。
- [ ] 将损失写成默认实际展开式，报告 `weight_next_token=0.05`、`weight_rank=1.0` 和固定预测方差。
- [ ] 分阶段重写超参数和像素预算。
- [ ] 把 vision abstractor 改为 Qwen3-VL 原生 vision encoder + merger。
- [ ] 修改 interleaved prompt 描述，使其符合各数据源真实模板。
- [ ] 基于这套配置重新检查所有表格和消融；旧结果不能自动沿用。



### 方案 B：以论文当前方法为准修改发布代码

至少需要：

- [ ] 新增 frozen Stage I YAML，仅包含 Table 1 的八个文件，并解决少 29 条的问题。
- [ ] 发布 KonIQ training split 的 8K pairs 及生成脚本。
- [ ] 按 `μ2-μ1` 生成第二张图相对第一张图的 relation。
- [ ] 按 Eq. (8)–(9) 在线或离线生成严格归一化、最多两项非零的 soft target。
- [ ] 在五档 logits 上做 closed-set KL。
- [ ] 按 Eq. (15) 使用预测方差，或删除该公式。
- [ ] 明确 λ、γ，并让代码总损失与 Eq. (12)、Eq. (17) 逐项一致。
- [ ] 将 epoch、global batch、LR 和 min pixels 设置为论文值，并提供不可被环境变量静默改变的 frozen `run.sh`。
- [ ] 发布论文四类 Assistant 评测和 Table 7/8 的专用脚本。
- [ ] 用新输出目录完整重跑，并记录 commit、环境、脚本 hash、数据 hash 和 checkpoint hash。



### 不建议的做法

- [ ] 只修改 README 而不确定论文结果究竟来自哪套协议。
- [ ] 把当前 `configs/*.yaml` 称为 canonical paper config，同时在论文保留另一套数值。
- [ ] 仅把 `pairs_180k.json` 改名为 8K/KonIQ；其内容和损失仍然不同。
- [ ] 静默归一化 `level_probs` 后继续引用旧结果。
- [ ] 以“可通过环境变量覆盖”为理由解释论文差异，但不提供当时实际 export 和日志。

---



## 七、提交前核对清单

- [ ] 确认 Table 3–8 对应的准确 checkpoint 与 commit。
- [ ] 确认真实 Stage I YAML 是否包含 `coinstruct_562k_t2c.json`。
- [ ] 确认真实 Stage II pairs 是 KonIQ 8K 还是五数据集 185K。
- [ ] 对训练和评测图像 ID 做逐数据集 overlap 检查。
- [ ] 保存所有 annotation、pairs、split 和脚本的 SHA-256。
- [ ] 将 Stage I/II 的 epoch、有效 global batch、分模块 LR、weight decay、warmup 和像素预算写入论文。
- [ ] 将 λ、γ 或实际 loss 权重写为数值。
- [ ] 统一 relative label 的方向。
- [ ] 验证所有 `level_probs` 非负且和为 1。
- [ ] 让 Eq. (9)、closed-set KL 和代码单元测试互相一致。
- [ ] 统一训练与推理的 processor 路径和像素预算。
- [ ] 发布 Q-Bench、2AFC-LMM、MICBench_v2 和 scalar IQA 的 frozen eval scripts。
- [ ] 为 Table 8 每个变体保存独立配置，禁止复用含旧 checkpoint 的输出目录。



## 最终判断

- [ ] 已处理

当前论文可以准确概括仓库的高层思想——Qwen3-VL 上的两阶段 comparative SFT 与 DeQA-style rating——但不能准确描述默认发布实现和数据。尤其是 C1、C2、C4、C5、C7、C9 属于会改变训练分布或优化目标的关键差异，应在声称代码复现论文前全部解决。C2 还直接影响 Table 7 的训练/测试关系，优先级最高。