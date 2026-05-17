# 实验课四 Pre 大纲：多模态大语言模型

---

## Task 1：MLLM 微调策略探索

### Slide 1. 从 Baseline 出发

**核心叙事**

- Task 1 在 GQA VQA 上微调 InternVL2-2B，输入为 `image + question`，输出为短答案。
- 任务限制明确：训练与推理只能使用 GQA 的 `image + question + answer`，不允许使用 bbox，也不允许引入 LVIS/COCO。
- 实验一开始暴露出关键现象：VQA 精确匹配评测不仅考察视觉理解，也高度依赖答案格式。

**Baseline 与 Prompt 观察**

| 设置 | Total Acc. | BBox Acc. | No-BBox Acc. | 备注 |
| --- | ---: | ---: | ---: | --- |
| InternVL2-2B, no bbox, 早期 prompt | 0.247 | 0.2487 | 0.2222 | 输出格式大量错误 |
| Qwen3-VL-8B, no bbox, 早期 prompt | 0.685 | 0.6777 | 0.7937 | 输出分布天然符合格式 |
| InternVL2-2B, no bbox, 规范化短答案 prompt | 0.685 | 0.6756 | 0.8254 | 约束格式后显著提升 |
| Qwen3-VL-8B, no bbox, 规范化短答案 prompt | 0.679 | 0.6692 | 0.8254 | 约束格式后基本稳定 |

开始跑实验就用了参考代码的 Prompt，其实都已经跑到 Task3 了，才发现改 Prompt 就能带来巨大提升，又从 Task1 开始把所有实验重跑了一遍。

原Prompt：`Question: {question}\nAnswer with the shortest correct answer only. `
新Prompt：`Question: {question}\nAnswer with the shortest correct answer only. The answer must be a simple short answer, usually one word, a short phrase, yes/no, or a number.`

![格式错误样例](images/task1_format_error_2386907.jpg)
- 图片：images/gqa/2386907.jpg
- 问题：Is the grass brown or green?
- 标准答案：green
- 模型输出：The grass is green.

---

### Slide 2. 模块冻结实验

**核心叙事**

- InternVL2-2B 可以拆解为视觉编码器、跨模态连接器和语言解码器三类模块。
- Task 1 固定数据、prompt 与训练流程，仅改变可训练模块，用以观察不同模块对 VQA SFT 的贡献。

**配置设计**

| 配置 | 可训练模块 | 可训练参数 | 占总参数比例 | 实验含义 |
| --- | --- | ---: | ---: | --- |
| A | Connector | 12.60M | 0.5710% | 只学习跨模态对齐与格式适配 |
| B | Connector + Language | 1.90B | 86.2173% | 标准 instruction tuning 风格 |
| C | Vision + Connector | 316.61M | 14.3537% | 视觉表征适配，不更新语言模型 |
| D | Vision + Connector + Language | 2.21B | 100.0000% | 全参数微调，作为性能上限候选 |

![原Prompt训练wandb图](images/task1_wandb.png)

这个 wandb 记录是在 5090 上使用原 Prompt 训练的，通过充分的手动超参搜索，统一使用 bs32 lr 1e-5 1024 samples，因此开始 loss 都比较高，但几步就能几乎到0，收益基本都是格式带来的，没有真正的能力提升。


---

### Slide 3. 主结果：全参数最优，但收益不与参数量线性相关

**实验设置**

- 模型：InternVL2-2B。
- 数据：GQA VQA 训练集，Task 1 约束下不使用 bbox，新 Prompt。
- 统一设置：`batch size = 16`，`learning rate = 5e-6`，`512 training samples`。
- Prompt：使用规范化短答案 prompt。
- 评估：验证集 `n = 1000`，报告 total / bbox subset / no-bbox subset；其中 bbox subset 仅用于事后分析，Task 1 推理不输入 bbox。

**重跑主结果**

| 配置 | Total Acc. | BBox Subset | No-BBox Subset | Trainable | Peak Mem. | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Connector | 0.684 | 0.6756 | 0.8095 | 0.5710% | 18.38 GiB | 141.9 s |
| B Connector + Language | 0.701 | 0.6926 | 0.8254 | 86.2173% | 19.39 GiB | 176.4 s |
| C Vision + Connector | 0.688 | 0.6798 | 0.8095 | 14.3537% | 20.41 GiB | 229.5 s |
| D Full | **0.707** | **0.6980** | **0.8413** | 100.0000% | 21.66 GiB | 282.0 s |

**分析要点**

- D_full 取得最高准确率，说明全参数微调仍具有最强的整体适配能力。
- B 接近 D，说明语言模型侧的 instruction tuning 对短答案 VQA 有稳定贡献。
- A 只训练 0.5710% 参数仍达到 0.684，表明轻量跨模态对齐具备较高性价比。
- C 未超过 B，说明在当前样本规模下，单独更新视觉编码器并不比更新语言模型更有效。

---

### Slide 4. Infra 与探索性实验

磨刀不误砍柴工，先优化 Infra 和进行充分的探索性实验总是有很大收益的，比如我们一开始就没有对 Prompt 进行充分探索，导致后面完全从头来过浪费了很多时间。

**Infra 修改记录**

- D_full 需要开启 `gradient_checkpointing` 控制显存，否则会 OOM
- 使用累积梯度更新，这样可以灵活设置 `batch size`
- D_full 512 samples 下原始速度：297s
- 发现每次图片都需要预处理，尝试 CPU 与 GPU 并行：282s
- 开启 Flash Attention 后：186s，如果没有优化将多出 60% 训练时间

**D_full 超参探索，原 Prompt**

| 设置 | Total | Mem | Time | 简短解释 |
| --- | ---: | ---: | ---: | --- |
| bs128 lr4e-5 | 0.588 | 21.65 GiB | 588.4 s | 官方参数起点，学习率偏大，效果明显不足 |
| bs64 lr2e-5 | 0.642 | 21.65 GiB | 585.3 s | 降低学习率后提升，但仍未进入稳定区间 |
| bs32 lr2e-5 | 0.641 | 21.65 GiB | 590.1 s | 缩小 batch 未带来额外收益 |
| bs32 lr1e-5 | 0.691 | 21.65 GiB | 589.5 s | 进入较优学习率区间，准确率明显跃升 |
| bs32 lr5e-6 | 0.692 | 21.65 GiB | 589.8 s | 与 1e-5 接近，说明该区间较稳定 |
| bs32 lr1e-6 | 0.693 | 21.65 GiB | 594.3 s | 继续降低学习率收益有限 |
| bs16 lr5e-6 | 0.695 | 21.65 GiB | 592.5 s | 更小 batch 略有提升，但时间基本不变 |
| bs16 lr5e-6, 512 samples | 0.699 | 21.65 GiB | 296.6 s | 样本减半后效果不降反升，提示任务主要是格式适配 |
| bs8 lr1e-6, 512 samples | 0.695 | 21.65 GiB | 296.2 s | 更小 batch 与低学习率没有明显优势 |
| bs8 lr1e-6, 256 samples | 0.664 | 21.65 GiB | 150.7 s | 样本过少导致性能下降，低成本但不可作为主配置 |

---

## Task 2：从视觉标注数据构造 Instruction Tuning 数据

### Slide 5. 数据构造：从非 QA 标注到 Instruction Tuning

**A/B 数据构造 Pipeline**

| 方案 | 输入给教师模型的信息 | 输出类型 | 清洗重点 | 目的 |
| --- | --- | --- | --- | --- |
| A text teacher | caption + objects + bbox 文本提示 | 多样 instruction-response | 去除 bbox 坐标、过滤非法 task type、去重 | 验证仅凭结构化标注能否生成可用 instruction |
| B vision teacher | image + caption + objects + bbox 文本提示 | 多样 instruction-response | 同上，额外依赖教师看图纠偏 | 比较图文联合驱动是否带来更高质量视觉样本 |

**生成约束**

- 每张图最多生成 8 条样本，覆盖 caption、物体识别、计数、密集摘要、空间关系等类型。
- 物体类别、caption 和 bbox 可以作为教师模型的中间提示，但最终 question 中不能出现 bbox 坐标或 bbox 列表。
- 输出必须是 JSON array，每条样本包含 `task_type`、`question`、`answer`。
- 对生成结果做格式过滤、bbox 泄漏过滤、非法 task type 过滤和去重。


```

  A 版原始提示词

  Objects in the image:
  {fmt_objects(row, keep_bbox=True)}
  Image caption: {row.get("caption", "")}
  
  Please generate 8 diverse multimodal instruction-response samples.

  Allowed task types:
  - image_captioning
  - object_recognition
  - object_counting
  - dense_visual_summary
  - spatial_reasoning

  1. Output a JSON array. Each item must contain task_type, question, and answer.
  2. task_type must be one of the allowed task types above. Prefer diverse task types but do not invent new types.
  3. question and answer must be in English.
  4. For object_recognition, object_counting, and spatial_reasoning, write question as a visual question answering question and make answer a short
   answer, usually one word, a short phrase, yes/no, or a number.
  5. The final question must not contain bbox coordinates or bbox lists.
  6. Output JSON only. Do not add explanations.
  
  B 版原始提示词
  
  Objects in the image:
  {fmt_objects(row, keep_bbox=True)}
  Image caption: {row.get("caption", "")}
  
  Please look at the image and use the annotations as hints. Generate 8 diverse multimodal instruction-response samples.
  
  （其余和 A 版相同）
  
```

---

### Slide 6. 数据质量：A/B 差异主要体现在视觉细节可靠性

**核心叙事**

- 方案 A 只看结构化文字，因此更依赖 caption 与标注覆盖范围；优点是生成稳定，缺点是容易忽略图像细节。
- 方案 B 同时看图和结构化提示，因此更可能生成计数、空间关系和细粒度视觉描述；但也会产生更复杂、更长或不符合短答案要求的样本。
- 两种方案的共同风险是：它们服务于“通用 instruction tuning”，不一定天然匹配 GQA 的短答案 VQA 评测分布。

**生成规模与分布**

| 数据版本 | 输入规模 | 生成/保留样本 | 类型分布 |
| --- | ---: | ---: | --- |
| A text teacher | 256 images × 8 | 1789 | caption / object / count / summary / spatial |
| B vision teacher | 256 images × 8 | 1522 | caption / object / count / summary / spatial |

**A/B 样例对比**

![118838 A/B 对比](images/task2_case_118838_labeled.jpg)

| 图像 | 来源 | 示例 | 观察 |
| --- | --- | --- | --- |
| `000000118838.jpg` | A text | `How many gravestones are visible in the image?` -> `3` | 仅凭标注即可生成正确计数，但问题主要来自类别和数量转写 |
| `000000118838.jpg` | B vision | `A bird sits on a gravestone, with two other gravestones and grassy surroundings visible.` | 看图后描述包含 grass / surroundings 等额外视觉细节，信息更丰富 |

![388903 异常计数样例](images/task2_case_388903_labeled.jpg)

| 图像 | 来源 | 示例 | 观察 |
| --- | --- | --- | --- |
| `000000388903.jpg` | A text | `How many suitcases are visible in the image?` -> `1` | 结构化标注转写更保守，答案形式符合短答案要求 |
| `000000388903.jpg` | B vision | `How many visible instances of the object 'apple' are there in the image?` -> `many` | 看图后引入更真实的场景内容，但答案变成模糊量词，无法适配精确匹配评测 |

**讲述重点**

- A/B 的核心差异是教师模型是否拥有图像证据：B 的视觉细节更丰富，但噪声也更难控制。
- 直接追求丰富 instruction 会引入 caption/summary 类数据，这些数据可能提升通用能力，但未必服务于短答案 VQA。
- 因此 Task 2 的关键问题是：非 QA 标注生成的数据是否能真正迁移到 GQA 验证分布。

---

### Slide 7. 训练结果：构造数据没有带来稳定增益，暴露分布迁移问题

**训练设置**

- 模型：InternVL2-2B，`D_full` 全参数微调。
- 训练参数：`bs16 lr5e-6 1024 samples`，早期使用旧 Prompt，后面全部重跑了。
- 对比方式：仅使用 Task2 数据、Task1 GQA 与 Task2 数据混合、以及 Task1 baseline。混合方法为每条依次使用不同数据源的样本，但控制总样本数为 1024。

**重跑 1024-sample 结果**

| 训练数据 | Total | BBox | No-BBox | 相对 baseline_1024 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| Task1 baseline_1024 | **0.704** | **0.6926** | **0.8730** | 0.000 | GQA 原生监督仍是最强基线 |
| A_only_1024 | 0.691 | 0.6830 | 0.8095 | -0.013 | 纯合成数据不足以替代 GQA QA |
| B_only_1024 | 0.687 | 0.6788 | 0.8095 | -0.017 | 图文教师数据仍存在目标分布偏差 |
| AB_1024 | 0.691 | 0.6830 | 0.8095 | -0.013 | 合并 A/B 增加数量但未提高质量 |
| Task1 + A_1024 | 0.697 | 0.6884 | 0.8254 | -0.007 | 与 GQA 混合后仍略降 |
| Task1 + B_1024 | 0.699 | 0.6905 | 0.8254 | -0.005 | B 更接近 baseline，但没有正增益 |
| Task1 + AB_1024 | 0.695 | 0.6862 | 0.8254 | -0.009 | 多源合并带来噪声累积 |

**结论**

- Task 2 数据构造本身是可行的：能够从非 QA 标注中批量生成 instruction-response 样本，并覆盖 caption、识别、计数、摘要、空间关系等类型。
- 但在验证集上没有超过 Task1 baseline，说明合成数据与 GQA 原生 VQA 之间仍存在分布差异。
- 主要瓶颈不是样本数量，而是问题风格、答案粒度、视觉 grounding 可靠性与目标验证集的一致性。
- 因为怀疑是分布问题，我们后续又生成了更偏向 GQA-style short-answer VQA 的数据，但实验上仍没有带来有效提升，说明仅靠改写问题形式不足以消除数据迁移差距。
- 这个结果也为 Task 3 提供动机：相比继续增加弱监督合成 QA，显式使用 bbox grounding 可能更直接提升定位敏感问题。

---

## Task 3：Bounding Box 信息的利用

### Slide 8. 从隐式视觉理解到显式 Grounding

**核心叙事**

- Task 1/2 的共同瓶颈是定位敏感问题：模型需要在图像中找到正确对象，再回答颜色、属性、数量或空间关系。
- Task 3 允许显式使用 GQA bbox，因此我们把问题从“让模型自己隐式注意到目标区域”转化为“把目标区域作为额外 grounding 信号输入”。
- 验证集中绝大多数样本都有 bbox：`937/1000` 个样本含 bbox，因此 bbox-aware 方法的收益主要会反映在 bbox subset 上。
- 本任务同时探索了文本化 bbox、图像可视化 bbox、bbox-aware prompt 和 hard case 再训练。

**验证集 bbox 分布**

| bbox 数量 | 样本数 | 占比 |
| ---: | ---: | ---: |
| 0 | 63 | 6.3% |
| 1 | 394 | 39.4% |
| 2 | 451 | 45.1% |
| 3 | 91 | 9.1% |
| 4 | 1 | 0.1% |

**Prompt 侧方案**

- `no_bbox`：只输入原问题，作为 Task1 baseline，不显式提供定位信息。
- `bbox_prompt`：在问题中加入类别和 bbox 坐标，让模型在文本上下文中获得目标位置。
- `color_prompt`：在问题中加入颜色框说明，让模型通过颜色词引用视觉区域。
- `color_bbox_prompt`：同时加入颜色框说明和 bbox 坐标，结合离散颜色锚点与连续坐标。

**图像侧方案**

- `original`：使用原图，不引入额外视觉噪声。
- `same_box`：所有 bbox 画同色框，提示目标区域，但难以区分多个框。
- `color_box`：不同 bbox 使用不同颜色，支持颜色引用和多目标区分。
- `color_box_label`：彩色框 + 编号/类别标签，同时显式化视觉区域和语义标签。

案例：id=0953938

```
  no_bbox：
  Question: Is the mattress to the right or to the left of the table the lamp is on?
  Answer with the shortest correct answer only.

  bbox_prompt：
  Objects in the image:
  1. lamp: normalized_bbox=[0.172, 0.397759, 0.276, 0.661064]
  2. table: normalized_bbox=[0.174, 0.616246, 0.41, 0.935574]
  3. mattress: normalized_bbox=[0.348, 0.492997, 0.944, 0.966387]

  Question: Is the mattress to the right or to the left of the table the lamp is on?
  Answer with the shortest correct answer only.
  
  color_bbox_prompt：
  Objects in the image:
  1. red box: lamp, normalized_bbox=[0.172, 0.397759, 0.276, 0.661064]
  2. blue box: table, normalized_bbox=[0.174, 0.616246, 0.41, 0.935574]
  3. green box: mattress, normalized_bbox=[0.348, 0.492997, 0.944, 0.966387]

  Question: Is the mattress to the right or to the left of the table the lamp is on?
  Answer with the shortest correct answer only.
```

![same_box](images/task3_0953938_same_box.jpg)
![color_box](images/task3_0953938_color_box.jpg)
![color_box_label](images/task3_0953938_color_box_label.jpg)

---

### Slide 9. Prompt × 图像的二维消融

**核心叙事**

- 先不训练模型，直接比较不同推理输入形式，可以判断 bbox 信息本身是否有效。
- 横轴是图像处理方式，纵轴是 prompt 方式；每个格子展示 `Total / BBox / No-BBox`。
- 对 InternVL2-2B 而言，最稳定的是 `bbox_prompt + original image`；单纯画框通常会造成视觉噪声。
- 对 Qwen3-VL-8B 而言，bbox prompt 达到 `0.842--0.845`，说明强模型能更充分利用显式 grounding。

**InternVL2-2B 原始模型二维消融：Total / BBox / No-BBox**

| Prompt \ Image | original | same_box | color_box | color_box_label |
| --- | --- | --- | --- | --- |
| `no_bbox` | `0.685 / 0.6756 / 0.8254` | `0.668 / 0.6574 / 0.8254` | `0.670 / 0.6596 / 0.8254` | `0.796 / 0.7940 / 0.8254` |
| `bbox_prompt` | **`0.799 / 0.7972 / 0.8254`** | `0.781 / 0.7780 / 0.8254` | `0.781 / 0.7780 / 0.8254` | `0.790 / 0.7876 / 0.8254` |
| `color_prompt` | `0.798 / 0.7962 / 0.8254` | `0.782 / 0.7791 / 0.8254` | `0.772 / 0.7684 / 0.8254` | `0.779 / 0.7759 / 0.8254` |
| `color_bbox_prompt` | `0.798 / 0.7962 / 0.8254` | `0.758 / 0.7535 / 0.8254` | `0.756 / 0.7513 / 0.8254` | `0.790 / 0.7876 / 0.8254` |

**教师模型参考**

| 模型与输入 | Total | BBox | No-BBox | 观察 |
| --- | ---: | ---: | ---: | --- |
| Qwen3-VL-8B, no_bbox | 0.679 | 0.6692 | 0.8254 | 强模型不依赖 prompt 格式，但缺少定位仍受限 |
| Qwen3-VL-8B, bbox_prompt | 0.842 | 0.8431 | 0.8254 | 显式 bbox 大幅提升 bbox subset |
| Qwen3-VL-8B, color_bbox_prompt + label | 0.844 | 0.8453 | 0.8254 | 接近直接 prompt 上限 |

**阶段性结论**

- 即使不画框也不给坐标信息如 `color_prompt + original image` 也能获得非常不错的效果，因为有些答案就是从标签中提取的。
- `color_box_label` 在 `no_bbox` 下有效，说明标签是关键；但一旦已有 bbox prompt，额外画框通常不增益。
- 因此后续训练主线选择 `bbox_prompt`，并把可视化方案作为消融分析。

---

### Slide 10. 训练策略：从模块冻结到 bbox-aware 全量训练

**核心叙事**

- 在 bbox prompt 推理已经有效的基础上，继续训练的目标是让模型稳定适配 bbox-aware 输入格式。
- 第一轮比较 A/B/C/D 冻结配置，发现全参数 D_full 最高，但 B_connector_language 已经非常接近。
- 第二轮在 Task1/D_full 基础上继续训练 bbox 数据，探索样本量、学习率和 hard case 采样。

**bbox 上训练的模块冻结结果，512 samples**

| 配置 | Total | BBox | No-BBox | Trainable | Mem | Time | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A Connector | 0.855 | 0.8581 | 0.8095 | 0.5710% | 19.17 GiB | 273.6 s | 轻量对齐也能利用 bbox |
| B Connector + Language | 0.869 | 0.8719 | 0.8254 | 86.2173% | 19.36 GiB | 376.3 s | 接近全参数，语言侧适配关键 |
| C Vision + Connector | 0.853 | 0.8538 | 0.8413 | 14.3537% | 21.20 GiB | 498.5 s | 视觉侧训练收益较小 |
| D Full | **0.872** | **0.8751** | 0.8254 | 100% | 21.63 GiB | 603.2 s | 作为后续强基线 |

**训练方式探索**

| 实验 | Total | BBox | No-BBox | 解释 |
| --- | ---: | ---: | ---: | --- |
| no_bbox 训练, bbox 推理 | 0.832 | 0.8324 | 0.8254 | 只在推理阶段加 bbox 已明显超过 Task1，但不如训练推理一致 |
| Task1/D_full + bbox 512 | 0.867 | 0.8687 | 0.8413 | 继续训练有效，但低于从头 bbox-aware D_full |
| same_box | 0.681 | 0.6702 | 0.8413 | 纯视觉同色框失败 |
| color_box | 0.686 | 0.6756 | 0.8413 | 彩色框仍不足 |
| color_box_label | 0.822 | 0.8218 | 0.8254 | 标签带来明显改善 |
| same_box_bbox_prompt | 0.860 | 0.8613 | 0.8413 | 文本 bbox 是核心，视觉框辅助有限 |
| color_box_color_prompt | 0.852 | 0.8527 | 0.8413 | 颜色 prompt 有效但低于 bbox_prompt |
| color_box_color_prompt_without_bbox | 0.860 | 0.8613 | 0.8413 | 去掉 bbox 坐标后不降，说明标签/颜色锚点比坐标更关键 |
| 4096 samples, bs64 lr5e-6 | 0.874 | 0.8762 | 0.8413 | 扩大样本小幅提升但成本很高 |

**COT 与 hard case 尝试**

- COT：参考 VoCoT，构造 bbox-aware 推理提示，让模型先根据 bbox 描述目标区域和推理路径，再输出最终短答案。
- COT 评估：在 100 个训练样本上比较 Qwen3-VL 直接 `bbox_prompt` 与 COT，直接回答为 `0.840 / 0.8404 / 0.8333`，COT 为 `0.810 / 0.8085 / 0.8333`。
- COT 结论：多步文本推理会增加输出复杂度，不适合当前短答案精确匹配任务。
- Hard case：先用 `task1/D_full + bbox_prompt` 跑 8000 条训练数据，正确 `6436/8000 = 0.8045`，筛出 `1564` 条错误样本。
- Hard case 单训：用这些错误样本训练时验证集下降到 `0.834 / 0.8346 / 0.8254`，说明只看难例会破坏原有分布。
- 后续策略：将 hard cases 与常规 Task1/Task3 样本混合，避免过拟合错误分布。

```
COT 提示词：
  Objects in the image:
  1. {category}: normalized_bbox={bbox}
  2. {category}: normalized_bbox={bbox}
  Question: {question}
  Answer the question by reasoning briefly from the image and the listed object bounding boxes (x_left_top, y_left_top, x_right_down,
  y_right_down). Use the boxes only as localization clues; do not invent new boxes. When an object is important to your reasoning path, annotate it
   with its given bounding box. Do not mention the prompt or say that the object information was provided.
  Your response should follow this format:
  Thought: <brief reasoning process>
  Answer: <short answer>
```

---

### Slide 11. 最终方案：bbox prompt + hard case 混合训练

**核心叙事**

- 最终选择最稳定的路径：以 `bbox_prompt` 作为统一输入格式，使用 D_full，在 bbox 数据上进行更长训练。
- 先用原始模型在 8000 条训练数据上做 bbox 推理，保留错误样本作为 hard cases：`6436/8000 = 0.8045`，筛出 `1564` 条。
- 直接只训 hard cases 会不稳定，因此先进行全量训练。
- 最优验证结果达到 `0.906 total / 0.9082 bbox / 0.8730 no-bbox`，相比 Task1 baseline 和 Task2 baseline 都有显著提升。

**长训练与最终结果**

训练均使用 `D_full`，`warmup_cosine`。

| 训练设置 | Total | BBox | No-BBox | 观察 |
| --- | ---: | ---: | ---: | --- |
| baseline：bbox_prompt | 0.799 | 0.7972 | 0.8254 | 原始模型仅推理输入 bbox 已大幅提升 |
| 8000 samples, bs16 lr1e-5  | 0.890 | 0.8922 | 0.8571 | 长训练显著提升 bbox subset |
| task1/3 hard 3128 samples, bs16 lr5e-6  | 0.894 | 0.8954 | 0.8730 | task1 + task3 hard case 混合有效 |
| shuffle 8000 samples, bs64 lr5e-6 / best | **0.906** | **0.9082** | **0.8730** | 最优 checkpoint 出现在 step 100，提交 Leaderboard 85.3，第一名 |

训练曲线：
![task3_wandb](images/task3_wandb.png)



**结论**

- Task 3 的关键提升来自显式 grounding，而不是更复杂的 instruction 数据或更长的推理链。
- bbox prompt 的收益最稳定：它把定位问题转化为模型可读的结构化上下文。
- 这个任务可能比较简单，COT之类的方案收益不大
