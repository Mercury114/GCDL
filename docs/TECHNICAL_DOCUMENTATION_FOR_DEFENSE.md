# 技术文档：图文情绪冲突的生成前检测与动态协同推理

> **用途**：本文件是本项目答辩、报告撰写和答辩稿生成的技术事实底稿。可以整体提供给其他 AI 生成答辩稿、PPT 文案或讲稿，但必须要求其遵守第 1 节的事实边界，不能将组件消融、历史实验和端到端主实验混为一谈。
>
> **项目名称**：多模态大模型的可信性——面向图文情绪冲突的前置检测与动态协同推理系统
>
> **英文名称**：Trustworthiness of Multimodal Large Language Models: A Pre-generation Detection and Dynamic Collaborative Reasoning System for Vision-Text Emotion Conflict

## 1. 一页式事实摘要

### 1.1 要解决的问题

多模态大模型面对“一张图片表达负向情绪、文字却表达正向情绪”这类图文情绪冲突输入时，可能被文本先验主导，最终综合结论偏离图像侧标注的情感极性。项目不把此问题表述为“读取人物真实心理状态”，而是将其**严格操作化**为：在 MVSA-Single 的图像/文本情感极性标签发生冲突时，最终输出是否与图像侧标签一致。

本项目的目标不是宣称已经解决所有多模态幻觉，而是验证一个工程假设：

> 在生成最终综合回答前，先独立估计视觉与文本情感；当二者不一致时，采用视觉优先的受限决策，是否能提高冲突样本上的视觉标签对齐率？

### 1.2 当前主方案

当前主方案由三个相互独立、可审计的步骤组成：

1. **视觉通道**：以 Qwen2.5-VL-7B 为基座，对视觉情感极性进行 LoRA 微调；只根据图像预测 `positive`、`neutral`、`negative`。
2. **文本通道**：使用独立的 TF-IDF + Logistic Regression 文本分类器，只根据原始文本预测同一三分类标签。
3. **路由与裁决**：视觉预测与文本预测不一致，则认定为冲突，进入 `visual_only` 路由，以视觉通道的受限标签作为最终输出；二者一致，则保留 Qwen2.5-Omni 的图文综合推理结果。

这里的“前置”指的是：**检测与路由发生在生成最终综合答案之前**。它不同于让主模型先生成一个回答、再用另一个 Prompt 解释或纠错的后置做法。

### 1.3 当前主结果：冻结、分层、真实标注的 MVSA-Single 留出集

| 对象 | 数量与分布 | 用途 |
|---|---:|---|
| 冲突留出集 | 102 条；图像标签 `negative/neutral/positive` 各 34 条 | 核心有效性评估 |
| 一致控制集 | 51 条；图像标签三类各 17 条 | 评估正常输入是否受损 |
| 总留出集 | 153 条 | 路由器与端到端系统评估 |
| 训练隔离 | 全部 153 个评估 ID 从视觉 LoRA 和文本分类器训练中剔除 | 防止训练/测试泄漏 |

| 指标 | Qwen2.5-Omni 基线 | 当前动态系统 | 变化 |
|---|---:|---:|---:|
| 冲突集严格视觉对齐率 | 33.33% (34/102) | **42.16% (43/102)** | **+8.82 个百分点**，相对 +26.47% |
| 冲突集宽松幻觉率（辅助指标） | 42.16% (43/102) | **31.37% (32/102)** | **-10.78 个百分点** |
| 一致控制集严格视觉对齐率 | 19.61% (10/51) | **29.41% (15/51)** | **+9.80 个百分点** |
| 全部 153 条严格视觉对齐率 | 28.76% (44/153) | **37.91% (58/153)** | **+9.15 个百分点** |

当前双通道冲突检测器在同一冻结集上的结果：Accuracy 52.94%，Precision 63.16%，Recall 70.59%，F1 66.67%。它将 114/153 条输入路由至 `visual_only`，39/153 条保留给 Omni。

**必须诚实说明的边界**：静态视觉支路在全部 153 条上的严格对齐率为 41.18%（63/153），高于动态系统的 37.91%（58/153）。因此，动态路由不是“所有指标都优于静态视觉分类”的方案；它的工程价值在于只在两个独立通道意见不一致时触发视觉优先，仍为 39 条输入保留图文协同的 Omni 路径。这是可解释的行为与效用权衡，不是绝对支配关系。

### 1.4 可引用的论文定位

本项目以以下研究问题为背景：

> Bohao Xing, Xin Liu, Guoying Zhao, Chengyu Liu, Xiaolan Fu, and Heikki Kälviäinen. 2026. *EmotionHallucer: Evaluating Emotion Hallucinations in Multimodal Large Language Models*. ICLR 2026, OpenReview.

本工程不声称复现或替代该论文的全部评测协议。它将“图文情绪冲突下的可信性”落实为一个可运行的、可审计的生成前检测与路由系统，并用 MVSA-Single 的真实图文情感标注验证该系统。

## 2. 问题形式化与指标定义

### 2.1 输入、标签与冲突定义

每个样本记为：

\[
x=(I,T,y_v,y_t)
\]

其中：

- \(I\)：图像；
- \(T\)：文本；
- \(y_v \in \{negative, neutral, positive\}\)：MVSA-Single 图像侧情感极性标签；
- \(y_t \in \{negative, neutral, positive\}\)：MVSA-Single 文本侧情感极性标签。

冲突真值定义为：

\[
c=\mathbb{1}[y_v \ne y_t]
\]

需要注意，\(y_v\) 与 \(y_t\) 是数据集提供的**情感极性注释**，并不是对人物内在心理状态的医学或心理学判断。答辩时应使用“视觉侧标签”“文本侧标签”“视觉事实对齐”的工程术语，避免把模型输出包装成对人类真实情绪的断言。

### 2.2 主指标：严格视觉对齐率

系统最终输出为 \(\hat y\)。对冲突样本，主指标定义为：

\[
\text{Strict Visual Alignment}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat y_i=y_{v,i}]
\]

该指标刻画：在图文标签互相冲突时，系统是否最终遵循视觉侧标注。它是本项目最重要的可信性指标。

### 2.3 为什么旧的“宽松幻觉率”不足以单独作为主指标

历史实验中曾使用如下宽松定义：

\[
\text{Loose Hallucination}=\mathbb{1}[\hat y \notin \{y_v,y_t\}]
\]

该定义会将“模型完全跟随文本标签 \(y_t\)”视为非幻觉。对于三分类的冲突样本，这会掩盖最关键的文本主导问题。例如：图像标签为 `negative`、文本标签为 `positive`，模型输出 `positive` 时，宽松定义认为它命中了一个已有标签，但从“视觉事实优先”的研究目标看，这正是需要诊断的偏差。

因此：

- **严格视觉对齐率是当前主指标**；
- 宽松幻觉率只保留为历史结果复核和辅助观察；
- 两类指标可以并列展示，但绝不能相互替换，也不能把不同数据划分下的值直接比较。

### 2.4 组件指标与端到端指标

冲突检测器的二分类指标为：

\[
\text{Precision}=\frac{TP}{TP+FP}, \quad
\text{Recall}=\frac{TP}{TP+FN}, \quad
F1=\frac{2PR}{P+R}
\]

当前冻结集上的混淆矩阵为：TP=72、FP=42、FN=30、TN=9。该集合中冲突样本占 2/3，因此 F1=66.67% 不能脱离 Precision/Recall 和类别比例单独解读。它说明检测器具备一定的冲突召回能力，但假阳性较多，仍有明显改进空间。

## 3. 系统架构

### 3.1 端到端数据流

```text
输入：图像 I + 文本 T
          |
          +-------------------------------+
          |                               |
          v                               v
视觉情感支路                        文本情感支路
Qwen2.5-VL + Visual LoRA            TF-IDF + Logistic Regression
仅输入 I                             仅输入 T
输出 y_hat_v                         输出 y_hat_t
          |                               |
          +---------------+---------------+
                          v
              是否 y_hat_v != y_hat_t ?
                    /               \
                  是                 否
                  |                  |
                  v                  v
         visual_only 路由        Omni 路由
         最终 y_hat=y_hat_v      Qwen2.5-Omni(I,T)
                  \                 /
                   +-------+-------+
                           v
     记录：预测、路由、最终标签、评估结果和指标版本
```

### 3.2 路由规则

令视觉支路预测为 \(\hat y_v=f_v(I)\)，文本支路预测为 \(\hat y_t=f_t(T)\)。冲突预测和最终输出为：

\[
\hat c=\mathbb{1}[\hat y_v\ne\hat y_t]
\]

\[
\hat y=
\begin{cases}
\hat y_v, & \hat c=1 \\ 
f_{omni}(I,T), & \hat c=0
\end{cases}
\]

关键约束是：\(y_v\)、\(y_t\) 只在**路由之后**参与离线评分，不进入线上预测或路由规则。`scripts/evaluate_token_supervised_system.py` 通过 ID 对齐视觉支路预测、文本预测与基线预测，并在代码中以 `visual_prediction != text_prediction` 得到 `detector_prediction`。

### 3.3 为什么叫“动态协同推理”

这里的协同不是让两个模型各生成一长段解释再投票，而是以两个互相独立的模态判断为前置证据：

- 一致时，系统承认图文信息相互支持，保留 Qwen2.5-Omni 的整合能力；
- 不一致时，系统不再试图靠自然语言提醒克服文本先验，而是改变后续可见信息：最终决策只使用视觉支路受限标签；
- 因而协同发生在“**先比较、再选择推理路径**”的控制层，而非依赖最终生成答案自我反思。

该设计可被理解为生成前的 guardrail，或一个可解释的 mixture-of-experts 路由器，但它并未训练一个端到端的 MoE 门控网络。答辩时不应把它描述为“大规模专家混合模型”。

## 4. 数据工程与防泄漏设计

### 4.1 数据来源

数据源为 MVSA-Single。每条可用样本包含：图像文件、文本文件、图像标签 `img_label`、文本标签 `text_label`。工程脚本将原始整数标签映射为：

| 原始值 | 工程标签 |
|---:|---|
| 0 | `negative` |
| 1 | `neutral` |
| 2 | `positive` |

构建脚本为 `scripts/build_stratified_eval.py`。它只保留图像和文本文件都存在的条目，并按图像标签进行分层采样，避免原始 100 条冲突样本标签失衡导致“模型只预测多数类也看似有效”的问题。

### 4.2 冻结分层留出集

`build_stratified_eval.py` 以如下原则生成固定评测集：

1. 将 `img_label != text_label` 的样本作为冲突候选；
2. 将 `img_label == text_label` 的样本作为一致控制候选；
3. 对图像标签的三个类别分别取固定数量；
4. 输出冲突集与控制集 JSON；
5. 后续训练集构建阶段读取这些 ID，并显式排除。

最终 102 条冲突样本按视觉标签均衡为 34/34/34；51 条控制样本均衡为 17/17/17。这里的“均衡”针对视觉侧标签，因为主指标评估的是视觉对齐。

### 4.3 训练/测试隔离

`scripts/build_visual_train.py` 读取两份冻结评测 JSON，汇总全部 153 个 ID，随后从可训练样本中排除这些 ID。文本分类器也只使用同一非评测部分训练。这个隔离原则避免模型以样本 ID、图像或文本记忆的形式在测试时获得不公平优势。

答辩时可直接说明：

> 先冻结、再训练；冻结集中的每一个 ID 都从视觉 LoRA 与文本分类器训练数据中删除。真实标签只在推理结束后用于计算分数。

### 4.4 数据边界和版权边界

仓库不分发完整 MVSA-Single 图像，仅保留少量元数据样例和结果摘要。完整数据需从合法来源取得，模型权重、原始图片、缓存、训练检查点不进入 Git 仓库。该策略同时减少版权、隐私与仓库体积风险。

## 5. 方法迭代：从失败到当前方案

### 5.1 版本 0：原始生成式基线

最初使用 Qwen2.5-Omni-7B 对图像和文本共同推理，以自由生成的整体情绪作为结论。历史 100 条冲突集上记录的宽松幻觉率为 58%。该数值属于旧数据划分和旧指标口径，不能与当前冻结集的 33.33% 严格对齐率放进同一张横向性能表。

### 5.2 版本 1：Prompt 预警，得到有价值的负结果

第一次改进是在 Prompt 前加入“视觉与文本情绪存在冲突，请以视觉事实为准”之类的自然语言警告。历史宽松幻觉率从 58% 上升到 60%，没有改善。

这个负结果的工程含义是：当主模型已经同时接收图像与强文本先验时，只增加一段说明性文字无法保证模型改变证据权重；甚至可能让生成过程更不稳定。因此，项目保留 `58% -> 60%` 作为**失败对照**，而不是删除不利实验。

### 5.3 版本 2：早期 LoRA 检测器与问题暴露

随后尝试将 Qwen2.5-VL 做 LoRA 微调，使其输出“冲突/一致”。早期训练曾出现 Loss=10.03，这暴露出任务和损失形式并不匹配：模型被按通用语言建模方式监督，训练目标包含大量与二分类决策弱相关的 token，不能证明其学到了冲突判断。

进一步审计还发现：即便训练损失在后续版本下降，若对 Assistant 的答案 token 之后的聊天结束符、换行等 token 也计算损失，模型仍可通过预测这些容易的结尾 token 获得较低平均损失。这种损失下降不等价于情感类别判别提升。

### 5.4 关键方法修正：只监督类别答案 token

当前视觉 LoRA 的训练脚本为 `scripts/finetune_visual_emotion_token_supervised.py`。训练格式为：

```text
System: You are a visual sentiment classifier. Classify the image only.
User: [image]
      Output exactly one label: positive, neutral, or negative.
Assistant: <positive | neutral | negative>
```

核心修正有四点：

1. 先用 Qwen chat template 生成无答案的 prompt 和带答案的 full sequence；
2. 计算 `prefix_length`，即 Assistant 类别 token 前的长度；
3. 将所有位置的 `labels` 初始化为 `-100`；
4. **只**将 Assistant 的单个类别 token 恢复为真实 token ID，prompt、结束符和换行全部保持 `-100`。

等价地，训练损失为：

\[
\mathcal{L}=-\log p_\theta(y_v\mid I,\text{classification prompt})
\]

而不是对整段对话的平均语言模型损失。脚本还在训练前验证 `positive`、`neutral`、`negative` 各自都恰好是一个 tokenizer token；若不满足会直接报错，避免标签截断或多 token 对齐错误。

LoRA 配置为 `r=16`、`lora_alpha=32`、`lora_dropout=0.05`，目标模块为语言注意力层中的 `q_proj` 和 `v_proj`，使用学习率 `2e-5`、3 个 epoch。视觉基座模型使用 Qwen2.5-VL-7B；适配器训练不把 LoRA 直接绑在视觉编码器模块上。

### 5.5 为什么从“生成式检测器”改成“双通道分歧检测”

旧生成式检测器要直接回答“是否冲突”，它同时承担视觉理解、文本理解、标签比较和自然语言生成解析，错误来源难以拆开。当前方案把问题拆成：

\[
\text{image} \rightarrow \hat y_v, \quad \text{text} \rightarrow \hat y_t, \quad \hat c=\mathbb{1}[\hat y_v\ne\hat y_t]
\]

这样做的优点是：

- 每个支路都有明确输入边界，便于审计；
- 可直接查看视觉预测、文本预测、分歧和最终路由；
- 不需从长自然语言输出中猜测冲突标签；
- “是否冲突”的定义与任务定义一致，即情感极性预测不一致。

文本通道采用 TF-IDF 的原因不是声称传统模型比大模型先进，而是为了让文本侧成为一个轻量、确定、可复现且与视觉侧独立的对照通道。`scripts/train_text_emotion.py` 使用词级 1--2 gram 和字符级 3--5 gram TF-IDF 特征，接入带类别平衡权重的 Logistic Regression。

### 5.6 为什么“预警”改为“前置矫正”

两者的本质差异如下：

| 方案 | 作用时机 | 是否仍把冲突文本交给最终生成模型 | 风险 |
|---|---|---|---|
| Prompt 预警 | 主模型生成时 | 是 | 文本先验仍可主导；警告文字本身也会扰动生成 |
| 前置视觉矫正 | 最终生成前 | 冲突时否 | 依赖视觉支路质量；可能误路由 |

当前方案的改进不是“更强烈地提醒模型”，而是通过控制流改变证据可见性。若检测到分歧，最终裁决阶段不再读取冲突文本，因此文本无法继续影响该路由的最终标签。这也是项目最清晰的工程创新点。

## 6. 实验设计与结果解读

### 6.1 比较对象的正确区分

| 版本 | 数据划分/口径 | 应如何使用 |
|---|---|---|
| 历史 Omni / Prompt 预警 / 前置矫正：58% / 60% / 52% | 旧 100 条冲突集，宽松幻觉率 | 用作迭代史和负结果，不能当当前主结论 |
| 平衡视觉 LoRA：39.22% 冲突、37.25% 控制 | 冻结集，旧训练目标 | 组件消融 |
| Token-only 静态视觉 LoRA：41.18% 冲突、41.18% 控制 | 冻结集，当前视觉支路 | 强静态组件基线 |
| Token-only 双通道动态系统：42.16% 冲突、29.41% 控制 | 冻结集，端到端主方案 | **当前主结论** |

这张表是答辩时最容易被问到的地方。不能从其中挑选不同口径的“最好数值”拼成一张表，也不能把静态视觉支路的 41.18% 写成动态系统的总体准确率。

### 6.2 主表：端到端系统与 Omni 基线

| 评估维度 | Omni 基线 | 双通道动态系统 | 绝对变化 | 解读 |
|---|---:|---:|---:|---|
| 冲突严格视觉对齐 | 34/102 = 33.33% | 43/102 = **42.16%** | **+8.82 pp** | 对核心问题有效 |
| 冲突宽松幻觉率 | 43/102 = 42.16% | 32/102 = **31.37%** | **-10.78 pp** | 辅助指标，与严格指标方向一致 |
| 一致控制严格视觉对齐 | 10/51 = 19.61% | 15/51 = **29.41%** | **+9.80 pp** | 在此数据集上未观察到控制集退化 |
| 总体严格视觉对齐 | 44/153 = 28.76% | 58/153 = **37.91%** | **+9.15 pp** | 混合输入的总体收益 |

相对提升仅用于补充叙述：\((42.16-33.33)/33.33=26.47\%\)。答辩优先说“绝对提升 8.82 个百分点”，因为它更不易误导。

### 6.3 检测器表现与路由成本

| 项目 | 当前值 | 含义 |
|---|---:|---|
| Accuracy | 52.94% | 受冲突类别比例影响，不能独立判断优劣 |
| Precision | 63.16% | 被路由为冲突的样本中，约 63% 确为冲突 |
| Recall | 70.59% | 真实冲突中，约 71% 被捕获 |
| F1 | 66.67% | Precision 与 Recall 的折中 |
| `visual_only` 路由 | 114/153 | 视觉优先被频繁触发 |
| Omni 路由 | 39/153 | 仍保留部分图文协同推理 |

假阳性意味着一致样本也可能被视觉优先覆盖；假阴性意味着真实冲突仍被交给 Omni。这正是动态系统不能等同于静态视觉分类器、且总体分数低于静态视觉支路的原因。

### 6.4 结果为何有说服力

本结果的说服力来自实验约束，而不是绝对百分比有多高：

1. 使用真实 MVSA-Single 图像与文本标签，不用合成标签计算主指标；
2. 使用标签分层的冻结留出集，避免视觉类别不均衡；
3. 所有 153 个评估 ID 从两条训练支路中排除；
4. 路由阶段不读取真值，真值只用于事后评分；
5. 保留 Prompt 预警失败、旧损失设计缺陷和静态分支更高总体分数等不利证据；
6. 主结果、历史结果和组件消融都以不同证据文件保存，防止事后混用。

### 6.5 不能夸大的地方

以下表述应避免：

- “解决了多模态情绪幻觉”；
- “准确率提高到 42.16%”；42.16% 是冲突集严格视觉对齐率，不是普适分类准确率；
- “检测器 F1=66.67%，因此系统可靠”；F1 必须连同类别比例、Precision、Recall 与路由代价解释；
- “视觉标签就是真实人物情绪”；数据集标签只是定义本项目评估的操作化标准；
- “前置矫正一定优于视觉-only”；当前静态视觉支路总体更高，动态路由强调的是保留多模态路径的权衡。

## 7. 核心实现说明

### 7.1 视觉 LoRA 的 token-only 监督

核心代码位于 `scripts/finetune_visual_emotion_token_supervised.py` 的 `collate` 函数。其关键思想可抽象为：

```python
labels = torch.full_like(item["input_ids"], -100)
labels[prefix_length:prefix_length + len(answer_ids)] = \
    item["input_ids"][prefix_length:prefix_length + len(answer_ids)]
item["labels"] = labels
```

其中 `prefix_length` 是没有 Assistant 答案时 prompt 的长度，`answer_ids` 是 `positive`、`neutral` 或 `negative` 的单 token ID。PyTorch 的 CrossEntropy 对标签为 `-100` 的位置忽略计算，因此损失只来自类别答案。

实现中还使用延迟加载：`Dataset.__getitem__` 只返回样本元数据，实际图像打开、processor 编码和张量拼接发生在 `collate`。这避免在构造数据集时将所有高分辨率图片 tensor 一次性常驻内存。

### 7.2 受限候选评分，而不是自由生成解析

`scripts/visual_constrained_eval.py` 不让视觉模型自由生成长文本，而是读取最后一个位置对三个固定候选 token 的 logits：

\[
p(k\mid I)=\frac{\exp(z_k)}{\sum_{j\in\{neg,neu,pos\}}\exp(z_j)}
\]

然后在三个候选中取最大概率标签。这样能避免“模型输出中文、英文或一段解释，后处理无法可靠抽取标签”的解析偏差，也会记录候选概率、置信度和熵。

### 7.3 基线输出的多语言归一化

基线模型可能输出中文词（如“开心”“中性”“悲伤”）或英文词（如 `positive`、`neutral`、`negative`）。`scripts/normalize_baseline.py` 使用受控别名表归一化输出，并保留原始预测字段。没有匹配到的输出会标为 `unknown`，而不是强行归到任意类别。

这个细节很重要：若采用字符串精确匹配，模型语义正确但输出语言不同会被错误记为失败，曾导致不合理的纯模态低分。归一化修复的是**评估解析**，不是修改模型预测。

### 7.4 双通道路由的可审计输出

`scripts/evaluate_token_supervised_system.py` 为每条样本保存：

- 样本 ID；
- 图像、文本与冲突真值标签；
- `visual_prediction`；
- `text_prediction`；
- `detector_prediction`；
- `route`（`visual_only` 或 `omni`）；
- Omni 预测、最终预测；
- 严格视觉对齐和宽松幻觉率布尔字段。

这使得任何一条结果都可以回溯：“为什么这个样本被路由？”“最终标签来自视觉还是 Omni？”“分数来自哪条真实标签？”这比只输出一个总准确率更适合作为可信性工程证据。

## 8. 可复现实验流程

以下命令使用占位路径。模型、完整 MVSA 数据和 LoRA 权重不纳入 Git 仓库，需按本地环境替换路径。

### 8.1 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

项目验证环境为 Python 3.10、PyTorch 2.7.1、Qwen2.5-VL-7B 与 Qwen2.5-Omni-7B 的本地权重。使用单卡时可设置 `CUDA_VISIBLE_DEVICES=0` 并传入 `--single-gpu`，将可见 GPU 映射为程序内的 CUDA 设备 0。

### 8.2 建立冻结评测集

```bash
python scripts/build_stratified_eval.py \
  --root /path/to/MVSA-Single \
  --output outputs/stratified_eval \
  --per_visual_label 34 \
  --controls_per_label 17
```

### 8.3 建立训练集并训练视觉 LoRA

```bash
python scripts/build_visual_train.py \
  --root /path/to/MVSA-Single \
  --eval outputs/stratified_eval/stratified_conflict_eval.json \
         outputs/stratified_eval/stratified_control_eval.json \
  --output outputs/visual_train.json

# 只对训练集完成类别均衡，不能重采样或改动冻结评测集。
python scripts/balance_visual_train.py \
  --input outputs/visual_train.json \
  --output outputs/visual_train_balanced.json

# 执行 token-only LoRA 训练。
CUDA_VISIBLE_DEVICES=0 python scripts/finetune_visual_emotion_token_supervised.py \
  --model /path/to/Qwen2.5-VL-7B \
  --data outputs/visual_train_balanced.json \
  --output models/visual_lora_token_supervised \
  --epochs 3 \
  --learning-rate 2e-5 \
  --single-gpu
```

### 8.4 训练文本分类器

```bash
python scripts/build_detector_train.py \
  --root /path/to/MVSA-Single \
  --conflict-eval outputs/stratified_eval/stratified_conflict_eval.json \
  --control-eval outputs/stratified_eval/stratified_control_eval.json \
  --conflicts-out outputs/detector_conflicts_train.json \
  --controls-out outputs/detector_controls_train.json

python scripts/train_text_emotion.py \
  --conflicts outputs/detector_conflicts_train.json \
  --controls outputs/detector_controls_train.json \
  --output models/text_emotion_tfidf.joblib
```

这里的训练 JSON 必须来自与冻结 153 个 ID 不相交的样本。运行前应再次核查 ID 交集为空。

### 8.5 产生视觉支路预测与 Omni 基线

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/visual_constrained_eval.py \
  --model /path/to/Qwen2.5-VL-7B \
  --adapter models/visual_lora_token_supervised \
  --data outputs/stratified_eval/stratified_conflict_eval.json \
  --output outputs/visual_conflict_eval.json \
  --single-gpu

# 对控制集重复一次，得到 visual_control_eval.json。

python scripts/normalize_baseline.py \
  --input outputs/omni_baseline_raw.json \
  --output outputs/omni_baseline_normalized.json
```

Omni 基线原始输出必须在路由前独立生成，并保留原始文本和归一化结果。不能先看到标签、再修改 Prompt 或挑选输出。

### 8.6 端到端评估

```bash
python scripts/evaluate_token_supervised_system.py \
  --visual-conflict outputs/visual_conflict_eval.json \
  --visual-control outputs/visual_control_eval.json \
  --text-model models/text_emotion_tfidf.joblib \
  --baseline outputs/omni_baseline_normalized.json \
  --output outputs/token_supervised_dual_channel.json
```

该命令会检查视觉结果与基线结果的样本 ID 集合完全相同；若 ID 不一致，直接报错，避免悄悄地在不同子集上比较模型。

### 8.7 项目自身检查

```bash
make compile
make test
make check
```

已包含的自动测试覆盖路由策略：一致输入保留 Omni、高置信冲突走视觉、置信度不足时转人工复核、缺失置信度不能被伪装成确定修正等。

## 9. Demo 说明

### 9.1 真实模型 CLI Demo

```bash
python demo/run_demo.py \
  --image /path/to/image.jpg \
  --text "这真是令人开心的一天。" \
  --vl-model /path/to/Qwen2.5-VL-7B \
  --omni-model /path/to/Qwen2.5-Omni-7B \
  --visual-adapter /path/to/visual_lora_token_supervised \
  --text-model /path/to/text_emotion_tfidf.joblib \
  --single-gpu
```

Demo 会打印视觉预测、文本预测、冲突状态、路由选择和最终结果。真实模型模式需要本地模型权重与足够显存；建议在有 GPU 的服务器运行。单卡 48GB 环境已验证两种路径均可运行：视觉冲突时输出 `visual_only`；两个支路一致时释放视觉模型显存后调用 Omni。

### 9.2 Streamlit 展示与真实实验的隔离

`demo/streamlit_app.py` 默认是确定性安全展示模式，适合演示路由交互，不应用其占位输出计算实验指标。真实实验结果仅来自 CLI 脚本和 `evidence/` 下的审计文件。答辩时应主动说明这种隔离，体现对 Demo 可视化与模型评估边界的重视。

### 9.3 建议的三分钟演示流程

1. 展示一条冲突样本：视觉支路与文本支路预测不同，系统透明地显示 `visual_only`；
2. 展示一条一致样本：两支路预测相同，显示 `omni` 路由；
3. 展示一行路由记录，说明真值不参与路由；
4. 展示主结果表，强调冻结、分层、训练隔离；
5. 主动展示 58% 到 60% 的 Prompt 预警失败，说明方案来自工程迭代，而不是只报告正结果。

## 10. 常见答辩问题与建议回答

### Q1：为什么不用一个大模型端到端判断冲突？

**回答要点**：可以，但早期生成式检测器把视觉理解、文本理解、冲突比较与自由生成混在一起，难以审计，也容易受到输出解析影响。双通道方法先把两个模态各自的极性判断显式化，再以不一致定义冲突，能够逐样本检查每一层错误。它牺牲了端到端的简洁性，换取了更强的可解释性和可复现性。

### Q2：为什么文本通道用传统 TF-IDF，而不是再用一个大模型？

**回答要点**：目标不是比较谁更“先进”，而是构造独立、轻量、确定的文本侧证据。若两边都依赖同一生成式模型，错误模式可能高度相关，分歧信号会变弱。TF-IDF + Logistic Regression 易复现、可固定随机性，也清楚地保证文本支路不看图像。

### Q3：为什么动态系统总体不如静态视觉支路？那动态还有意义吗？

**回答要点**：这是已披露的真实权衡。静态视觉支路对“始终按视觉标签评估”的任务更直接，因此总体达到 41.18%；动态方案为 39 条预测一致的输入保留 Omni 图文综合路径，目标是在可信性控制与保留多模态功能之间平衡。动态方案在核心冲突集上达到 42.16%，高于静态视觉支路 41.18%，但差异只有 1 条样本，不能宣称显著优于静态方法。下一步应在独立验证集上校准阈值和路由策略。

### Q4：42.16% 似乎不高，能证明系统有效吗？

**回答要点**：不能据此宣称模型已经可靠，但它在完全冻结、分层、真实标注的冲突集上相对 Omni 提升 8.82 个百分点。对于 102 条样本，这是从 34 条正确增加到 43 条正确。项目的贡献是把“预警无效”的问题转化为可审计的前置路由，并如实给出上限与失败模式，而不是制造高分。

### Q5：检测器 Accuracy 只有 52.94%，为什么 F1 有 66.67%？

**回答要点**：留出集刻意包含 102 条冲突、51 条一致，冲突占 2/3。Accuracy 受类别比例影响；F1 聚焦正类冲突检测。必须同时报告 Precision=63.16%、Recall=70.59%、TP/FP/FN/TN 和路由数，不能只报 F1。

### Q6：为什么严格视觉对齐是主指标？文本也可能是合理信息。

**回答要点**：这是本项目的研究假设，不是唯一正确的社会语义判断。在已知图文标签冲突的压力测试中，项目选择视觉侧标注作为“视觉事实锚点”，以测量文本是否不恰当地覆盖图像证据。若任务目标改为整体社交语义理解，应设计另一套主指标，而非直接套用本项目的严格视觉对齐率。

### Q7：数据是否是合成的？会不会泄漏？

**回答要点**：主实验使用 MVSA-Single 的真实图像和文本情感标注；历史工程早期曾用合成流程验证管线，但当前主结果完全来自真实数据。最终 153 个评估 ID 被明确排除出视觉 LoRA 和文本分类器训练；路由规则不读取真值标签。

### Q8：你们与 EmotionHallucer 的关系是什么？

**回答要点**：EmotionHallucer 提供了“多模态大模型情绪幻觉需要被系统评测”的研究背景。本项目没有宣称完整复现论文，而是围绕其痛点实现一个工程原型：前置检测、动态路由、严格指标审计和可运行 Demo。

## 11. 进一步优化路线：应做什么，不应做什么

### 11.1 可以在后续工作中严格验证的改进

1. **三分法路由与阈值校准**：利用视觉候选概率与熵，在验证集上设置“Omni / visual-only / human-review”三路阈值，报告覆盖率、人工复核率和准确率的 Pareto 曲线；阈值不能在最终测试集上调。
2. **类别重平衡与负类增强**：当前视觉支路在 `negative` 视觉类别上仍是显著短板。可以补充负向视觉样本、使用 class-balanced loss 或 focal loss，并按每类报告 precision/recall，不能只优化整体分数。
3. **更强但保持独立的文本支路**：可比较冻结的句向量分类器或小型文本 Transformer；需要固定训练集与测试集，避免为了提高数字反复在留出集调参。
4. **置信度校准**：对视觉候选 softmax 采用 temperature scaling，并用 Expected Calibration Error、风险-覆盖率曲线验证“高置信度才强制视觉路由”的合理性。
5. **多重随机划分与置信区间**：当前主评测集规模为 153，建议使用多个固定随机种子或 bootstrap 置信区间，报告平均值和方差，而不是只强调单次最大结果。
6. **人工误例审查**：抽样检查视觉标签噪声、反讽文本和抽象图片，记录典型的检测假阳性、假阴性和视觉分类失败，不将标签噪声简单归咎于模型。
7. **延迟与成本评估**：统计视觉路由、Omni 路由、人工复核路径的平均响应时间、显存占用和吞吐量，明确可信性收益的系统成本。

### 11.2 当前不应做的“优化”

- 不能在最终冻结集上不断调阈值、再报告最高的一次；
- 不能删除 Prompt 预警失败或静态视觉分支更高总体分数；
- 不能把历史 58/60/52 与当前 33.33/42.16 混合成同一个提升链；
- 不能使用评测真值来决定单条样本的路由；
- 不能把未重新运行的“预期提升”写入实验结果；
- 不能把 Demo 的规则占位输出计入模型实验分数。

这些限制不是削弱故事，而是本项目“可信性”主题本身的必要条件。高质量工程答辩应展示：系统如何防止自己在评估中产生不可审计的乐观偏差。

## 12. 报告与答辩建议结构

### 12.1 8 页 PPT 的建议逻辑

1. **问题场景**：图文情绪冲突，文本先验可能掩盖视觉信息；
2. **研究动机**：引用 EmotionHallucer 作为背景，说明项目聚焦前置防护；
3. **真实数据与协议**：MVSA-Single、102 冲突 + 51 控制、标签分层、训练隔离；
4. **失败迭代**：Prompt 预警 58% 到 60%，说明“提醒”不等于控制；
5. **系统架构**：双通道独立预测、分歧检测、动态路由；
6. **关键技术修正**：token-only loss，解释为什么低 loss 不一定代表学到分类任务；
7. **主结果与消融**：33.33% 到 42.16%，同时展示动态/静态的真实权衡；
8. **Demo、局限与下一步**：展示一条路由记录，说明负类、样本规模和阈值校准仍待改进。

### 12.2 30 秒结论版本

> 我们没有把情绪幻觉当成只能靠提示词解决的问题。基于 MVSA-Single 的真实标注，我们先发现简单预警会使历史宽松幻觉率从 58% 恶化到 60%。随后将系统拆成独立视觉和文本判断：分歧时在生成前切换到视觉受限裁决，一致时保留 Omni。为了避免训练损失虚低，我们只监督 Assistant 的类别 token，并在 153 条完全隔离、标签分层的留出集上评估。结果显示，冲突样本的严格视觉对齐率从 33.33% 提升到 42.16%，增加 9 条正确样本。我们同时披露了检测器假阳性、负类薄弱和静态视觉支路更高总体分数等局限，因此结论是“前置路由带来真实但有限的可信性增益”，而不是宣称解决了所有情绪幻觉。

### 12.3 可直接提供给答辩稿生成 AI 的提示词

```text
请基于下方技术事实生成一份 8 分钟中文本科工程导论答辩稿，语言自然、结构清楚，包含：问题、数据与评测协议、两轮失败/改进、系统架构、实验结果、Demo 讲解、局限和结论。

严格约束：
1. 不能编造任何指标、显著性检验、训练数据量、延迟或对比方法；
2. 当前主结果只能写为：MVSA-Single 冻结分层留出集，102 条冲突、51 条一致控制，所有 153 个评估 ID 从训练中排除；
3. 主指标是“严格视觉对齐率”，冲突集从 33.33% (34/102) 提升至 42.16% (43/102)，绝对提升 8.82 个百分点；
4. 辅助宽松幻觉率从 42.16% 降至 31.37%，不可替代主指标；
5. 历史 58% -> 60% -> 52% 属于旧 100 条冲突集的宽松指标，只能作为迭代历史和 Prompt 预警负结果，不能与当前主结果直接比较；
6. 必须承认局限：动态系统总体严格对齐 37.91%，低于静态视觉支路 41.18%；当前检测器 F1=66.67%、Precision=63.16%、Recall=70.59%；MVSA 标签是情感极性注释，不是人物心理状态真值；
7. 不要说“解决了情绪幻觉”，应说“在该冻结协议下实现了有限且可审计的视觉事实对齐改善”；
8. 把贡献表述为：生成前的双通道分歧检测、视觉优先动态路由、token-only 监督修正、真实数据与防泄漏审计。

技术事实如下：
[在这里粘贴本文件第 1、3、5、6、10、12 节]
```

## 13. 证据与代码索引

| 内容 | 仓库位置 |
|---|---|
| 当前主结果摘要 | `evidence/token_supervised_dual_channel_summary.json` |
| 实验协议与版本边界 | `docs/EXPERIMENT_MANIFEST.md` |
| 报告叙事与证据顺序 | `docs/REPORT_STORY.md` |
| 系统架构 | `docs/SYSTEM_DESIGN.md` |
| 历史指标审计 | `evidence/metrics_audit.json` |
| 数据构建 | `scripts/build_stratified_eval.py` |
| 训练集隔离 | `scripts/build_visual_train.py` |
| 视觉 LoRA token-only 训练 | `scripts/finetune_visual_emotion_token_supervised.py` |
| 文本支路训练 | `scripts/train_text_emotion.py` |
| 视觉受限候选推理 | `scripts/visual_constrained_eval.py` |
| 基线归一化 | `scripts/normalize_baseline.py` |
| 当前端到端评估 | `scripts/evaluate_token_supervised_system.py` |
| 真实模型 CLI Demo | `demo/run_demo.py` |
| 路由策略与自动化测试 | `src/emotion_conflict/`、`tests/` |
| AI 协作与人工修正记录 | `AI_USAGE_LOG.md` |

## 14. 最终结论

本工程最有价值的结果不是某个孤立的高分，而是一条完整且可追溯的技术链路：在真实图文情感标注上发现 Prompt 预警无效，识别出生成式监督和宽松指标的评估缺陷，修正为只监督情感类别 token 的视觉 LoRA，并通过独立视觉/文本预测的分歧在生成前进行路由。在冻结、分层且训练隔离的 MVSA-Single 评测集上，系统将冲突样本的严格视觉对齐率从 33.33% 提升到 42.16%。

该提升是有限的，仍受检测器假阳性、负类识别不足、数据集标签噪声和样本规模约束。正因为这些局限均被保留、量化并与正结果共同呈现，本项目能够将“多模态可信性”从一个抽象口号落实为可运行、可复核、可继续迭代的工程原型。
