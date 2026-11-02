# EmotionHallucination-Conflict-Correction

面向多模态大模型图文情绪冲突的检测与前置视觉矫正工程。系统先判断图像情绪与文本情绪是否冲突；发生冲突时只依据视觉事实裁决，否则使用 Qwen2.5-Omni 进行综合推理。

## 目录

```text
scripts/                 主实验脚本
scripts/visual_constrained_eval.py  视觉候选标签受限评分
data/sample_data/        前 10 条 JSON 样例，不含图片
demo/run_demo.py         单图片+文本交互 Demo
demo/streamlit_app.py    可视化 Demo（默认安全演示模式）
src/emotion_conflict/    可测试的路由策略库
docs/EXPERIMENT_MANIFEST.md  隔离数据划分和实测结果清单
docs/REPORT_STORY.md      报告叙事、指标口径和答辩顺序
docs/                    系统设计、课程对照和相关工作
tests/                   路由策略自动化测试
evidence/                已运行结果和审计证据
```

## 环境依赖

Python 3.10、PyTorch 2.7.1，以及 Transformers、PEFT、Datasets、TRL、Accelerate、Hugging Face Hub 和 Pillow。模型权重不随仓库提交；请根据本地环境自行配置路径。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 运行步骤

构建 MVSA-Single 真实数据：

```bash
python scripts/build_real_conflict_dataset.py --root /path/to/MVSA-Single --out ./data/generated
```

运行基线：

```bash
python scripts/baseline_test.py --model /path/to/Qwen2.5-Omni-7B --data ./data/generated/real_conflict_test_set.json --output ./outputs/baseline_results.json
```

LoRA 微调：

```bash
python scripts/finetune_conflict_detector.py --base /path/to/Qwen2.5-VL-7B --data ./data/generated --output ./models/conflict_detector_lora_v2
```

前置矫正评估：

```bash
python scripts/evaluate_with_correction.py --base /path/to/Qwen2.5-VL-7B --adapter ./models/conflict_detector_lora_v2 --data ./data/generated/real_conflict_test_set.json --baseline ./outputs/baseline_results.json --detector ./outputs/detector_results_v2.json --output ./outputs/correction_results.json
```

Demo：

```bash
python demo/run_demo.py --image /path/to/image.jpg --text "这真是令人开心的一天。" --vl-model /path/to/Qwen2.5-VL-7B --omni-model /path/to/Qwen2.5-Omni-7B --visual-adapter /path/to/visual_lora_balanced --text-model /path/to/text_emotion_tfidf.joblib
```

该命令默认运行当前双通道主方案；旧生成式 detector 兼容模式使用 `--mode legacy --adapter /path/to/conflict_detector_lora_v2`。

启动可视化 Demo：

```bash
pip install streamlit
PYTHONPATH=src streamlit run demo/streamlit_app.py
```

Demo 默认是确定性的安全演示模式，不会把占位输出混入实验结果。真实模型推理请使用 `demo/run_demo.py`，显式传入模型和 adapter 路径。

## 动态协同推理

`src/emotion_conflict/routing.py` 将检测结果和置信度分为三路：

| 条件 | 路由 | 行为 |
|---|---|---|
| 一致或低于门控阈值 | `omni` | 保留图文协同推理 |
| 冲突且置信度 >= 0.75 | `visual_only` | 物理隔离文本，视觉支路裁决 |
| 冲突且 0.55 <= 置信度 < 0.75 | `human_review` | 不强制覆盖，返回人工复核状态 |

检测器没有提供置信度时，系统不会伪造 1.0；冲突样本默认进入人工复核。这是为了避免把旧日志中的二分类标签误称为概率。

严格指标审计：

```bash
PYTHONPATH=src python scripts/audit_results.py \
  --correction /path/to/correction_results.json \
  --detector /path/to/detector_results_v2.json \
  --output ./outputs/audit_report.json
```

审计报告同时输出严格视觉对齐率和旧版宽松幻觉率。严格视觉偏离定义为 `final_emotion != visual_emotion`，避免在三分类冲突数据中“输出文本标签也被误判为正确”。

视觉候选标签评分实验：

```bash
python scripts/visual_constrained_eval.py \
  --model /path/to/Qwen2.5-VL-7B \
  --data /path/to/real_conflict_test_set.json \
  --output outputs/visual_constrained_results.json
```

该脚本只在推理后使用真实标签计算准确率；标签不会进入 Prompt 或候选选择。它记录候选概率、置信度和熵，可用于后续阈值门控。

固定置信度门控实验：

```bash
python scripts/evaluate_confidence_gate.py \
  --visual outputs/visual_constrained_results.json \
  --baseline /path/to/baseline_results_v2.json \
  --threshold 0.60 \
  --output outputs/confidence_gate_results.json
```

在当前100条真实冲突样本上，固定阈值0.60的门控结果为18%严格视觉对齐率，基线为17%。这是小幅真实改进，不应包装成显著提升。

冲突检测器也提供 `detector_constrained_eval.py`，直接比较“冲突/一致”单 token 概率，避免自由生成解析偏置；其 confidence 才可用于动态路由，不能把自由生成标签当作概率。

双通道检测改进：

```bash
python scripts/train_text_emotion.py \
  --conflicts /path/to/detector_conflicts_train.json \
  --controls /path/to/detector_controls_train.json \
  --output outputs/text_emotion_tfidf.joblib
python scripts/evaluate_dual_channel.py \
  --text-model outputs/text_emotion_tfidf.joblib \
  --visual outputs/visual_heldout_rows.json \
  --data /path/to/heldout_detector_eval.json \
  --output outputs/dual_channel.json
```

双通道规则是“独立视觉预测 != 独立文本预测 => 冲突”。在隔离153条样本上，检测F1为70.48%，召回78.43%；它是当前端到端系统的候选 detector。

最新的平衡隔离实验见 `evidence/stratified_results_summary.json` 和 `evidence/dual_channel_summary.json`：102条冲突评估集按视觉标签各34条，51条一致控制集各17条。视觉 LoRA 使用排除评估样本后的真实 MVSA 图像训练，平衡训练后视觉支路冲突对齐率39.22%，控制集37.25%；对应 Omni 分别为33.33%和19.61%。双通道 detector（独立文本分类器 + 视觉 LoRA disagreement）将端到端冲突对齐率提升到39.22%，控制集31.37%，detector F1=70.48%。旧生成式 detector 的21.67%结果作为失败对照保留。

## 实验结论摘要

基于 MVSA-Single 真实图文情感标注冲突集，历史报告中的宽松指标为：Qwen2.5-Omni-7B 基线 **58%**、简单 Prompt 预警 **60%**、前置视觉矫正 **52%**。扩展审计发现，三分类冲突样本中“输出文本标签”会被旧规则误算为非幻觉；因此新增严格指标 `final_emotion == visual_emotion`。对现有 100 条矫正结果重算后，严格视觉对齐率为 **16%**。两者必须并列报告，52% 不能单独作为视觉矫正成功的证据。本摘要不包含纯视觉或纯文本单模态实验。

## 数据来源声明

实验数据来源为 MVSA-Single，包含图像、推文文本、图像情感标签和文本情感标签。本仓库只提供前 10 条 JSON 元数据样例，不分发完整图片数据；请从合法来源准备完整数据，并根据本地环境自行配置路径。

## 已知限制

模型权重、完整数据、训练缓存和实验日志不纳入仓库。不同 GPU、Transformers 版本和生成参数可能造成运行时间及输出差异。

本工程不是心理诊断工具，不能判断真实心理状态，也不能把视觉表情当作唯一真实情绪。MVSA-Single 的标签是图像/文本情感极性；反讽、强颜欢笑、文化差异和标签噪声需要人工复核。项目目前的 58% -> 60% -> 52% 结果来自既有 100 条冲突测试集，扩展的置信度门控和一致样本代价实验必须在重新运行后再报告，不能用预估数字替代。

## 与课程要求的对应

- 问题定义：冲突文本进入主模型后可能劫持视觉情绪判断。
- 代码实现：数据构建、LoRA 检测器、动态路由、评估审计和 Demo。
- 价值与伦理：低置信度不强制视觉裁决，保留人工复核路径；不把系统输出用于心理诊断。
- 范式讨论：质疑“只需增加 Prompt/反思文本即可自动平衡模态”的默认假设，实验保留了预警从 58% 恶化到 60% 的负结果。
- 可复现性：模型权重和完整图片不入库，路径通过命令行传入；样例元数据仅用于演示数据格式。

相关工作定位和课程逐项对照见 `docs/RELATED_WORK.md`、`docs/SYSTEM_DESIGN.md` 和 `docs/COURSE_ALIGNMENT.md`。课程说明要求第1章问题定义和第3章伦理判断由团队成员独立完成；本仓库不代替这些署名责任。

提交前检查：

```bash
make compile
make test
make check
```

报告证据整理见 `docs/REPORT_EVIDENCE.md`；第1章和第3章的人工填写工作表见 `docs/PROBLEM_EVIDENCE_WORKSHEET.md` 与 `docs/ETHICS_WORKSHEET.md`。
提交前评分自检见 `docs/RUBRIC_SELF_CHECK.md`。它不会预测教师分数，但会阻止缺少关键产出、样例字段或实验审计文件的提交包。
报告叙事和答辩证据顺序见 `docs/REPORT_STORY.md`。
