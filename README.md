# EmotionHallucination-Conflict-Correction

面向多模态大模型图文情绪冲突的检测与前置视觉矫正工程。系统先判断图像情绪与文本情绪是否冲突；发生冲突时只依据视觉事实裁决，否则使用 Qwen2.5-Omni 进行综合推理。

## 目录

```text
scripts/                 主实验脚本
data/sample_data/        前 10 条 JSON 样例，不含图片
demo/run_demo.py         单图片+文本交互 Demo
```

## 环境依赖

Python 3.10、PyTorch 2.7.1，以及 Transformers、PEFT、Datasets、TRL、Accelerate、Hugging Face Hub 和 Pillow。模型权重不随仓库提交；请根据本地环境自行配置路径。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
python demo/run_demo.py --image /path/to/image.jpg --text "这真是令人开心的一天。" --vl-model /path/to/Qwen2.5-VL-7B --omni-model /path/to/Qwen2.5-Omni-7B --adapter /path/to/conflict_detector_lora_v2
```

## 实验结论摘要

基于 MVSA-Single 真实图文情感标注冲突集，Qwen2.5-Omni-7B 基线幻觉率为 **58%**；简单 Prompt 预警后为 **60%**；前置视觉矫正后降至 **52%**。本摘要不包含纯视觉或纯文本单模态实验。

## 数据来源声明

实验数据来源为 MVSA-Single，包含图像、推文文本、图像情感标签和文本情感标签。本仓库只提供前 10 条 JSON 元数据样例，不分发完整图片数据；请从合法来源准备完整数据，并根据本地环境自行配置路径。

## 已知限制

模型权重、完整数据、训练缓存和实验日志不纳入仓库。不同 GPU、Transformers 版本和生成参数可能造成运行时间及输出差异。
