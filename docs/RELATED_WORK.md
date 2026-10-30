# Related work and positioning

本项目不声称“首次研究图文冲突”或“首次缓解情绪幻觉”。截至 2026-10-30 的检索，至少存在以下相邻方向：

| 工作 | 关系 | 本项目的差异 |
|---|---|---|
| EmotionHallucer / PEP-MEK | 直接研究多模态情绪幻觉，并采用预测-解释-再预测 | 本项目关注 MVSA 图文冲突的工程路由，并额外审计严格视觉对齐、假阳性和人工复核 |
| CrossCheck-Bench | 评测跨模态冲突解析能力 | 本项目不是通用 benchmark，而是面向情绪标签的可运行推理系统 |
| Words or Vision? | 研究图文冲突时模型是否盲信文本 | 本项目把该诊断转化为 LoRA 检测器 + 动态路由原型 |
| Cross-modal knowledge conflict | 研究视觉与文本答案不一致 | 本项目使用真实 MVSA 情感标签并提供课程作业级端到端代码和 Demo |

因此，报告中应使用“工程整合与实证扩展”而非“无人做过的原创算法”。真正需要证明的不是新模型名字，而是路由、指标和残余风险是否被实测和审计。

参考入口：

- EmotionHallucer: https://arxiv.org/abs/2505.11405
- CrossCheck-Bench: https://doi.org/10.1609/aaai.v40i31.39788
- Words or Vision?: https://openaccess.thecvf.com/content/CVPR2025/papers/Deng_Words_or_Vision_Do_Vision-Language_Models_Have_Blind_Faith_in_CVPR_2025_paper.pdf
