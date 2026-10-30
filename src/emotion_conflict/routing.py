"""Deterministic routing policy kept separate from model inference."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RouteDecision:
    label: int
    confidence: Optional[float]
    route: str
    review_required: bool
    explanation: str


def choose_route(label: int, confidence: Optional[float], high: float = 0.75,
                 low: float = 0.55) -> RouteDecision:
    """Choose a route without silently treating missing confidence as certainty."""
    if confidence is None:
        if label == 1:
            return RouteDecision(label, None, "human_review", True,
                                 "检测器未提供置信度，拒绝无依据地执行强制视觉裁决。")
        return RouteDecision(label, None, "omni", False,
                             "检测器未提供置信度，按一致分支保留多模态推理。")
    if label == 1 and confidence >= high:
        return RouteDecision(label, confidence, "visual_only", False,
                             f"高置信度冲突（{confidence:.2f}），执行视觉-only 裁决。")
    if label == 1 and confidence >= low:
        return RouteDecision(label, confidence, "human_review", True,
                             f"中等置信度冲突（{confidence:.2f}），保留结果并请求人工复核。")
    return RouteDecision(label, confidence, "omni", False,
                         "未达到冲突门控阈值，保留 Omni 图文协同推理。")


def choose_dual_channel_route(conflict: bool) -> RouteDecision:
    """Route the evaluated dual-channel disagreement policy.

    The disagreement rule is the detector itself; it does not invent a
    probability. A conflict therefore goes to visual correction, while an
    agreement keeps Omni synthesis.
    """
    if conflict:
        return RouteDecision(1, None, "visual_only", False,
                             "视觉与文本独立情绪预测不一致，执行视觉-only 裁决。")
    return RouteDecision(0, None, "omni", False,
                         "视觉与文本独立情绪预测一致，保留 Omni 图文协同推理。")
