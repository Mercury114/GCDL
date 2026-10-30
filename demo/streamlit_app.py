#!/usr/bin/env python3
"""Interactive demo; mock mode is deterministic and needs no model weights."""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emotion_conflict.routing import choose_route

st.set_page_config(page_title="Emotion Conflict Guard", page_icon="🛡️", layout="wide")
st.title("Emotion Conflict Guard")
st.caption("Pre-generation conflict detection and dynamic collaborative reasoning")

with st.sidebar:
    st.header("Runtime")
    mock = st.toggle("Demo mode (no model weights)", value=True)
    high = st.slider("High-confidence threshold", 0.50, 0.99, 0.75, 0.01)
    low = st.slider("Review threshold", 0.30, high, 0.55, 0.01)

left, right = st.columns([1, 1])
with left:
    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    text = st.text_area("Text", "今天真是太开心了！", height=140)
    run = st.button("Run guarded inference", type="primary", use_container_width=True)
    if image:
        st.image(image, use_container_width=True)

if run:
    if not image:
        st.error("Please upload an image first.")
    elif mock:
        # This is a UI smoke-test route, not an experimental metric.
        score = 0.86 if any(word in text for word in ("开心", "开心", "happy", "great")) else 0.62
        decision = choose_route(1, score, high=high, low=low)
        result = "视觉-only 结果（演示占位）"
        detector_raw = "冲突（demo heuristic）"
        st.info("Demo mode uses a deterministic placeholder detector. It must not be used as an experiment result.")
    else:
        st.error("Real inference is intentionally launched from demo/run_demo.py so model paths remain explicit and auditable.")
        st.stop()
    with right:
        st.subheader("Decision trace")
        st.metric("Detector", "Conflict" if decision.label else "Consistent")
        st.metric("Confidence", "unknown" if decision.confidence is None else f"{decision.confidence:.2f}")
        st.write({"route": decision.route, "review_required": decision.review_required})
        st.success(result)
        st.code(json.dumps({
            "detector_raw": detector_raw,
            "route": decision.route,
            "explanation": decision.explanation,
            "experiment_safe": False,
        }, ensure_ascii=False, indent=2), language="json")
