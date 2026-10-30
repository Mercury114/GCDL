#!/usr/bin/env python3
"""Build a balanced, label-stratified MVSA evaluation set without synthetic data."""
import argparse
import json
from collections import Counter
from pathlib import Path

LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per_visual_label", type=int, default=34)
    ap.add_argument("--controls_per_label", type=int, default=17)
    args = ap.parse_args()
    root = Path(args.root)
    meta = json.loads((root / "test.json").read_text(encoding="utf-8"))
    rows = []
    for sid, value in meta.items():
        image = root / "data" / f"{sid}.jpg"
        text = root / "data" / f"{sid}.txt"
        if not image.exists() or not text.exists():
            continue
        visual = LABEL[value["img_label"]]
        textual = LABEL[value["text_label"]]
        rows.append({
            "id": sid, "image_path": str(image.resolve()),
            "text": text.read_text(errors="replace").strip(),
            "visual_emotion": visual, "text_emotion": textual,
            "is_conflict": visual != textual, "source": "MVSA-Single",
            "synthetic": False,
        })
    conflicts = []
    controls = []
    for label in ("negative", "neutral", "positive"):
        conflict_rows = [x for x in rows if x["is_conflict"] and x["visual_emotion"] == label]
        control_rows = [x for x in rows if not x["is_conflict"] and x["visual_emotion"] == label]
        conflicts.extend(conflict_rows[:args.per_visual_label])
        controls.extend(control_rows[:args.controls_per_label])
    if len(conflicts) != args.per_visual_label * 3:
        raise RuntimeError("not enough conflict samples for all three visual labels")
    if len(controls) != args.controls_per_label * 3:
        raise RuntimeError("not enough controls for all three visual labels")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "stratified_conflict_eval.json").write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "stratified_control_eval.json").write_text(json.dumps(controls, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"conflicts": len(conflicts), "controls": len(controls),
                      "conflict_visual_distribution": dict(Counter(x["visual_emotion"] for x in conflicts)),
                      "control_visual_distribution": dict(Counter(x["visual_emotion"] for x in controls))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
