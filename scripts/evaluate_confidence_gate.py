#!/usr/bin/env python3
"""Fuse visual-only constrained predictions with Omni using a fixed confidence gate."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", required=True, help="visual_constrained_eval.json")
    ap.add_argument("--baseline", required=True, help="normalized Omni baseline results")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    visual = json.loads(Path(args.visual).read_text(encoding="utf-8"))["rows"]
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    rows = []
    for v, b in zip(visual, baseline):
        use_visual = float(v["confidence"]) >= args.threshold
        final = v["prediction"] if use_visual else b["prediction_normalized"]["overall_emotion"]
        rows.append({
            "id": v["id"],
            "visual_emotion": v["visual_emotion"],
            "text_emotion": v["text_emotion"],
            "visual_prediction": v["prediction"],
            "visual_confidence": v["confidence"],
            "baseline_prediction": b["prediction_normalized"]["overall_emotion"],
            "route": "visual_only" if use_visual else "omni",
            "final_emotion": final,
            "strict_visual_correct": final == v["visual_emotion"],
        })
    result = {
        "method": "fixed_confidence_gate",
        "threshold": args.threshold,
        "n": len(rows),
        "visual_routes": sum(x["route"] == "visual_only" for x in rows),
        "omni_routes": sum(x["route"] == "omni" for x in rows),
        "strict_visual_correct": sum(x["strict_visual_correct"] for x in rows),
        "strict_visual_alignment": sum(x["strict_visual_correct"] for x in rows) / len(rows),
        "rows": rows,
        "note": "Threshold fixed before reading ground-truth labels; no label is used for route selection.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
