#!/usr/bin/env python3
"""Produce strict metrics and an auditable report from existing JSON results."""
import argparse
import json
from collections import Counter
from pathlib import Path

from emotion_conflict.routing import choose_route


def rate(items, key):
    return sum(bool(x.get(key)) for x in items) / len(items) if items else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correction", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    correction = json.loads(Path(args.correction).read_text(encoding="utf-8"))
    detector = json.loads(Path(args.detector).read_text(encoding="utf-8"))
    strict_visual = [x["final_emotion"] != x["visual_emotion"] for x in correction]
    detector_labels = [int(x["detector_prediction"]) for x in detector]
    truth = [int(bool(x["is_conflict"])) for x in detector]
    tp = sum(p == 1 and y == 1 for p, y in zip(detector_labels, truth))
    fp = sum(p == 1 and y == 0 for p, y in zip(detector_labels, truth))
    fn = sum(p == 0 and y == 1 for p, y in zip(detector_labels, truth))
    tn = sum(p == 0 and y == 0 for p, y in zip(detector_labels, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "data_provenance": "MVSA-Single real labels; no synthetic samples",
        "correction_n": len(correction),
        "strict_visual_alignment": 1 - sum(strict_visual) / len(strict_visual),
        "legacy_loose_hallucination_rate": rate(correction, "hallucination"),
        "detector": {"n": len(detector), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "accuracy": (tp + tn) / len(detector), "precision": precision,
                     "recall": recall, "f1": f1},
        "route_counts": dict(Counter(
            choose_route(int(x["detector_prediction"]), x.get("confidence")).route
            for x in detector)),
        "interpretation": "strict_visual_alignment is the primary metric for text-induced visual drift; legacy loose hallucination is retained only for comparison.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
