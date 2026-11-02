#!/usr/bin/env python3
"""Evaluate a visual adapter and disagreement route on one frozen split."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def read_rows(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["rows"] if isinstance(data, dict) else data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-conflict", required=True)
    parser.add_argument("--visual-control", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    visual = {
        str(row["id"]): row
        for row in read_rows(args.visual_conflict) + read_rows(args.visual_control)
    }
    baseline = read_rows(args.baseline)
    if set(visual) != {str(row["id"]) for row in baseline}:
        raise RuntimeError("visual and baseline sample IDs differ")
    text_model = joblib.load(args.text_model)
    text_predictions = text_model.predict([row["text"] for row in baseline])

    rows = []
    for row, text_prediction in zip(baseline, text_predictions):
        visual_row = visual[str(row["id"])]
        visual_prediction = visual_row["prediction"]
        text_prediction = str(text_prediction)
        conflict_prediction = int(visual_prediction != text_prediction)
        route = "visual_only" if conflict_prediction else "omni"
        final = visual_prediction if route == "visual_only" else row["prediction_normalized"]["overall_emotion"]
        rows.append({
            "id": row["id"], "is_conflict": row["is_conflict"],
            "visual_emotion": row["visual_emotion"], "text_emotion": row["text_emotion"],
            "visual_prediction": visual_prediction, "text_prediction": text_prediction,
            "detector_prediction": conflict_prediction, "route": route,
            "omni_prediction": row["prediction_normalized"]["overall_emotion"],
            "final_emotion": final,
            "strict_visual_correct": final == row["visual_emotion"],
            "loose_hallucination": final not in {row["visual_emotion"], row["text_emotion"]},
        })

    conflicts = [row for row in rows if row["is_conflict"]]
    controls = [row for row in rows if not row["is_conflict"]]
    y_true = [int(row["is_conflict"]) for row in rows]
    y_pred = [row["detector_prediction"] for row in rows]
    output = {
        "method": "token_supervised_visual_lora_plus_dual_channel_disagreement",
        "evaluation_protocol": "frozen MVSA-Single held-out split; routing does not read labels",
        "n": len(rows), "conflict_n": len(conflicts), "control_n": len(controls),
        "detector": {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        },
        "system": {
            "overall_strict_visual_alignment": sum(row["strict_visual_correct"] for row in rows) / len(rows),
            "conflict_strict_visual_alignment": sum(row["strict_visual_correct"] for row in conflicts) / len(conflicts),
            "control_strict_visual_alignment": sum(row["strict_visual_correct"] for row in controls) / len(controls),
            "conflict_loose_hallucination_rate": sum(row["loose_hallucination"] for row in conflicts) / len(conflicts),
            "visual_routes": sum(row["route"] == "visual_only" for row in rows),
            "omni_routes": sum(row["route"] == "omni" for row in rows),
        },
        "static_visual": {
            "conflict_strict_visual_alignment": sum(
                row["visual_prediction"] == row["visual_emotion"] for row in conflicts
            ) / len(conflicts),
            "control_strict_visual_alignment": sum(
                row["visual_prediction"] == row["visual_emotion"] for row in controls
            ) / len(controls),
        },
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
