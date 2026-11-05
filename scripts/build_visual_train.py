#!/usr/bin/env python3
"""Create a visual-emotion training split disjoint from the stratified eval IDs."""
import argparse
import json
from pathlib import Path

LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--eval", required=True, nargs="+",
        help="One or more frozen evaluation JSON files whose IDs must be excluded.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    eval_ids = {
        row["id"]
        for eval_path in args.eval
        for row in json.loads(Path(eval_path).read_text(encoding="utf-8"))
    }
    meta = json.loads((root / "test.json").read_text(encoding="utf-8"))
    rows = []
    for sid, value in meta.items():
        image = root / "data" / f"{sid}.jpg"
        if sid in eval_ids or not image.exists():
            continue
        rows.append({"id": sid, "image_path": str(image.resolve()),
                     "visual_emotion": LABEL[value["img_label"]],
                     "source": "MVSA-Single", "synthetic": False})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print(json.dumps({"n": len(rows), "excluded": len(eval_ids),
                      "distribution": Counter(x["visual_emotion"] for x in rows)},
                     ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
