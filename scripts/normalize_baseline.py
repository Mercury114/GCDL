#!/usr/bin/env python3
"""Normalize multilingual emotion aliases before scoring; preserve raw output."""
import argparse
import json
from pathlib import Path

ALIASES = {
    "中性": "neutral", "平静": "neutral", "放松": "neutral", "专注": "neutral",
    "高兴": "positive", "开心": "positive", "快乐": "positive", "积极": "positive",
    "正面": "positive", "兴奋": "positive", "励志": "positive", "勇敢": "positive",
    "担忧": "negative", "悲伤": "negative", "难过": "negative", "愤怒": "negative",
    "生气": "negative", "消极": "negative", "负面": "negative", "恐惧": "negative",
    "害怕": "negative", "惊讶": "negative", "孤独": "negative", "严肃": "negative",
    "受伤": "negative",
}


def normalize(value):
    text = str(value).strip().lower()
    for alias, label in ALIASES.items():
        if alias in text:
            return label
    for label in ("positive", "neutral", "negative"):
        if label in text:
            return label
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    for row in rows:
        raw = row.get("prediction", {})
        normalized = {k: normalize(v) for k, v in raw.items()}
        row["prediction_normalized"] = normalized
        row["strict_visual_correct"] = normalized.get("overall_emotion") == row["visual_emotion"]
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n={len(rows)} strict_visual_correct={sum(x['strict_visual_correct'] for x in rows)}")


if __name__ == "__main__":
    main()
