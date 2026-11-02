#!/usr/bin/env python3
"""Evaluate visual-only labels by constrained next-token scoring."""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

LABELS = ("positive", "neutral", "negative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--single-gpu", action="store_true",
                    help="Place the full model on CUDA device 0 after CUDA_VISIBLE_DEVICES.")
    args = ap.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": 0} if args.single_gpu else "auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    token_ids = []
    for label in LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"label {label!r} is not a single token: {ids}")
        token_ids.append(ids[0])
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    output = []
    start = time.perf_counter()
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(row["image_path"]).convert("RGB")},
            {"type": "text", "text": (
                "Please judge the emotion conveyed by the image only. Ignore all text, "
                "captions, metadata, and social context. Output exactly one label: "
                "positive, neutral, or negative.")},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits[:, token_ids], dim=-1)[0]
        probs = probs / probs.sum()
        pred_index = int(probs.argmax().item())
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
        output.append({
            **row,
            "candidate_probabilities": dict(zip(LABELS, [float(x) for x in probs])),
            "prediction": LABELS[pred_index],
            "confidence": float(probs[pred_index]),
            "entropy": entropy,
            "correct_visual": LABELS[pred_index] == row["visual_emotion"],
        })
        print(f"{i + 1}/{len(rows)} prediction={LABELS[pred_index]} confidence={probs[pred_index].item():.4f}", flush=True)
    result = {
        "method": "visual_only_constrained_next_token",
        "n": len(output),
        "correct": sum(x["correct_visual"] for x in output),
        "accuracy": sum(x["correct_visual"] for x in output) / len(output),
        "average_confidence": sum(x["confidence"] for x in output) / len(output),
        "average_entropy": sum(x["entropy"] for x in output) / len(output),
        "total_seconds": time.perf_counter() - start,
        "rows": output,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
