#!/usr/bin/env python3
"""Small local demo for detector-gated emotion inference."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
import joblib
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_conflict.routing import choose_dual_channel_route, choose_route

LABELS = ("positive", "neutral", "negative")


def generate(processor, model, content, max_new_tokens=32):
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["thinker_pixel_values"] = inputs.pop("pixel_values")
    if "image_grid_thw" in inputs:
        inputs["thinker_image_grid_thw"] = inputs.pop("image_grid_thw")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def visual_scores(processor, model, image_path):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": (
            "Please judge the emotion conveyed by the image only. Ignore all text, "
            "captions, metadata, and social context. Output exactly one label: "
            "positive, neutral, or negative.")},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    token_ids = [processor.tokenizer(x, add_special_tokens=False).input_ids[0] for x in LABELS]
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, token_ids]
        probs = torch.softmax(logits, dim=-1)[0]
    probs = probs / probs.sum()
    index = int(probs.argmax().item())
    return LABELS[index], float(probs[index]), dict(zip(LABELS, [float(x) for x in probs]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--vl-model", required=True)
    parser.add_argument("--omni-model", required=True)
    parser.add_argument("--adapter", default=None,
                        help="Legacy detector adapter; required only with --mode legacy")
    parser.add_argument("--text-model", default=None,
                        help="TF-IDF text-emotion model for the dual-channel route")
    parser.add_argument("--mode", choices=("dual", "legacy"), default="dual")
    parser.add_argument("--visual-adapter", default=None,
                        help="Optional visual-emotion LoRA; defaults to detector adapter")
    parser.add_argument("--detector-confidence", type=float, default=None,
                        help="Calibrated detector confidence; omit to require review on conflict")
    args = parser.parse_args()
    image_path = str(Path(args.image).resolve())

    processor = AutoProcessor.from_pretrained(args.vl_model, local_files_only=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_model, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto")
    detector = None
    detector_raw = ""
    visual_model = base
    visual_label = None
    if args.mode == "dual":
        if not args.text_model or not args.visual_adapter:
            parser.error("dual mode requires --text-model and --visual-adapter")
        visual_model = PeftModel.from_pretrained(base, args.visual_adapter)
        visual_model.eval()
        visual_label, visual_confidence, visual_probs = visual_scores(processor, visual_model, image_path)
        text_model = joblib.load(args.text_model)
        text_label = str(text_model.predict([args.text])[0])
        is_conflict = visual_label != text_label
        decision = choose_dual_channel_route(is_conflict)
        detector_raw = json.dumps({
            "visual_prediction": visual_label,
            "visual_confidence": visual_confidence,
            "visual_probabilities": visual_probs,
            "text_prediction": text_label,
            "rule": "visual_prediction != text_prediction",
        }, ensure_ascii=False)
    else:
        if not args.adapter:
            parser.error("legacy mode requires --adapter")
        detector = PeftModel.from_pretrained(base, args.adapter)
        detector.eval()
        prompt = ("你是一个情绪冲突检测器。请判断输入图像的视觉情绪与文本表达的情绪是否冲突。"
                  f"\n文本：{args.text}\n请只回答‘冲突’或‘一致’。")
        detector_raw = generate(processor, detector, [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": prompt}], max_new_tokens=8)
        is_conflict = "冲突" in detector_raw or "conflict" in detector_raw.lower()
        decision = choose_route(int(is_conflict), args.detector_confidence)

    if decision.route == "visual_only":
        if args.mode == "dual":
            result = f"{visual_label} (视觉-only 受限标签)"
        else:
            result = generate(processor, visual_model, [
                    {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    {"type": "text", "text": "请仅根据这张图片判断人物的情绪，输出一个词。"}])
        route = "visual-only correction (Qwen2.5-VL)"
    elif decision.route == "omni":
        omni_processor = AutoProcessor.from_pretrained(args.omni_model, local_files_only=True)
        omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.omni_model, local_files_only=True, torch_dtype="auto", device_map="auto")
        result = generate(omni_processor, omni, [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"请综合判断图像和文本的整体情绪。文本：{args.text}"}])
        route = "multimodal synthesis (Qwen2.5-Omni)"
    else:
        result = "等待人工复核：检测到冲突但没有足够置信度执行强制视觉裁决。"
        route = "human review"
    print(f"detector_raw: {detector_raw}")
    print(f"detector_status: {'conflict' if is_conflict else 'consistent'}")
    print(f"mode: {args.mode}")
    print(f"route_decision: {decision.route}")
    print(f"review_required: {decision.review_required}")
    print(f"route_explanation: {decision.explanation}")
    print(f"route: {route}")
    print(f"final_result: {result}")


if __name__ == "__main__":
    main()
