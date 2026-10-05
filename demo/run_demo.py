#!/usr/bin/env python3
"""Small local demo for detector-gated emotion inference."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5_VLForConditionalGeneration


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--vl-model", required=True)
    parser.add_argument("--omni-model", required=True)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()
    image_path = str(Path(args.image).resolve())

    processor = AutoProcessor.from_pretrained(args.vl_model, local_files_only=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_model, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto")
    detector = PeftModel.from_pretrained(base, args.adapter)
    detector.eval()
    prompt = ("你是一个情绪冲突检测器。请判断输入图像的视觉情绪与文本表达的情绪是否冲突。"
              f"\n文本：{args.text}\n请只回答‘冲突’或‘一致’。")
    detector_raw = generate(processor, detector, [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": prompt}], max_new_tokens=8)
    is_conflict = "冲突" in detector_raw or "conflict" in detector_raw.lower()

    if is_conflict:
        with detector.disable_adapter():
            result = generate(processor, detector, [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": "请仅根据这张图片判断人物的情绪，输出一个词。"}])
        route = "visual-only correction (Qwen2.5-VL)"
    else:
        omni_processor = AutoProcessor.from_pretrained(args.omni_model, local_files_only=True)
        omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.omni_model, local_files_only=True, torch_dtype="auto", device_map="auto")
        result = generate(omni_processor, omni, [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"请综合判断图像和文本的整体情绪。文本：{args.text}"}])
        route = "multimodal synthesis (Qwen2.5-Omni)"
    print(f"detector_raw: {detector_raw}")
    print(f"detector_status: {'conflict' if is_conflict else 'consistent'}")
    print(f"route: {route}")
    print(f"final_result: {result}")


if __name__ == "__main__":
    main()
