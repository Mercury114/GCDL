#!/usr/bin/env python3
"""LoRA fine-tune a Qwen2.5-VL visual-only emotion classifier."""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                          Trainer, TrainingArguments)

SYSTEM = "你是一个视觉情绪分类器，只根据图像判断情绪。"


def messages(answer=None):
    rows = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "请只输出 positive、neutral 或 negative。"}]}]
    if answer is not None:
        rows.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    ds = Dataset.from_list(rows)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    def encode(row):
        image = Image.open(row["image_path"]).convert("RGB")
        prompt = processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(messages(row["visual_emotion"]), tokenize=False, add_generation_prompt=False)
        # Preserve image placeholder tokens; truncating to 512 can remove part
        # of the visual sequence and makes the processor reject the example.
        q = processor(text=prompt, images=image, return_tensors="pt", truncation=True, max_length=2048)
        z = processor(text=full, images=image, return_tensors="pt", truncation=True, max_length=2048)
        output = {k: v.squeeze(0) for k, v in z.items()}
        labels = output["input_ids"].clone()
        labels[:q["input_ids"].shape[1]] = -100
        output["labels"] = labels
        return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in output.items()}

    ds = ds.map(encode, remove_columns=ds.column_names)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="auto", max_memory={0: "45GiB", "cpu": "120GiB"})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

    def collate(features):
        output = {}
        for key in ("input_ids", "attention_mask", "labels"):
            values = [torch.tensor(x[key]) for x in features]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                values, batch_first=True,
                padding_value=-100 if key == "labels" else processor.tokenizer.pad_token_id)
        output["pixel_values"] = torch.cat([torch.tensor(x["pixel_values"]) for x in features])
        output["image_grid_thw"] = torch.cat([torch.tensor(x["image_grid_thw"]) for x in features]).reshape(-1, 3)
        return output

    train_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        learning_rate=2e-5, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, gradient_checkpointing=True,
        remove_unused_columns=False, save_strategy="epoch", logging_steps=10,
        report_to="none")
    trainer = Trainer(model=model, args=train_args, train_dataset=ds, data_collator=collate)
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    Path(args.output, "training_manifest.json").write_text(json.dumps({
        "n": len(rows), "epochs": args.epochs, "learning_rate": 2e-5,
        "eval_disjoint": True, "task": "visual_emotion_only"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
