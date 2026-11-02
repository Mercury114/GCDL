#!/usr/bin/env python3
"""Fine-tune Qwen2.5-VL for visual sentiment with answer-token-only loss.

The prompt and all chat-template termination tokens are masked.  The loss is
computed solely on the one-token class answer (positive, neutral, negative),
so a low training loss directly reflects the classification target instead of
easy-to-predict end-of-turn tokens.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments

LABELS = ("positive", "neutral", "negative")
SYSTEM = "You are a visual sentiment classifier. Classify the image only."


def messages(answer=None):
    turns = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Output exactly one label: positive, neutral, or negative."},
        ]},
    ]
    if answer is not None:
        turns.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return turns


class VisualRows(Dataset):
    """Keep only small metadata in memory; image tensors are built in collate."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--single-gpu", action="store_true",
                        help="Place the full base model on CUDA device 0 (after CUDA_VISIBLE_DEVICES).")
    args = parser.parse_args()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if any(row["visual_emotion"] not in LABELS for row in rows):
        raise ValueError("all visual labels must be one of the three fixed classes")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    label_ids = {label: processor.tokenizer(label, add_special_tokens=False).input_ids for label in LABELS}
    if any(len(ids) != 1 for ids in label_ids.values()):
        raise RuntimeError(f"class labels must each be one token: {label_ids}")

    dataset = VisualRows(rows)
    placement = {"": 0} if args.single_gpu else "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map=placement)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

    def collate(features):
        encoded = []
        for row in features:
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
            full = processor.apply_chat_template(messages(row["visual_emotion"]), tokenize=False, add_generation_prompt=False)
            prompt_batch = processor(text=prompt, images=image, return_tensors="pt", truncation=True, max_length=2048)
            full_batch = processor(text=full, images=image, return_tensors="pt", truncation=True, max_length=2048)
            prefix_length = prompt_batch["input_ids"].shape[1]
            answer_ids = label_ids[row["visual_emotion"]]
            suffix = full_batch["input_ids"][0, prefix_length:prefix_length + len(answer_ids)].tolist()
            if suffix != answer_ids:
                raise RuntimeError(f"unexpected answer suffix: expected {answer_ids}, got {suffix}")
            item = {key: value.squeeze(0) for key, value in full_batch.items()}
            labels = torch.full_like(item["input_ids"], -100)
            labels[prefix_length:prefix_length + len(answer_ids)] = item["input_ids"][prefix_length:prefix_length + len(answer_ids)]
            item["labels"] = labels
            encoded.append(item)
        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            values = [item[key] for item in encoded]
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                values, batch_first=True,
                padding_value=-100 if key == "labels" else processor.tokenizer.pad_token_id)
        batch["pixel_values"] = torch.cat([item["pixel_values"] for item in encoded])
        batch["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in encoded]).reshape(-1, 3)
        return batch

    train_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, gradient_checkpointing=True,
        remove_unused_columns=False, save_strategy="epoch", logging_steps=10,
        report_to="none")
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=collate)
    result = trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    Path(args.output, "training_manifest.json").write_text(json.dumps({
        "n": len(rows), "epochs": args.epochs, "learning_rate": args.learning_rate,
        "eval_disjoint": True, "task": "visual_emotion_only",
        "supervision": "assistant_class_token_only",
        "single_gpu": args.single_gpu,
        "class_tokens": label_ids,
        "train_metrics": result.metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
