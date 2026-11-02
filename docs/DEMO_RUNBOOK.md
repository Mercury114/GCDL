# Demo runbook

## Primary dual-channel demo

Prepare local paths for Qwen2.5-VL-7B, the token-supervised visual LoRA adapter, and the TF-IDF text classifier:

```bash
python demo/run_demo.py \
  --image /path/to/image.jpg \
  --text "这真是令人开心的一天。" \
  --vl-model /path/to/Qwen2.5-VL-7B \
  --omni-model /path/to/Qwen2.5-Omni-7B \
  --visual-adapter /path/to/visual_lora_token_supervised \
  --text-model /path/to/text_emotion_tfidf.joblib \
  --single-gpu
```

The primary route prints independent visual/text predictions, the disagreement rule, the selected route, and the final result. It does not load test labels.

## Legacy comparison mode

```bash
python demo/run_demo.py --mode legacy \
  --adapter /path/to/conflict_detector_lora_v2 \
  --image /path/to/image.jpg --text "这真是令人开心的一天。" \
  --vl-model /path/to/Qwen2.5-VL-7B --omni-model /path/to/Qwen2.5-Omni-7B
```

Legacy mode is retained only to demonstrate the earlier generation-based detector.

## Defense evidence

Show `evidence/token_supervised_dual_channel_summary.json` and `docs/EXPERIMENT_MANIFEST.md` for the held-out split. The current end-to-end dual-channel result is 42.16% versus 33.33% Omni baseline. State the detector precision/recall and static-branch ablation when answering questions about trade-offs.
