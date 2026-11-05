#!/usr/bin/env python3
"""Pre-submission integrity and rubric checklist; exits non-zero on missing artifacts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "AI_USAGE_LOG.md", "requirements.txt", ".gitignore",
    "scripts/baseline_test.py", "scripts/finetune_conflict_detector.py",
    "scripts/evaluate_with_correction.py", "scripts/audit_results.py",
    "scripts/visual_constrained_eval.py", "scripts/detector_constrained_eval.py",
    "scripts/evaluate_stratified_system.py", "scripts/build_stratified_eval.py",
    "scripts/train_text_emotion.py", "scripts/evaluate_dual_channel.py",
    "scripts/finetune_visual_emotion_token_supervised.py",
    "scripts/evaluate_token_supervised_system.py",
    "demo/run_demo.py", "demo/streamlit_app.py",
    "data/sample_data/conflict_samples_10.json",
    "evidence/metrics_audit.json", "evidence/stratified_results_summary.json",
    "evidence/dual_channel_summary.json",
    "evidence/token_supervised_dual_channel_summary.json",
    "evidence/tuning_decision.json",
    "evidence/heldout_end_to_end_summary.json", "docs/RELATED_WORK.md",
    "docs/SYSTEM_DESIGN.md", "docs/COURSE_ALIGNMENT.md",
    "docs/DEMO_RUNBOOK.md",
    "docs/REPORT_STORY.md",
    "docs/TECHNICAL_DOCUMENTATION_FOR_DEFENSE.md",
]


def main():
    missing = [x for x in REQUIRED if not (ROOT / x).exists()]
    samples = json.loads((ROOT / "data/sample_data/conflict_samples_10.json").read_text())
    evidence = json.loads((ROOT / "evidence/metrics_audit.json").read_text())
    forbidden = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".log"}:
            forbidden.append(str(path.relative_to(ROOT)))
    checks = {
        "required_artifacts": not missing,
        "sample_count_is_10": len(samples) == 10,
        "sample_fields": all({"image_path", "text", "visual_emotion", "text_emotion", "is_conflict"} <= set(x) for x in samples),
        "real_data_audit_present": evidence.get("data_provenance", "").startswith("MVSA-Single real"),
        "no_large_model_or_log_files": not forbidden,
        "strict_metric_present": "strict_visual_alignment" in evidence,
        "ai_log_present": (ROOT / "AI_USAGE_LOG.md").stat().st_size > 500,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if missing:
        print("missing:", ", ".join(missing))
    if forbidden:
        print("forbidden:", ", ".join(forbidden))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
