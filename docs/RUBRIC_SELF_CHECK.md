# Submission self-check

This is an engineering self-check, not a promised grade. The instructor's judgment remains authoritative.

| Course dimension | Current evidence | Remaining human responsibility |
|---|---|---|
| Problem definition | Real MVSA sample metadata, strict audit, explicit scope | Write the signed <=80-character problem statement and first-hand evidence in the team's own words |
| Value and ethics | `ETHICS_WORKSHEET.md`, human-review route, no-diagnosis boundary | Complete stakeholder harms, who bears residual risk, and the value ordering |
| Cross-domain paradigm | `RELATED_WORK.md`, pre-generation isolation versus prompt warning | Team must defend the challenged assumption orally and write the final Chapter 4 judgment |
| Runnable implementation | Four experiment scripts, routing package, CLI and Streamlit Demo | Run on the target machine and record environment/GPU information |
| Technical trade-off | 58% -> 60% negative warning result; strict metric exposes 16% alignment | Explain why a lower legacy rate is not automatically a trustworthy improvement |
| AI usage compliance | `AI_USAGE_LOG.md` records prompts, drafts, changes and reasons | Add team member names and verify the log matches the actual authorship |
| Reproducibility | `requirements.txt`, `pyproject.toml`, Makefile, sample metadata | Provide local model/data paths and attach a short run transcript |

## Suggested answer structure for the defense

1. What is observable? A conflict detector can route inputs, but its 62.67% accuracy and missing calibration limit forced correction.
2. What failed? A warning changed the historical loose rate from 58% to 60%.
3. What did the audit discover? The old loose metric accepted either visual or text labels; strict visual alignment was only 16% on the existing correction output.
4. What is the engineering response? Add confidence-aware routing and human review instead of unconditional visual override.
5. What remains unresolved? MVSA sentiment labels are not ground truth for private psychological state, and the detector needs calibrated confidence and a held-out control evaluation.
