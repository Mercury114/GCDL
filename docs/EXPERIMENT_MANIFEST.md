# Reproducible experiment manifest

All reported values below come from the same held-out MVSA split:

- conflict evaluation: 102 samples, visual labels 34/34/34;
- control evaluation: 51 samples, visual labels 17/17/17;
- detector and visual LoRA training exclude all 153 evaluation IDs.

## Reported measurements

| System | Conflict strict visual alignment | Control strict visual alignment |
|---|---:|---:|
| Qwen2.5-Omni baseline | 33.33% | 19.61% |
| Base Qwen2.5-VL constrained visual | 32.35% | not used as final route |
| Balanced visual LoRA branch | 39.22% | 37.25% |
| Held-out detector + visual LoRA + Omni | 34.31% | 19.61% |
| Token-supervised visual LoRA (static ablation) | 41.18% | 41.18% |
| Token-supervised dual-channel dynamic system | **42.16%** | **29.41%** |

Held-out detector: TP=13, FP=5, FN=89, TN=46, F1=21.67%. The low detector recall is a real limitation and explains why the end-to-end route is close to the Omni baseline.

## Current primary system

The token-supervised visual LoRA masks prompt, end-of-turn, and newline tokens, and computes loss only on the one-token class answer. Its dual-channel disagreement route produces 43/102 conflict strict visual matches, versus 34/102 for Omni: +8.82 percentage points. On the same conflict split, loose hallucination decreases from 43/102 (42.16%) for Omni to 32/102 (31.37%) for the dynamic system. On controls, strict alignment is 15/51 (29.41%) versus 10/51 (19.61%) for Omni.

The current detector has TP=72, FP=42, FN=30, TN=9, precision=63.16%, recall=70.59%, and F1=66.67%. Because the held-out split contains two-thirds conflict samples, this detector F1 must not be interpreted without its precision/recall and route counts. The static visual branch is a stronger alignment ablation (63/153 overall) than dynamic routing (58/153 overall); dynamic routing is retained because it preserves an Omni route for 39/153 inputs, so this is an explicit system-behavior trade-off rather than a claim that routing dominates every static policy.

## Interpretation rule

The balanced and token-supervised visual branches are component ablations. The end-to-end values are detector-routed results. Never replace a routed-system number with a visual-branch number in a table or presentation.
