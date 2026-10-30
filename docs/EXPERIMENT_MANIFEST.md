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

Held-out detector: TP=13, FP=5, FN=89, TN=46, F1=21.67%. The low detector recall is a real limitation and explains why the end-to-end route is close to the Omni baseline.

## Interpretation rule

The balanced visual branch is a component ablation. The end-to-end value is the detector-routed result. Never replace the latter with the former in a table or presentation.
