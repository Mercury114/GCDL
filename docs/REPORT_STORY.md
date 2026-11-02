# Report-ready story and evidence map

This document is a report-writing aid. It does not replace the team's own problem definition, ethics analysis, or source attribution.

## One-sentence contribution

We build an auditable pre-generation conflict guard for vision-text emotion inputs: two independent modality predictors expose disagreement before generation, disagreement routes the sample to a visual-only branch, and agreement preserves the multimodal Omni route.

## The engineering story

### 1. The problem is measurable

MVSA-Single provides real image labels and text labels. We select samples whose labels disagree, then hold out a balanced evaluation set rather than relying only on the original imbalanced 100-sample experiment. The final held-out evaluation contains 102 conflict samples, with 34 negative, 34 neutral, and 34 positive visual labels, plus 51 consistent controls. All 153 evaluation IDs are excluded from detector and visual-adapter training.

### 2. The first fix was not enough

The historical generation-based pipeline produced an attractive-looking loose hallucination result: Omni 58%, prompt warning 60%, and visual correction 52%. The audit showed why this was insufficient: on a three-class conflict sample, counting either the visual label or the text label as correct can hide a text-dominant answer. The old correction output had only 16% strict visual alignment. The prompt warning is therefore retained as a useful negative result, not presented as a successful solution.

### 3. The revised system makes the route auditable

The current primary system predicts visual polarity with a token-supervised visual LoRA branch and text polarity with an independent TF-IDF classifier. It declares a conflict only when the two predictions disagree. The visual LoRA masks prompt and chat-termination tokens, so its loss is computed only on the single class-answer token. The route decision does not inspect ground-truth labels. A conflict goes to visual-only constrained prediction; agreement stays on the Omni multimodal route. This separates detection, routing, and final prediction, making each stage testable.

### 4. The held-out result is a real but bounded improvement

| System | Conflict strict visual alignment | Control strict visual alignment |
|---|---:|---:|
| Qwen2.5-Omni baseline | 33.33% (34/102) | 19.61% (10/51) |
| Token-supervised dual-channel dynamic routing | **42.16% (43/102)** | **29.41% (15/51)** |
| Absolute change | **+8.82 pp** | **+9.80 pp** |

The conflict gain is 26.47% relative to the Omni baseline alignment, while the absolute gain is 8.82 percentage points. On the same conflict split, the auxiliary loose hallucination rate decreases from 42.16% (43/102) to 31.37% (32/102). The system detector reaches F1=66.67%, conflict precision=63.16%, and conflict recall=70.59% on the same 153-sample held-out set. These numbers support a meaningful engineering improvement, not a claim of complete hallucination removal.

### 5. The limitation is part of the result

The system routes 114/153 samples through the visual branch and 39/153 through Omni. The static token-supervised visual branch has higher overall strict alignment (63/153) than dynamic routing (58/153), so the Omni-preserving route is an explicit behavior trade-off rather than a universal accuracy win. The detector is not perfect, and the visual branch is not a human-verified emotion judge. MVSA labels are sentiment-polarity annotations, not ground-truth psychological states. The correct conclusion is therefore “pre-generation routing improves visual-fact alignment under this evaluation protocol,” not “the system solves emotion hallucination in general.”

## Recommended defense sequence

1. Show one conflict input and the independent visual/text predictions.
2. Show disagreement, the selected `visual_only` route, and the final constrained label.
3. Show one consistent input and the `omni` route.
4. Show the 58% -> 60% historical negative result and explain why a textual warning did not solve modality dominance.
5. Show the held-out table above, then disclose the dataset, label, and model limitations.

## Metric rules

- Primary metric: strict visual alignment, `final_emotion == visual_emotion`.
- Historical auxiliary metric: loose hallucination, where an output is treated as non-hallucinated if it matches either the visual or text label. It is retained only for reproducing the earlier report and must not replace the strict metric.
- Never report a visual-branch ablation as an end-to-end result. The current primary end-to-end result is 42.16% conflict strict visual alignment in `evidence/token_supervised_dual_channel_summary.json`; earlier system versions remain in their own evidence files.

## Evidence map

| Claim | File |
|---|---|
| Real-data provenance and historical metric audit | `evidence/metrics_audit.json` |
| Balanced, evaluation-disjoint split | `evidence/stratified_results_summary.json` |
| Current detector and end-to-end result | `evidence/token_supervised_dual_channel_summary.json` |
| Exact split and anti-confusion rule | `docs/EXPERIMENT_MANIFEST.md` |
| Demo procedure | `docs/DEMO_RUNBOOK.md` |
