# System design

## Input and evidence

The system accepts one image and one text. The current primary detector has two independent channels: a visual LoRA predicts one of three visual emotion polarities, while a TF-IDF text classifier predicts one of three text polarities. A disagreement is the conflict label. The legacy generative detector remains available for comparison only. Every request should preserve the original text, image identifier, channel predictions, route, model output and metric version in an audit record.

## Dynamic route

1. `omni`: independent visual and text predictions agree; run Qwen2.5-Omni on image + text.
2. `visual_only`: independent predictions disagree; use the constrained visual label as the result and remove text from the decision.
3. `human_review`: optional policy route for deployments that require calibrated confidence; the current offline dual-channel experiment reports the direct disagreement route separately.

The thresholds are policy parameters, not learned facts. They must be tuned on a validation split and reported with the trade-off between conflict recall, false positives and consistent-sample utility.

## Metrics

- Conflict detector: accuracy, precision, recall, F1, confusion matrix.
- Strict visual alignment: final label equals `visual_emotion`.
- Legacy loose rate: final label is neither visual nor text label; retained solely to reproduce the original report.
- Routing cost: route counts, latency, review fraction and visual-only invocation fraction.

The strict metric is primary because a conflict sample has two valid labels in a three-class space; accepting either label can hide text-following errors.
