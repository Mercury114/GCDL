# Evidence

`metrics_audit.json` is generated from the existing real MVSA-Single result files:

- correction results: 100 conflict samples;
- detector results: 100 conflict + 50 consistent samples;
- no synthetic samples are included in this audit.

The audit intentionally reports both the historical loose metric (52% hallucination rate) and the stricter visual alignment metric (16%). The latter requires the final emotion to equal the ground-truth visual label and is the primary metric for future experiments.

`token_supervised_dual_channel_summary.json` is the current primary frozen-split result. It records the token-only visual-supervision fix, the Omni baseline, the static visual ablation, the routed system, detector precision/recall/F1, and route counts in one place.
