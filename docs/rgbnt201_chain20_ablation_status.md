# RGBNT201 Chain20 Ablation Status

Date: 2026-06-02

This note records the current implementation decision for the RGBNT201 chain20
ablation overlays. These runs are short diagnostic checks with the 120-epoch
scheduler preserved but training stopped at epoch 20. They should not be
presented as final published numbers without a full validation plan.

## Current Decision

Use A2 as the evidence-backed chapter-3 path:

```text
configs/RGBNT201/default.yml
configs/RGBNT201/ablations/chain20/a2_quality.yml
```

A3 is kept as an exploratory AGF/TPM branch. The latest checks did not show a
stable mAP improvement over A2, and continued training after resuming from A2
degraded. Do not treat A3, adapters, part branch, or auxiliary losses as the
default final stack unless they are revalidated with a stronger run.

## Diagnostic Results

| Row | Overlay | Best mAP | Best Rank-1 | Decision |
| --- | --- | ---: | ---: | --- |
| A0 | `a0_backbone.yml` | 80.76 | 77.63 | Baseline is healthy |
| A1 | `a1_hs_facss.yml` | 80.76 | 77.63 | Old row was structurally ineffective |
| A2 | `a2_quality.yml` | 85.43 | 83.49 | Current evidence-backed main path |
| A3 | `a3_agf.yml` | 83.94 | 81.70 | Strong but below A2 |
| A4 | `a4_adapter.yml` | 80.14 | 77.03 | Not retained as mainline |
| A5 | `a5_full.yml` | 74.27 | 71.05 | Full stack is harmful in this diagnostic |

Additional A3 TPM-branch resume testing from the A2 checkpoint matched A2 at the
first validation point on mAP (`85.43`) but did not exceed it, and the final
epoch-20 metric dropped to `81.94` mAP. That behavior supports stopping the
chapter-3 evidence path at A2 for now.
