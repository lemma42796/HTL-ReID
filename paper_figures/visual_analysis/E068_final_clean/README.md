# E068 final clean visual analysis

These artifacts were generated from the frozen E060 epoch-50 checkpoint on RGBNT201 with a single view, no TTA, no re-ranking, and no post-run descriptor tuning.

- `fig_hs_union_selection.png`: final HS union-broadcast selection and frequency additions. The remote raw filename was `fig_hs_consensus_specific.png`; only the local paper-facing copy was renamed because the selected fallback configuration has `HS_CONSENSUS_SPECIFIC=0`.
- `fig_facr_routing.png`: mean routing weights for the three implementation-level ACI stages that constitute the gradient-isolated FACR branch.
- `fig_robustness.png`: missing-modality comparison and controlled single-modality occlusion curves.
- `routing_statistics.npz`: full-test routing arrays with PID and camera labels.
- `summary.json`: original E068 generation summary.

The authoritative protocol and interpretation are recorded in `实验记录/E068_final_visual_analysis.md`.
