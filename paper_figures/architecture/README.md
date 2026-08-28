# HTL-ReID Architecture Figure

`htl_reid_architecture.svg` is the editable source for the paper's overall
architecture figure. Matching vector PDF and high-resolution PNG exports are
provided beside it.

The diagram follows the frozen paper-facing names HS, ACI, PLR, and DHF. It
reflects the implementation at commit `c340309`: ACI reads complete backbone
tokens under the shared HS mask, PLR reads HS-selected patch features, and DHF
runs independently on complete backbone tokens through three single-modal,
three bi-modal, and one all-modal route. The final descriptor shows the frozen
E045 inference weights; PLR remains visible as a trained branch but has weight
zero at inference.

Files:

- `htl_reid_architecture.svg`: editable vector source.
- `htl_reid_architecture.pdf`: paper-ready vector export.
- `htl_reid_architecture.png`: high-resolution raster export.
