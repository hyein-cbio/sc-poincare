# Full Planaria: exact MU fixed, 1,500 updates (2026-11-19)

On a 48 GiB Mac, exact MU fixed fit all **21,612 Planaria cells** with a
bounded dense-memory override. At update 1,500, it preserved graph-distance
rank and both measured local neighborhoods better than the authors' public
Poincaré Maps coordinates. The public coordinates retained higher global PCA
distance rank and better cell-weighted label purity. This is a **single-seed
comparison of saved embeddings**, not an equal-compute comparison, convergence
claim, or biological validation.

## Input and training

- Fitting source: [`68aacde`](https://github.com/hyein-cbio/sc-poincare/commit/68aacdeb751de5354cc76b3c7eb99a813ed9a202).
  Use the authors' [Planaria CSV](https://raw.githubusercontent.com/facebookresearch/PoincareMaps/7ffa518cabe6e6424a875d480c4298bda3bd9678/datasets/Planaria.csv)
  in its original row order. Use the 50 supplied PCA columns as float64,
  with no additional PCA, scaling, filtering, or reordering. Labels were
  read only after fitting. The CSV SHA-256 is
  `d59a30b1d2ddf959912d4a1fc4c20910de88823f2a67223d9e1809db065e382d`;
  the C-contiguous float64 feature-array SHA-256 is
  `aeee2b739c71f607bed827983198f32e11b36a3351a3789b451c53a2ea6fdb28`.
- Public AnnData `poincare_map`, `method="mu"`, `use_rep="X"`, seed 0,
  batch size 256, default MU fixed parameters (global-width local teacher,
  k=15, tangent Adam). No root recentering or display adjustment.
- The original fit started from classical scaling. Continuations at 5, 100,
  and 500 carried the full Adam state and loss trace rather than starting a
  new optimizer. The continuation drivers reproduced the saved loss and
  coordinates at each resume point before advancing. The endpoint was fixed
  at 1,500 updates; no comparison metric chose a checkpoint or model. At
  update 1,500, final and lowest-training-loss coordinates were the same.
- On the final continuation, Python 3.14.10, NumPy 2.5.5, SciPy 1.18.1,
  scikit-learn 1.9.1, AnnData 0.13.4 and Torch 2.14.1 ran on macOS arm64,
  CPU float64, with six Torch and BLAS threads. The batch retains all
  comparison columns. A separate watchdog enforced a 22 GiB sampled RSS
  threshold, 5 GiB available-memory reserve and a time limit. The dense-array
  guard was set to **16 GiB**, not disabled; its default remains 512 MiB.

This is the public fit call for a fresh run. It is not the watchdog or the
staged replay harness used for the recorded run. Supply the verified feature
matrix as `adata.X` with the original cell order and set resource limits
outside this call before attempting a full fit.

```python
from sc_poincare import tl

tl.poincare_map(
    adata, method="mu", use_rep="X", random_state=0,
    n_epochs=1500, batch_size=256, max_dense_bytes=16 * 1024**3,
)
```

## Comparison protocol

The reference is the authors' published
[numeric coordinate CSV](https://raw.githubusercontent.com/facebookresearch/PoincareMaps/7ffa518cabe6e6424a875d480c4298bda3bd9678/results/Planaria_MFIsym15_klSym_sigma%3D2.00_gamma%3D2.00_lr%3D1.25_lrm%3D1.00_pca%3D0_epochs%3D10000_batchsize%3D2000_connected.csv),
SHA-256 `8fc73b7c41ce1661d83a4f6ffd208de6af211179e78e70b7ff169da0401249ef`.
It has no cell IDs, so the comparison assumes its row order matches the
authors' Planaria input CSV. The checksum does not prove that alignment.
No reference implementation was used in the comparison; do not infer matched
preprocessing, training time or update budgets from the reference filename.

Both embeddings are scored with Poincaré distances, with self pairs excluded:

- **Graph metrics:** shortest-path distances on the same frozen input-space
  mutual k=15 graph with component bridges and Euclidean edge lengths.
  The full graph was saved in the earlier scale study's row permutation
  (NumPy seed 2027, then original row 14,646 swapped to the front). In that
  order, use 128 fixed anchors (seed 41), mean 15-neighbor overlap, and one
  pooled Spearman over nonself anchor-to-cell distances. The archived graph
  file SHA-256 is `329725d8b5a1e581c17090c4799f94ac7ba4c155b30b910c06d4b32e075c81ed`.
  The sensitivity evaluation uses k=20 and 128 anchors (seed 714), with
  the same row permutation and retrieval size 15. This changes both graph
  and anchors; it does not isolate either factor.
- **Direct PCA and label metrics:** in original cell order,
  `evaluate_neighborhoods(..., ks=(15,), global_n_anchors=128, global_random_state=714)`
  compares Euclidean PCA neighbors with embedding neighbors over all cells,
  averages row-wise Spearman across the fixed anchors, and reports cell-weighted
  and equal-population purity across 51 labels. Label purity is descriptive;
  the metrics are not the paper's Qlocal/Qglobal. Graph rank should not be
  treated as independent of MU's graph-distance objective.

## Results

Higher is better. The [checkpoint CSV](planaria-mu-fixed-1500-2026-11-19.csv)
contains all aggregate scores, including the early checkpoints and both
sensitivity metrics. It contains no raw input data, labels or individual
coordinates.

| Metric | Published reference | MU 500 | MU 1,000 | MU 1,500 |
|---|---:|---:|---:|---:|
| Graph distance rank (Spearman) | 0.7723 | 0.8800 | 0.8933 | **0.8942** |
| Direct PCA distance rank (Spearman) | **0.8037** | 0.7647 | 0.7772 | 0.7780 |
| Graph 15-neighbor recall | 12.14% | **19.95%** | 19.79% | 19.79% |
| Direct PCA 15-neighbor recall | 13.73% | 19.01% | 19.27% | **19.29%** |
| Cell-weighted 15-neighbor label purity | **80.12%** | 78.19% | 78.71% | 78.77% |
| Equal-population 15-neighbor label purity | 67.16% | 66.66% | 67.33% | **67.45%** |

Graph rank fell from 0.775 at initialization to 0.405 at update 20, then
recovered and passed the published value by update 200. At update 1,500,
the k=20 sensitivity comparison also favors MU on graph distance rank
(0.8848 vs 0.8084) and recall (17.92% vs 11.67%). Improvements after 500
were smaller. The highest observed k=15 graph recall was at update 700
(20.42%), and the highest observed equal-population purity was at 1,300
(67.47%). These observations are not a new stopping rule. The final
0.29-percentage-point lead in equal-population purity is descriptive, not
evidence that labels or lineages are generally better resolved.

## Resources and preservation

The 500-to-1,500 continuation, including preparation and result export, took
**2 h 48 min 11 s** with **13.87 GiB** peak process RSS and no resource stop.
Preparation took 190.1 s and fitting took 9,888.6 s. These are the timings for
that continuation, not a measured cold-start time or hardware guarantee.
The final H5AD, NPZ and Adam state were reread and cross-checked for cell
identity, loss trace and coordinates. All 44 files in the local SHA-256
manifest verified again before documenting this result. The final H5AD
SHA-256 is `74059ee57e5532c29362bb384b582cebb21e1d418747f01d639c7cffedbb1514`;
the Adam checkpoint SHA-256 is
`74f02bf87ad597d498b4b7d2d5b0a3196f8b0c8890211cf50809a34f539ab186`.

The local run directory is `data/planaria/mu-fixed-1500-20261119/` (ignored by
Git). It holds the runner, protocol, resource logs, checkpoint and coordinate
files every 100 updates from 600 onward, full per-population evaluations,
`fit.h5ad` and `SHA256SUMS`. Inputs, reference coordinates, embeddings and
model state are not vendored; original artifact terms apply. Local preservation
is not a claim of an off-machine backup. No algorithm or default was changed.
Scale-up beyond this range remains [deferred](development-roadmap.md#beyond-planaria-deferred).
