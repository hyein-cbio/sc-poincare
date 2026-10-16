# MU integration and bounded scale-up protocol

## Frozen scope (before quality inspection)

Two explicit experimental bundles from Astra branch commit
`3489e586caaa5b0408a0751220d6349426457bfe` were integrated separately from the
then-default RFA path. After this study, the operator selected MU fixed as the
provisional public default; [migration](mu-default-migration.md) preserves explicit
`method="rfa"` and the low-level numerical/checkpoint formats. A subsequent
[display-only extraction](display-layout-deprecation.md) intentionally changes
`_disk.py`'s source hash, not the measured fitting behavior; old states are not
hash-migrated. Historical study checkpoints retain their original source tags:

- **MU fixed:** transformed mutual shortest-path global stress, a fixed-width
  union local teacher, classical-scaling initialization and tangent-coordinate Adam.
- **MU local-radius:** only the local teacher bandwidth changes to the kth-neighbor
  radius rule; global target, initial array and optimizer settings are shared.

These are bundled method comparisons, not an isolated causal test of graph
choice, objective or optimizer. Default promotion was not part of this frozen
experiment; the subsequent operator decision does not imply biological validation.

### Small stage

Use all six existing study input artifacts, uniformly cap at 384 cells with
seed 714 if larger, retain row order and existing explicit feature representation.
Moignard/Paul are the previous 384-cell inputs (Paul PCA was fit on that subset);
Olsson uses its previously provided full-data PCA20. No labels choose samples.

At 500 updates, compare MU fixed/local-radius with Poincare seeds 0/1/2,
RFA mutual baseline seeds 0/1/2, and native Euclidean MU controls seed 0 for
both bandwidth rules: **66 planned fits**, with failed conditions retained.
The RFA baseline uses explicit median sigma and the same input-only classical
mutual initializer as MU, not its normal spectral default; this is labeled
as a shared-initialization baseline. Raw KL is never compared across objectives.

### Full measured-data stage

Use all **3,934 Moignard cells** and **2,730 Paul cells**, original CSV row
order. Moignard retains 42 supplied features. Paul excludes labels before
explicit full-SVD PCA20 on all 2,730 rows; this differs from subset PCA and must
not be pooled as if the input representation were unchanged across stages.

At **1,500 updates**, seed **0** only, run RFA with shared classical initializer,
MU fixed/local-radius in Poincare geometry, and both native Euclidean controls:
**10 planned full fits**. This repeats the maintained implementation at full
measured-data size, not a new independent dataset or a full-data seed-robustness
claim. Within a geometry/bandwidth pair initial arrays/target hashes must match.
Euclidean shares tangent arrays, not exact initial distances or physical steps.

### Fixed settings and limits

- Astra's `GraphStressConfig` values are retained: graph k15, distance scale 2,
  gamma .1*2/3, local weight .03, relative floor .01, initial disk radius .2,
  jitter .0002, tangent Adam learning rate .02*2/3.
- RFA: self-inclusive symmetric KL, median sigma, gamma 2, learning rate .01,
  Riemannian clip 2.5, burn-in 5, patience equal to the update budget.
- Both fitters use exact all-column row chunks of 256; this is workspace chunking,
  not negative sampling. Every method selects its own lowest training-loss checkpoint, including epoch 0.
  Evaluation scores never select checkpoints or conditions.
- Resource-only pilots use 20 MU-fixed updates at N=512/1024/2048/3934, one thread;
  a full-size two-thread timing probe is separate from quality evaluation.
- Small stage: one thread per worker, at most two workers, 10-minute stage limit.
  Full stage: two threads per worker, at most two workers, **45-minute** stage
  limit. The supervisor samples aggregate child RSS, stops at **6 GiB**, and
  preserves at least **2 GiB** system-available memory. Sampling is not an
  instantaneous RSS guarantee. Keep all resource stops/failures/skips explicit.
- All preparations/fits retain the default 512 MiB dense preflight. No disabled
  memory guard, negative sampling, matrix-free approximation, kernel retuning,
  full Planaria, or n>4,096 exact run is included.
- No quality-dependent allocation of further full seeds or schedules. Unfinished
  1,500-update jobs are partial, not finished comparisons.

## Evaluation and views

- Fixed k15/k30 neighbor purity, macro purity and input-neighbor recall; 128
  global feature-distance anchors, seed 714. Euclidean controls are evaluated
  using native Euclidean distances, never projected into a disk.
- Fixed 600px disk / 4px marker visibility and intrinsic nearest-other distance
  for Poincare methods only. Native Euclidean axes are separate; no disk
  visibility score is fabricated for those controls.
- Include provided-expression neighborhood reconstruction MSE where the original
  numeric gene matrix is available: normalize each positive-variance gene by
  its variance, average over genes/cells; omit/report constant genes. This is
  input preservation, not held-out prediction or biological accuracy.
- Seed 0 is the fixed figure representative. Raw views are always retained.
  Full-data root-centered secondary views use one fixed input-selected root:
  Moignard PS feature medoid and Paul's unique `root` row. Root labels do not
  enter fitting or initialization. Use the same isometry per model to move that
  same cell to the origin; do not pick roots by appearance or metric score.
  If floating-point boundary precision prevents translation, record it rather
  than clipping display coordinates. No other alignment/fan/jitter/repel is used.
- Original data licenses apply. No authors' implementation or reference coordinates
  are used in this new study. Known development data, small init jitter, fixed
  budgets and absent biological marker/order validation limit conclusions.

## Results

### Current judgment

**Both bundles are worth retaining and investigating at a few-thousand-cell
scale.** The main step is RFA to MU fixed; local-radius is a smaller refinement,
not a universal winner. All four full Poincare MU jobs and both full RFA baselines
completed 1,500 updates. Full seed 0 is development evidence, not replication.

- Full Moignard: macro purity .5474 -> .5893/.5909 and gene MSE .4163 ->
  .3958/.3940; input-neighbor recall at k15 falls .2951 -> .2692/.2697.
- Full Paul: macro purity .4935 -> .5654/.5668 and gene MSE .9125 ->
  .9076/.9071. Fixed MU k15 input recall is slightly lower than RFA; local-radius
  is slightly higher. Several population-level losses remain below.
- Full raw-disk cross-label center crowding falls from about 95% to 37–38% in
  both datasets. This supports usability, not raster occlusion or biology.
- Native Euclidean MU also works well. On full Paul its local-radius control
  slightly exceeds Poincare local-radius on purity/macro/input recall and MSE,
  while Poincare retains higher global feature rank. Hyperbolic-specific
  superiority is not established. Full Moignard Euclidean controls initially hit
  the time limit and were subsequently completed by exact state continuation;
  they also show mixed geometry advantages rather than a universal winner.
- A single full MU fit costs roughly 8–17 minutes here. Do not equate feasible
  2,730/3,934-cell reference runs with atlas-scale practicality.

These results compare fixed method bundles, not isolated optimizer or geometry
effects. RFA uses the shared MU classical initializer and explicit median sigma;
it is not the literal default Scanpy initialization/schedule. Neither published
author coordinates nor a tuned reference implementation is a comparator.

### Completion and resource accounting

| Stage | Planned jobs | Status counts | Stage wall seconds | Observed peak aggregate child RSS (GiB) |
|---|---:|---|---:|---:|
| small | 66 | `{'completed': 63, 'failed': 3}` | 145.1 | 1.10 |
| full initial attempt | 10 | `{'completed': 8, 'interrupted': 2}` | 2701.2 | 3.09 |
| full exact-state continuation | 2 resumed, not new conditions | `{'completed': 2}` | 535.9 | 2.83 |

The supervisor samples RSS; worker metadata also records whole-process peak
RSS including imports, preparation, fitting and evaluation. Neither the dense
preflight nor sampled RSS is a promise of a hard instantaneous OS memory cap.
Small-stage RFA fails for all three krumsiek seeds at frozen median sigma
(~0.197988), before fitting. MU succeeds with its different predefined global
and local target construction, not by silently repairing the RFA kernel.

The full stage ended at its 45-minute limit. Both Paul Euclidean controls
completed; both Moignard Euclidean controls were interrupted during fitting,
with last persisted evaluated epochs **900** (fixed) and **0** (local-radius).
Those are checkpoint epochs, not exact total update counts at interruption.
No partial control scores or substitute smaller datasets enter the full table.
A subsequent operator request authorized a separately bounded continuation:
20-minute cap, the same two workers/two threads and memory policy. Both bare Adam
states were restored to the same total target of 1,500, not restarted with another
1,500 updates. Completion took 535.9 seconds; no learning rate, teacher, seed,
chunk or data changed. The four numerical/checkpoint source files were unchanged.
The final distinct full conditions are **10/10 completed**, not 12 new conditions.
The original resource stops remain in the record.

### Small quality table

K15 scores use all cells. Global rank is the mean per-anchor feature-distance
Spearman correlation, not pooled ranks or the source study's graph diagnostic.
MSE is variance-normalized provided-expression reconstruction (lower is better),
not prediction on unseen cells/genes. Disk visibility is unavailable for native
Euclidean controls by design, not treated as zero. Ranges are seed ranges, not
confidence intervals. Full-stage seed 0 alone is not seed robustness.

| Dataset | Method | Completed/planned | K15 purity ↑ | K15 macro ↑ | K15 input recall ↑ | Global feature rank ↑ | K15 expression MSE ↓ | Raw disk cross-label overlap % ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Moignard2015 | rfa | 3/3 | 0.5281–0.5292 | 0.4919–0.4925 | 0.4490–0.4516 | 0.7355–0.7355 | 0.4774–0.4777 | 76.04–76.04 |
| Moignard2015 | mu_fixed | 3/3 | 0.5549–0.5561 | 0.5178–0.5187 | 0.4993–0.5040 | 0.7960–0.7964 | 0.4427–0.4455 | 3.65–4.69 |
| Moignard2015 | mu_local | 3/3 | 0.5559–0.5576 | 0.5191–0.5207 | 0.4977–0.5035 | 0.7963–0.7965 | 0.4425–0.4442 | 2.34–4.95 |
| Moignard2015 | euclidean_fixed | 1/1 | 0.5594 | 0.5234 | 0.5007 | 0.8009 | 0.4432 | native axes |
| Moignard2015 | euclidean_local | 1/1 | 0.5576 | 0.5214 | 0.5012 | 0.8022 | 0.4426 | native axes |
| MyeloidProgenitors | rfa | 3/3 | 0.7446–0.7453 | 0.6559–0.6565 | 0.7597–0.7630 | 0.8311–0.8314 | 0.0329–0.0331 | 84.64–84.64 |
| MyeloidProgenitors | mu_fixed | 3/3 | 0.7641–0.7651 | 0.6740–0.6750 | 0.9260–0.9280 | 0.9084–0.9093 | 0.0029–0.0029 | 3.39–3.65 |
| MyeloidProgenitors | mu_local | 3/3 | 0.7613–0.7642 | 0.6715–0.6743 | 0.9217–0.9260 | 0.9086–0.9091 | 0.0028–0.0028 | 3.91–4.17 |
| MyeloidProgenitors | euclidean_fixed | 1/1 | 0.7630 | 0.6731 | 0.9278 | 0.8962 | 0.0029 | native axes |
| MyeloidProgenitors | euclidean_local | 1/1 | 0.7637 | 0.6738 | 0.9262 | 0.8956 | 0.0028 | native axes |
| Olsson | rfa | 3/3 | 0.7534–0.7534 | 0.7480–0.7480 | 0.5935–0.5939 | 0.8736–0.8736 | 0.6673–0.6676 | 56.54–56.54 |
| Olsson | mu_fixed | 3/3 | 0.7794–0.7825 | 0.7769–0.7804 | 0.6501–0.6511 | 0.9391–0.9398 | 0.6496–0.6498 | 0.00–0.00 |
| Olsson | mu_local | 3/3 | 0.7794–0.7825 | 0.7752–0.7811 | 0.6534–0.6555 | 0.9344–0.9386 | 0.6483–0.6490 | 0.00–0.00 |
| Olsson | euclidean_fixed | 1/1 | 0.7803 | 0.7744 | 0.6506 | 0.9345 | 0.6500 | native axes |
| Olsson | euclidean_local | 1/1 | 0.7743 | 0.7691 | 0.6517 | 0.9291 | 0.6500 | native axes |
| Paul | rfa | 3/3 | 0.4753–0.4771 | 0.3210–0.3230 | 0.4434–0.4451 | 0.8269–0.8271 | 0.9563–0.9564 | 82.03–82.29 |
| Paul | mu_fixed | 3/3 | 0.4997–0.5078 | 0.3539–0.3604 | 0.5316–0.5337 | 0.7615–0.7628 | 0.9443–0.9444 | 1.04–2.08 |
| Paul | mu_local | 3/3 | 0.5059–0.5116 | 0.3613–0.3656 | 0.5351–0.5378 | 0.7601–0.7609 | 0.9438–0.9442 | 1.30–2.60 |
| Paul | euclidean_fixed | 1/1 | 0.4984 | 0.3480 | 0.5300 | 0.7347 | 0.9468 | native axes |
| Paul | euclidean_local | 1/1 | 0.5085 | 0.3582 | 0.5342 | 0.7373 | 0.9449 | native axes |
| ToggleSwitch | rfa | 3/3 | 0.8040–0.8043 | 0.5853–0.5856 | 0.7807–0.7810 | 0.8073–0.8075 | 0.0304–0.0306 | 43.50–43.50 |
| ToggleSwitch | mu_fixed | 3/3 | 0.8000–0.8020 | 0.5791–0.5810 | 0.9437–0.9480 | 0.8820–0.8829 | 0.0127–0.0128 | 4.00–4.00 |
| ToggleSwitch | mu_local | 3/3 | 0.7997–0.8020 | 0.5788–0.5811 | 0.9400–0.9463 | 0.8805–0.8818 | 0.0132–0.0134 | 5.00–5.00 |
| ToggleSwitch | euclidean_fixed | 1/1 | 0.8010 | 0.5799 | 0.9480 | 0.8863 | 0.0127 | native axes |
| ToggleSwitch | euclidean_local | 1/1 | 0.8013 | 0.5800 | 0.9413 | 0.8851 | 0.0132 | native axes |
| krumsiek11_blobs | rfa | 0/3 (failed) | — | — | — | — | — | — |
| krumsiek11_blobs | mu_fixed | 3/3 | 0.8377–0.8380 | 0.7247–0.7254 | 0.6969–0.7014 | 0.7558–0.7589 | 0.0382–0.0410 | 8.85–9.11 |
| krumsiek11_blobs | mu_local | 3/3 | 0.8394–0.8401 | 0.7280–0.7292 | 0.7017–0.7040 | 0.7552–0.7583 | 0.0367–0.0375 | 8.85–9.38 |
| krumsiek11_blobs | euclidean_fixed | 1/1 | 0.8378 | 0.7251 | 0.6998 | 0.7521 | 0.0395 | native axes |
| krumsiek11_blobs | euclidean_local | 1/1 | 0.8392 | 0.7276 | 0.7059 | 0.7516 | 0.0374 | native axes |

### Full quality table

K15 scores use all cells. Global rank is the mean per-anchor feature-distance
Spearman correlation, not pooled ranks or the source study's graph diagnostic.
MSE is variance-normalized provided-expression reconstruction (lower is better),
not prediction on unseen cells/genes. Disk visibility is unavailable for native
Euclidean controls by design, not treated as zero. Ranges are seed ranges, not
confidence intervals. Full-stage seed 0 alone is not seed robustness.

| Dataset | Method | Completed/planned | K15 purity ↑ | K15 macro ↑ | K15 input recall ↑ | Global feature rank ↑ | K15 expression MSE ↓ | Raw disk cross-label overlap % ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Moignard2015 | rfa | 1/1 | 0.5843 | 0.5474 | 0.2951 | 0.6930 | 0.4163 | 95.14 |
| Moignard2015 | mu_fixed | 1/1 | 0.6255 | 0.5893 | 0.2692 | 0.8073 | 0.3958 | 37.42 |
| Moignard2015 | mu_local | 1/1 | 0.6268 | 0.5909 | 0.2697 | 0.8081 | 0.3940 | 37.93 |
| Moignard2015 | euclidean_fixed | 1/1 (exact resume) | 0.6240 | 0.5875 | 0.2681 | 0.8115 | 0.3950 | native axes |
| Moignard2015 | euclidean_local | 1/1 (exact resume) | 0.6251 | 0.5888 | 0.2701 | 0.8118 | 0.3954 | native axes |
| Paul | rfa | 1/1 | 0.5424 | 0.4935 | 0.3224 | 0.8228 | 0.9125 | 95.27 |
| Paul | mu_fixed | 1/1 | 0.5974 | 0.5654 | 0.3207 | 0.8799 | 0.9076 | 37.95 |
| Paul | mu_local | 1/1 | 0.5973 | 0.5668 | 0.3236 | 0.8784 | 0.9071 | 37.84 |
| Paul | euclidean_fixed | 1/1 | 0.5969 | 0.5632 | 0.3204 | 0.8689 | 0.9078 | native axes |
| Paul | euclidean_local | 1/1 | 0.5980 | 0.5673 | 0.3246 | 0.8699 | 0.9067 | native axes |

### Full-data execution details

| Dataset | Method | Status | Actual updates | Best epoch | Preparation s | Fit s | Process peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| Paul | mu_fixed | completed | 1500 | 1500 | 1.4 | 508.9 | 1.17 |
| Moignard2015 | mu_fixed | completed | 1500 | 1500 | 3.9 | 1017.5 | 1.63 |
| Paul | mu_local | completed | 1500 | 1497 | 1.6 | 496.5 | 1.24 |
| Moignard2015 | mu_local | completed | 1500 | 1493 | 4.0 | 997.6 | 1.73 |
| Paul | rfa | completed | 1500 | 1500 | 2.2 | 425.4 | 1.21 |
| Moignard2015 | rfa | completed | 1500 | 1500 | 5.2 | 895.4 | 1.60 |
| Paul | euclidean_fixed | completed | 1500 | 1500 | 1.6 | 279.0 | 1.14 |
| Moignard2015 | euclidean_fixed | completed by resume | 1500 total | 1500 | 4.2 (continuation) | 237.3 (continuation) | 1.41 |
| Paul | euclidean_local | completed | 1500 | 1500 | 1.6 | 295.3 | 1.14 |
| Moignard2015 | euclidean_local | completed by resume | 1500 total | 1500 | 4.3 (continuation) | 527.2 (continuation) | 1.48 |

Continuation timings cover only the resumed process; they must not be read as
full-run latency or erase the original interrupted work. The fixed-control state
resumed from epoch 900, local-radius from epoch 0. Their complete loss histories
retain the original evaluated prefix. Fixed budgets and best checkpoints are not
evidence of convergence. Costs
include common-initializer construction; they do not benchmark the ordinary
RFA spectral-start API's production latency. Equal updates are not equal physical
steps or optimization convergence across Riemannian fitting and tangent Adam.

### Full-data population counterexamples

All observed populations are retained. Deltas are candidate minus RFA at seed 0;
negative purity/recall changes are not removed. Label granularity, constant
size ceilings and legitimate continuous differentiation limit interpretation.

#### Moignard2015

| Population | Cells | Fixed purity delta | Local-radius purity delta | Fixed input-recall delta | Local-radius input-recall delta |
|---|---:|---:|---:|---:|---:|
| 4SFG | 983 | -0.0028 | -0.0022 | -0.0013 | -0.0020 |
| 4SG | 770 | +0.1108 | +0.1124 | -0.0363 | -0.0365 |
| HF | 1005 | +0.0430 | +0.0422 | -0.0411 | -0.0397 |
| NP | 552 | -0.0002 | -0.0005 | -0.0324 | -0.0307 |
| PS | 624 | +0.0588 | +0.0653 | -0.0216 | -0.0214 |

#### Paul

| Population | Cells | Fixed purity delta | Local-radius purity delta | Fixed input-recall delta | Local-radius input-recall delta |
|---|---:|---:|---:|---:|---:|
| 10GMP | 153 | +0.0301 | +0.0314 | +0.0017 | +0.0065 |
| 11DC | 30 | +0.1356 | +0.1511 | +0.1444 | +0.1600 |
| 12Baso | 69 | +0.0309 | +0.0280 | +0.0029 | +0.0010 |
| 13Baso | 300 | +0.0640 | +0.0642 | -0.0247 | -0.0222 |
| 14Mo | 373 | +0.0497 | +0.0497 | -0.0642 | -0.0590 |
| 15Mo | 186 | +0.0871 | +0.0824 | -0.0161 | -0.0093 |
| 16Neu | 164 | +0.0760 | +0.0736 | +0.0142 | +0.0175 |
| 17Neu | 22 | +0.3697 | +0.3606 | +0.1909 | +0.1697 |
| 18Eos | 9 | +0.2370 | +0.2741 | +0.1926 | +0.2593 |
| 19Lymph | 31 | -0.0151 | -0.0151 | +0.0645 | +0.0581 |
| 1Ery | 43 | +0.0279 | +0.0171 | +0.1194 | +0.1194 |
| 2Ery | 329 | +0.0415 | +0.0426 | +0.0140 | +0.0215 |
| 3Ery | 246 | +0.0848 | +0.0851 | +0.0388 | +0.0425 |
| 4Ery | 124 | -0.0102 | -0.0102 | +0.0134 | +0.0215 |
| 5Ery | 180 | -0.0174 | -0.0174 | -0.0237 | -0.0170 |
| 6Ery | 173 | +0.0200 | +0.0227 | -0.0173 | -0.0235 |
| 7MEP | 167 | +0.1313 | +0.1329 | -0.0192 | -0.0236 |
| 8Mk | 68 | +0.0794 | +0.0814 | +0.0912 | +0.0863 |
| 9GMP | 62 | +0.0161 | +0.0118 | -0.0333 | -0.0484 |
| root | 1 | +0.0000 | +0.0000 | +0.0667 | +0.0667 |

### Resource-only pilot

The following are cold preparation + 20-update timings, not quality-selected
conditions or a fitted extrapolation model. One thread, MU fixed, Moignard
feature subsets chosen without labels (seed 714); full row order at N=3934.

| Cells | Seconds | Observed process peak GiB | Exit code |
|---:|---:|---:|---:|
| 512 | 2.61 | 0.50 | 0 |
| 1024 | 3.22 | 0.61 | 0 |
| 2048 | 6.98 | 0.85 | 0 |
| 3934 | 22.96 | 1.51 | 0 |

A separate two-thread full-size timing probe completed 20 updates in 17.08s.
The full study fixes two threads per worker, rather than selecting threads by
quality. A launcher import-path error was corrected before that timing probe;
no model parameter or input was changed.

## Geometry and display interpretation

Raw Poincare views share 600px disks and 4px markers. Full-data secondary views
center the same predeclared input cell per model (PS feature medoid / unique
Paul root). Visibility can change under that isometry while intrinsic distances
do not; the plot audit validates 16 fixed anchor rows against all cells. Any
precision failure suppresses that centered panel without clipping or choosing
an alternative root. Raw views remain. Euclidean views use common native axes
for the two controls, not disk projection or a fabricated disk-overlap score.

Lower overlap supports a display-usability endpoint, not biological correctness.
Input-neighbor recall and expression error use data that helped construct the
input representation/teacher; they are not independent successes or held-out
prediction. Development labels/plots informed earlier research. No independent
data, marker continuity, branch order or pseudotime validation is established.

## Scale-up decision boundary

This exact implementation provides an interpretable reference at a few thousand
cells, not an atlas-scale claim. Targets remain dense O(N²); dense classical
scaling is super-quadratic work. At 21,612 cells a float64 matrix alone is
about 3.74 GB, and the four-array preflight is about 14.95 GB, well above the
unchanged 512 MiB budget. No full Planaria result is inferred or attempted.
Further scale-up requires separately validated approximation/matrix-free or
landmark work against this exact reference, not simply disabling the guard.

## Software validation and provenance

- Imported opt-in implementation/configurations from Astra commit `3489e58`.
- Both bandwidth rules match the frozen branch exactly for 20 updates on the
  fixed Paul subset: initial/current/best coordinates and loss traces, plus
  input/target/teacher/initializer hashes. This does not establish full-run
  byte identity across environments or biological superiority.
- Study-time local complete suite: **365 passed**, two existing exact-zero-target warnings.
- Study-time separate Torch-free installed-wheel checks: **188 passed, 19 Torch-dependent
  skips**; preparation and explicit lazy-Torch failure were also smoke-tested
  outside the source tree. The [Actions workflow](../.github/workflows/tests.yml)
  defines automated checks rather than asserting a remote run status here.
- Exact independent/chunk-gradient, finite-difference, persisted Adam resume,
  native Euclidean, expression-MSE, owned-child stop/time/memory handling and
  missing-control plotting tests are included. The subsequent default adapter
  preserved explicit RFA defaults and numerical sources at that stage. The later
  [display-layout cleanup](display-layout-deprecation.md) has a separate,
  intentional whole-file source/checkpoint boundary; historical results are not
  retagged. See [migration and additional regressions](mu-default-migration.md).

## Latest local artifacts

`data/graph_diagnostics_integration/` is replaced by this latest MU study, not an
accumulating source/checkpoint archive. It contains shared stage inputs, compact
per-job summaries/results, one last checkpoint per long job, aggregate scores
and predetermined seed-0 figures. Dense targets and source trees are not saved.
Original data licenses apply; no raw CSVs, original reference coordinates or
private session exports are added to Git. Local optimizer states are not a
guarantee of durable off-machine backup.
