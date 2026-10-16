# Union graph: display visibility across additional cases

## Fixed extension protocol

The user requested additional cases after observing wider visual spread in the
initial Moignard/Paul plots. Display criteria on those initial two datasets are
therefore exploratory/post-hoc. The following extension and display definitions
were fixed before inspecting the four new paired fits. These datasets were used
in earlier development, so none is a blinded independent holdout.

| Added case | Cells | Explicit fitting representation |
|---|---:|---|
| ToggleSwitch | 200 (all) | 2 supplied features, unchanged |
| MyeloidProgenitors | 640 (all) | 11 supplied features, unchanged |
| krumsiek11_blobs | 1,280 (all) | 11 supplied features, unchanged |
| Olsson | 382 (all) | labels removed before full-SVD PCA20 of 532 features |

ToggleSwitch, MyeloidProgenitors and the related blobs stress case are synthetic
cases, not three independent measured biological validations. Olsson is the
additional measured-data case. Original CSV order is retained, with unique
string IDs derived from original row indices; no labels choose cells or settings.
No normalization or raw-count interpretation is inferred from a dataset name.

- Unchanged fitter: self-inclusive symmetric KL, gamma 2, learning rate .01,
  Riemannian clipping 2.5, burn-in 5, patience 500, row chunks 128.
- 500 updates, seeds 0/1/2, mutual/union = 24 additional fits. One CPU thread,
  512 MiB dense preflight and a 600-second additional-fit batch limit.
- Resolve median sigma once from the mutual graph, reuse that exact number;
  generate mutual spectral initialization per seed, share the exact array.
- Report k15/30 neighborhood diagnostics and fixed 128-anchor global feature
  rank (seed 714) without choosing winners from these scores.
- Plot seed 0 in advance, not a favorable seed. No root translation, alignment,
  fan, repulsion, jitter or other display transformation.

## Fixed display definitions

The disk has diameter 600 pixels; circular marker diameter is 4 pixels. Shared
axes are exactly [-1,1] on both axes. Raw learned coordinates are used.

- **Population display width:** twice the 90th percentile Euclidean distance
  from each population's arithmetic display center. Macro width weights
  populations equally, including zero singleton widths. This describes spread,
  not between-population separation or uniform use of screen area.
- **Marker-center overlap fraction:** fraction of cells whose nearest other
  center is strictly less than 4 pixels away. Cross-label overlap restricts the
  candidate neighbors to other populations. This is a geometric diagnostic,
  not raster occlusion, alpha-compositing or a human readability assessment.
- **Median nearest-center distance:** screen-pixel distance to the nearest
  other cell, irrespective of label.
- **Occupied 10-pixel bins:** number of occupied cells of a fixed screen grid.
  This is grid/orientation and cell-count dependent; only within-case paired
  differences are meaningful, not a ranking of different datasets.
- **Intrinsic nearest-other-population distance:** median hyperbolic distance
  to each cell's nearest differently labeled cell. Report alongside display
  width; a wider screen population can become intrinsically closer to others.
  One-label comparisons have undefined cross-label quantities, not zeroes.

All observed populations and all seeds remain in the saved output. Wider is a
potential usability benefit, but broadening into unrelated populations is not
an automatic quality gain. Label mixing can also represent legitimate continuous
transitions. No thresholds are selected after looking at results, and the
algorithm/default is not changed to increase these display scores.

## Execution and results

**Union has a display-usability advantage in some of these cases, not universal
superiority.** At the fixed 4px marker threshold, cross-label center crowding
falls in four of five completed cases across all three seeds. ToggleSwitch is
a counterexample. Increased population width is not universal: the initial
Moignard/Paul impression is better described by less center crowding and more
occupied screen bins than by every population becoming wider.

The additional four-case schedule planned 24 fits. **18 completed**, each 500 updates; the six krumsiek11_blobs paired fits were not started because reference graph preparation failed. The first batch ran for 38.21s; Olsson completed in another 12.56s within the remaining original 600-second wall budget.
Including the initial 12 fits, there are **30 completed fits / 15 paired seeds**
across five successful cases, with one explicit preparation failure. No setting
was changed to rescue the failed stress case, and no completed pair was dropped.

### Screen-crowding proxy: all three seeds

Percentages refer to cell centers, not rendered occlusion. Negative delta means
less cross-label center crowding. Do not compare absolute scores across datasets.

| Case | Cells | Mutual cross-label overlap % | Union cross-label overlap % | Paired change (percentage points) |
|---|---:|---|---|---|
| Moignard2015 | 384 | 77.86 to 78.39 | 71.09 to 71.09 | -7.29 to -6.77 |
| Paul | 384 | 84.90 to 85.16 | 78.39 to 79.17 | -6.77 to -5.99 |
| ToggleSwitch | 200 | 44.00 to 44.00 | 44.50 to 44.50 | +0.50 to +0.50 |
| MyeloidProgenitors | 640 | 70.94 to 70.94 | 69.84 to 69.84 | -1.09 to -1.09 |
| Olsson | 382 | 53.40 to 53.40 | 50.00 to 50.00 | -3.40 to -3.40 |
| krumsiek11_blobs | 1,280 | Not fitted | Not fitted | Reference graph underflow |

### Spread, occupied bins and intrinsic geometry

Width ratio is union/mutual macro population width. Values below 1 mean smaller
within-population display width, not necessarily poorer readability. Intrinsic
delta is the median nearest-other-population hyperbolic distance change; it is
not display distance and does not measure marker visibility.

| Case | Macro width ratio | Occupied-bin delta | Intrinsic nearest-other delta | K15 input-recall delta | K15 macro-purity delta | Global feature-rank delta |
|---|---|---|---|---|---|---|
| Moignard2015 | 0.844 to 0.844 | +16 to +18 | +0.006 to +0.016 | -0.096 to -0.088 | -0.005 to +0.002 | +0.025 to +0.025 |
| Paul | 0.895 to 1.072 | +16 to +22 | -0.124 to -0.120 | -0.044 to -0.042 | -0.003 to -0.000 | +0.021 to +0.024 |
| ToggleSwitch | 1.021 to 1.021 | +2 to +3 | -0.723 to -0.715 | -0.018 to -0.018 | -0.030 to -0.029 | +0.001 to +0.002 |
| MyeloidProgenitors | 0.989 to 0.990 | +0 to +2 | -1.353 to -1.285 | -0.054 to -0.046 | +0.006 to +0.007 | -0.035 to -0.033 |
| Olsson | 0.843 to 0.843 | -1 to +1 | -0.245 to -0.244 | -0.045 to -0.043 | +0.019 to +0.020 | +0.023 to +0.023 |

- The initial two datasets have more occupied bins and less cross-label center
  crowding in union. This supports the user's usability observation even though
  macro width is lower in Moignard and seed-sensitive in Paul.
- Olsson has lower center crowding and higher macro purity/global rank, but
  lower input-neighbor recall and lower intrinsic nearest-other distance. Its
  occupied-bin change is not seed-consistently positive.
- MyeloidProgenitors also shows a global-rank loss in every seed, unlike the
  initial two datasets. Its small reduction in crowding is not a global-quality
  gain. ToggleSwitch slightly broadens but has worse center crowding and macro purity.
- All five completed cases lose k15 input-neighbor recall in every seed. That
  does not invalidate a usability benefit; it is a cost to disclose for this
  particular graph/optimizer/update-budget combination.

### Failed case: not a missing unfavorable plot

krumsiek11_blobs failed during mutual reference graph preparation, before any
of its six fits. Median sigma resolved to approximately **0.0943468**; Gaussian
underflow left the weighted graph in **three components**. The fixed protocol
cannot proceed to the shared-reference paired experiment. This does **not**
establish that union would independently fail, nor is it evidence of its success.
No artificial bridge floor, sigma search, data truncation or preprocessing
rescaling was introduced. A different bandwidth rule would be a separately
declared experiment. Olsson was continued independently with the remaining
original wall-time budget; it was not a retuned retry of the failed case.

### Population-level display changes

Every observed population is retained. Width change is pixels; overlap change
is percentage points. These are ranges across seeds 0/1/2. Small groups, singleton
roots and fine-label transitions require interpretation; wider is not automatically better.

#### Moignard2015

| Population | Cells | Width delta (px) | Cross-label overlap delta (pp) |
|---|---:|---|---|
| 4SFG | 95 | -310.59 to -310.45 | -1.05 to -1.05 |
| 4SG | 69 | +61.21 to +61.27 | +8.70 to +8.70 |
| HF | 103 | -80.80 to -80.78 | -21.36 to -20.39 |
| NP | 54 | -39.44 to -39.38 | -16.67 to -14.81 |
| PS | 63 | -69.84 to -69.82 | -4.76 to -3.17 |

#### Paul

| Population | Cells | Width delta (px) | Cross-label overlap delta (pp) |
|---|---:|---|---|
| 10GMP | 21 | -136.00 to +104.67 | -52.38 to -47.62 |
| 12Baso | 12 | -18.39 to +25.94 | -58.33 to -50.00 |
| 13Baso | 44 | -25.54 to -22.85 | +2.27 to +2.27 |
| 14Mo | 60 | -9.41 to -7.47 | +0.00 to +1.67 |
| 15Mo | 19 | +2.02 to +3.86 | +0.00 to +0.00 |
| 16Neu | 22 | +4.46 to +5.52 | +0.00 to +0.00 |
| 17Neu | 2 | +15.41 to +82.91 | -50.00 to -50.00 |
| 18Eos | 3 | +53.68 to +59.48 | -33.33 to -33.33 |
| 19Lymph | 4 | -72.51 to -34.35 | +0.00 to +0.00 |
| 1Ery | 4 | -0.42 to +1.21 | +0.00 to +0.00 |
| 2Ery | 55 | -0.81 to -0.70 | +1.82 to +1.82 |
| 3Ery | 35 | +3.53 to +4.18 | +0.00 to +0.00 |
| 4Ery | 14 | -64.04 to -61.99 | +0.00 to +0.00 |
| 5Ery | 22 | +4.17 to +5.90 | +0.00 to +0.00 |
| 6Ery | 27 | -11.63 to -8.00 | +0.00 to +3.70 |
| 7MEP | 17 | -341.50 to -265.11 | -11.76 to -11.76 |
| 8Mk | 13 | +243.79 to +448.97 | -15.38 to -7.69 |
| 9GMP | 10 | -130.07 to +168.49 | -50.00 to -50.00 |

#### ToggleSwitch

| Population | Cells | Width delta (px) | Cross-label overlap delta (pp) |
|---|---:|---|---|
| branch1 | 75 | -3.64 to -3.48 | +0.00 to +0.00 |
| branch2 | 55 | +40.07 to +40.28 | -1.82 to -1.82 |
| inter1 | 24 | -6.09 to -5.59 | +4.17 to +4.17 |
| inter2 | 45 | +2.79 to +3.37 | +2.22 to +2.22 |
| root | 1 | +0.00 to +0.00 | +0.00 to +0.00 |

#### MyeloidProgenitors

| Population | Cells | Width delta (px) | Cross-label overlap delta (pp) |
|---|---:|---|---|
| branchEr | 80 | +2.28 to +2.70 | -1.25 to -1.25 |
| branchMe | 80 | -0.63 to +0.53 | +0.00 to +0.00 |
| branchMo | 79 | +14.96 to +15.37 | +1.27 to +1.27 |
| branchNe | 80 | -36.70 to -36.21 | -3.75 to -3.75 |
| erythrocyt | 80 | -0.58 to -0.54 | -5.00 to -5.00 |
| megakaryoc | 80 | +0.35 to +0.37 | +0.00 to +0.00 |
| monocyte | 80 | -1.42 to -1.41 | +0.00 to +0.00 |
| neutrophil | 80 | -2.80 to -2.79 | +0.00 to +0.00 |
| root | 1 | +0.00 to +0.00 | +0.00 to +0.00 |

#### Olsson

| Population | Cells | Width delta (px) | Cross-label overlap delta (pp) |
|---|---:|---|---|
| Eryth | 13 | -0.16 to -0.16 | +0.00 to +0.00 |
| Gran | 59 | -28.46 to -28.46 | -15.25 to -15.25 |
| HSPC-1 | 59 | -10.76 to -10.75 | -5.08 to -5.08 |
| HSPC-2 | 63 | -36.27 to -36.23 | +0.00 to +0.00 |
| MDP | 33 | -22.07 to -22.04 | -6.06 to -6.06 |
| Meg | 18 | -3.24 to -3.24 | +0.00 to +0.00 |
| Mono | 95 | -74.75 to -74.64 | +4.21 to +4.21 |
| Multi-Lin | 28 | -9.38 to -9.37 | +3.57 to +3.57 |
| Myelocyte | 14 | -1.50 to -1.37 | -28.57 to -28.57 |

## Interpretation and use

For exploratory visualization, choosing explicit `graph_mode="union"` can be
reasonable when its less crowded view serves the stated use. It is a different
learned target, not a harmless display-only force; record that choice. Mutual
remains the reference default, not because visual usability is irrelevant but
because no universal winner or biological superiority has been demonstrated.

The fixed screen proxies do not test human readability, raster occlusion, alpha
compositing, marker clipping, branch/marker continuity or pseudotime. Width can
grow by mixing unrelated populations. Raw disk views depend on centering; a
different root isometry can change visibility while preserving intrinsic
distances. Marker-size sensitivity and blinded user assessment remain untested.
Three seeds are small jitter around a common spectral layout, not independent
biological replication or arbitrary-initialization robustness. All fits have
a fixed 500-update budget, not demonstrated convergence.

## Reusable diagnostics and tests

- `src/sc_poincare/visibility.py`: read-only display/intrinsic diagnostics,
  matched-input comparison and exact-size disk plotting.
- `examples/evaluate_graph_visibility.py`: post-process saved pairs, audit
  input/coordinate hashes, preserve failed cases and plot predetermined seed 0.
- New tests cover known geometric widths/distances, duplicate/one-label cases,
  scaling and non-mutation, undefineds, exact 600px axes/4px markers, unsafe paths,
  altered coordinates, duplicate pairs, representative-seed policy and failures.
- Full local suite: **295 passed**, with two existing exact-zero-target warnings.
- Separate Torch-free installed-wheel environment: **137 passed**; installed
  visibility-module smoke import passed outside the source tree.
- `git diff --check` passed. Automated checks are defined in the
  [Actions workflow](../.github/workflows/tests.yml).

## Local latest results

At execution time, `data/graph_diagnostics_integration/` held the combined
six-case paired view and fixed-size figures. That directory is intentionally
reused by subsequent studies, including the [MU scale-up](mu-scaleup-study.md),
rather than preserving every old run tree. The numerical tables above retain
this study's evidence. The overview was resized; individual PNGs used the scoring
canvas dimensions. No raw CSVs, source snapshots or dense target caches are
published or added to Git. Original data licenses apply.

## Original CSV checksums for the four added cases

| Case | SHA-256 |
|---|---|
| ToggleSwitch | `25769c7044c1411a043ce00c41177aaf1e2091a950106caaa5976d7649e94fc1` |
| MyeloidProgenitors | `2329fb44f0a8501dc54454046b76355d36bc3a427a0483f4404aa6430ac7ff25` |
| krumsiek11_blobs | `cf7ad3594f622dda2613d582e7aa4f00530edd9b18eafeb93d78565125ddb647` |
| Olsson | `68ae49fefd96b42ac57ee59ac51c49e90827477ea07a08da0fe41af872d395f8` |
