# Graph construction, optimization and embedding quality

The subsequent [controlled mechanism follow-up](embedding-followup.md) is a
separate 37-execution record, including a user-stopped partial protocol. The
18-fit counts and results below refer to this earlier study.

This study compared independently learned embeddings with coordinates published
by Facebook Research's PoincareMaps release. It motivates the optional
[graph stress API](graph-stress.md). It does not replace the default RFA method
or establish general superiority over the authors' implementation.

## Main findings

- Graph construction matters before optimization. In the provided Moignard
  features, mutual k15 has 160 components and 145 isolated cells before 159
  MST bridges; union k15 is connected. The bridge-supported target can encode
  different relationships from a union graph. Union connectivity alone does
  not establish biological correctness.
- A lower loss or a higher rank correlation with the training graph can coexist
  with worse population separation and developmental ordering. Report direct
  feature, alternative graph, biological and visual diagnostics together.
- Intrinsic separation and screen visibility differ. A terminal population can
  occupy more display area while becoming closer to other populations in the
  embedding's own distance metric.
- A tangent-coordinate optimizer and a graph stress/local-distribution target
  provided limited improvements. Same-objective Euclidean controls also
  improved, so this is not evidence of a hyperbolic-specific biological benefit.
- Fixed and local-radius bandwidth rules have different population-level
  effects. An input-calibrated row-temperature alternative failed on both real
  datasets despite improving a training-related global metric.

## Completed measured-data comparison

After an execution environment was replaced, the recorded fixed rule was
independently reconstructed, not recovered byte-for-byte. The preserved
comparison contains **18 completed 1,500-update fits**: fixed MU and the
local-radius follow-up each have three Poincare seeds and one native Euclidean
control on Moignard and Paul (16 fits), plus two failed temperature controls.
MU denotes a mutual global graph and a union local graph.

The following K15 diagnostics use **all cells**. Population purity is the
fraction of a cell's 15 embedding neighbors with its supplied label. Macro
purity weights populations equally. Expression MSE compares each gene with
the mean expression of embedding neighbors, normalized by that gene's input
variance. It is an input-preservation diagnostic, not held-out prediction.
The [machine-readable summary](embedding-study-results.json) preserves exact
aggregate values, primary graph diagnostics and source revision/hash metadata;
it contains no raw cells, reference coordinates or private filesystem paths.

| Dataset and condition | Overall purity ↑ | Macro purity ↑ | All-gene MSE ↓ |
|---|---:|---:|---:|
| Moignard published coordinates | .617150 | .580113 | .396605 |
| Moignard fixed MU, Poincare seeds 0–2 | .625521–.626724 | .589328–.590831 | .395604–.395911 |
| Moignard local radius, Poincare seeds 0–2 | .625792–.627216 | .589855–.591205 | .393664–.394031 |
| Moignard local radius, Euclidean seed 0 | .625097 | .588772 | .395383 |
| Paul published coordinates | .594628 | .577434 | .908904 |
| Paul fixed MU, Poincare seeds 0–2 | .597265–.597534 | .563423–.565423 | .907531–.907746 |
| Paul local radius, Poincare seeds 0–2 | .597314–.598462 | .566578–.567330 | .906855–.907121 |
| Paul local radius, Euclidean seed 0 | .598046 | .567307 | .906745 |

Moignard contains 3,934 cells and 42 measured features. Paul contains 2,730 cells
and 1,000 provided genes; label exclusion preceded explicit full-SVD PCA20 in
fitting. Both preprocessing choices belong to the comparison, not to hidden
dataset-specific implementation branches.

## Visibility and biological counterexamples

Using the same selected input root, point size and a normalized 600-pixel disk
canvas, Moignard's 4SFG population display width (twice its 90th-percentile
radius about its display center) increased from 22.1 pixels to 77.2–78.7 pixels.
The proxy for overlap with other populations' marker centers decreased from
15.77% to 4.88–5.09%. This is a marker-center distance diagnostic, not a raster
measurement of occlusion. The median intrinsic distance to the closest other
population instead fell from 1.002 to .505–.514. The display improvement does
not establish increased intrinsic population separation.

Important counterexamples remain:

- Moignard local-radius K30 expression MSE is .397059–.398009, worse than the
  published .396260. Developmental stage rank from the fixed PS medoid is
  .317819–.326295 versus .333032. Other PS roots give mixed results; changing
  roots is not a universal improvement.
- Paul's local-radius macro purity is below the published result in every
  seed. K30 expression and purity also remain worse. Rare Eos, Neu and Lymph
  populations retain losses. Fine-label Ery mixing partly occurs within the
  same broad Ery prefix, but broad-prefix aggregation does not remove all
  population losses and is not independently verified lineage annotation.
- The input-calibrated temperature alternative yielded Paul overall/macro
  purity .585348/.548584 and Moignard expression MSE .410796. Those failed
  runs were retained rather than promoted on their global graph score.

Representative seed 0 was fixed in advance rather than selected for an
attractive plot. Comparisons matched cell order, colors, root, point size and
canvas; disk recentering and orthogonal alignment were the only display
transformations. Euclidean controls retain their own Euclidean axes.

## Evaluation and provenance limits

Graph diagnostics use 128 fixed evaluation rows (seed 714), graph k=15/20/30
and 15 retrieved neighbors. They are separate from all-cell population/MSE
diagnostics and are not a reproduction of the paper's Qlocal/Qglobal code.
Direct feature neighbors and union-graph neighbors can coincide, so they are
not independent successes. Sensitivity checks and individual population/gene
results must accompany aggregate gains.

The three seeds change a small jitter around the same classical-scaling
initialization. Their ranges are not confidence intervals or a test of
arbitrary-initialization robustness. Targets and initial tangent arrays match
between geometries; their initial pairwise distances are not exactly equal.
A fixed update budget does not establish convergence or separate geometry
expressiveness from optimization speed.

The authors' public CSV row count/order was used. Independent cell-ID matching,
author preprocessing and equivalent optimizer budgets were not verified.
Labels and plots influenced development decisions even though they never enter
the fitting function. These are known development datasets, not a blind holdout.
The stronger synthetic-data observations from the lost environment are not
merged into the preserved measured-data evidence.

Full Planaria cold-start validation remains incomplete. A lost warm-start
observation and re-downloading its 21,612-row inputs/reference are not a
validated large-data result. A separate 21,296-row artifact cannot be aligned
by truncation or mixed into quantitative comparisons.

## Reproduce and preserve

Public input and coordinate URLs are pinned to PoincareMaps commit
[`7ffa518cabe6e6424a875d480c4298bda3bd9678`](https://github.com/facebookresearch/PoincareMaps/tree/7ffa518cabe6e6424a875d480c4298bda3bd9678).
The comparison did not inspect, copy or adapt the authors' implementation.
Respect the upstream data/reference licenses; the package's MIT license does
not relicense those files. No raw data, reference coordinates, private session
exports or optimizer payloads are included in this repository.

The maintained package port, CLI and their tests are separate from the frozen
development source. Use the [explicit-input API](graph-stress.md), record the
source revision and package versions, and compare raw learned coordinates.
The raw research archive is retained separately with checksummed inputs,
states, source snapshots, failed runs and restoration instructions.

An execution environment replacement destroyed an earlier scratch directory.
Later runs used atomic checkpoints and externally stored snapshots. An actual
empty-directory archive extraction verified all 18 coordinate/Adam hashes;
an input-restoration supplement reconstructed 12 run folders from two fitted
feature matrices. A complete re-audit for one preserved candidate in each
dataset exactly matched its saved metrics. These are recovery checks, not
additional independent biological validation or a forced-container-replacement
test.
