# Landmark reference check

`sc_poincare.landmark` compares one declared approximation with exact mutual-graph
stress. It does not change `prepare_graph_stress`, `fit_graph_stress`, the
public MU default, or the checkpoint source hashes of `experimental.py`,
`_proximity.py`, `_fitting.py` and `_disk.py`.

## What is frozen before the check

`freeze_comparison` binds the feature hash, `GraphStressConfig`, landmark count,
seeds, primary metric, sensitivity metrics and a positive `max_dense_bytes`.
`max_epochs` must be 0. Labels must already be excluded. Known development
identifiers `paul`, `moignard`, `moignard2015`, `pancreas` and `nestorowa`
cannot be declared `unseen_biological`. Any unseen role can be declared for
another identifier and is still refused by `compare_landmark_reference`.
Refusal is not a certificate that some other name is a blind holdout.

## What the check measures

On the exact classical-scaling initializer, with no Adam step:

1. Independent landmark shortest paths against the exact target columns, both
   scaled by the exact nonself geodesic mean.
2. Triangle completion `min_l d(i,l)+d(l,j)` against the exact geodesic. This
   is an upper bound. Its own nonself mean is compared with the exact mean.
3. Landmark MDS, after the same maximum-radius scaling, against exact classical
   scaling through orthogonal Procrustes, including reflection.
4. When requested, both-endpoint gradients of the unchanged exact objective on
   the completed target, and of the landmark-column objective. The column
   objective keeps the exact local cross-entropy and the `N*(N-1)` denominator.
   Pairs that do not touch a landmark contribute nothing. When every cell is a
   landmark, both gradients are the exact objective's gradient. Chunk size must
   not change that column gradient.

`fits_executed` is 0 and `biological_validation` is false. `write_reference_tree`
stores `report.json`, `arrays.npz` and `MANIFEST.json`, then rereads the
checksums. That reread is not an external backup and not a fitted-model
re-evaluation.

## What this does not establish

Synthetic recovery with every cell used as a landmark shows the reference path
agrees with exact MU. A smaller landmark set is a different target: the tests
require a strict triangle excess and a changed column gradient. Neither result
is the Paul (2,730) or Moignard (3,934) reference, and neither authorizes
disabling the dense guard, full Planaria, or a new default.

The completion still allocates an N-by-N matrix inside a run that already built
the exact target. It is not a matrix-free training method. A fitter that never
stores the dense target remains a separate experiment.
