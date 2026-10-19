# Experimental graph stress embeddings

The `sc_poincare.experimental` API combines a graph-distance target with a local
neighborhood distribution. It is a different method from RFA symmetric KL. The
operator has selected its fixed MU recipe as the provisional public
`tl.poincare_map` default; `method="rfa"` preserves the previous fitter/defaults.
See the [migration guide](mu-default-migration.md) for method-specific arguments.
The low-level numerical/checkpoint implementation described here is unchanged.

The [source-branch development study](https://github.com/hyein-cbio/sc-poincare/blob/3489e586caaa5b0408a0751220d6349426457bfe/docs/embedding-study.md) found limited improvements in
Moignard neighborhood diagnostics and terminal visibility. Paul retains
population-level losses, and the Euclidean controls also improve. This API
does not establish a general biological advantage or a hyperbolic-specific
benefit.

## Explicit feature input

Install `pip install -e '.[torch,benchmark]'`. Supply a finite numeric feature
matrix, with labels and identifiers removed before fitting. Feature selection,
normalization and PCA are caller decisions; the method does not choose them
from a dataset name.

```python
from sc_poincare.experimental import (
    GraphStressConfig,
    prepare_graph_stress,
    fit_graph_stress,
)

# features: cells x numeric features, with no labels or root information.
config = GraphStressConfig(local_bandwidth="local_radius")
setup = prepare_graph_stress(features, config=config)
result = fit_graph_stress(setup, epochs=1500, random_state=0)
coordinates = result.coordinates
```

`local_bandwidth="global"` selects the fixed-bandwidth development control.
`"local_radius"` selects the follow-up rule. Neither is an automatic
dataset-specific choice. A root can be applied afterward with a disk isometry;
it is not a fitting input. Display jitter, repulsion and fan layouts are not
part of this method or its evaluation.

The CSV example requires the caller to identify any label column and request
PCA explicitly:

```bash
python examples/run_graph_stress.py --input data/cells.csv \
  --csv-label-column -1 --pca 20 --local-bandwidth local_radius \
  --epochs 1500 --seed 0 --output data/new-graph-stress-run
```

The output must be a **new directory**. It contains `features.npy`, `run.json`,
`result.npz`, and an atomically replaced `checkpoint.pt` with matching state
and current/best coordinates. A resume uses the same input, preprocessing and
options, a new output directory, `--resume old-run/checkpoint.pt` and a larger
total `--epochs`. See `python examples/run_graph_stress.py --help` for options.
Paul's development comparison excluded its
last label column before full-SVD PCA20. Moignard used its 42 provided features.
These are preprocessing choices for those comparisons, not core branches.

## Targets and optimization

1. Construct a mutual kNN graph (k=15 in the study) and connect components with
   the core's exact component-MST rule. Let m be the median positive edge
   length. Transform each edge by `log1p(length / m)` **before** shortest paths.
   Normalize the nonself mean path distance to scale s=2, obtaining T.
2. Construct a separate connected union-kNN graph in the original feature
   distances. For `global`, use Gaussian width
   `max(q95_positive_edge, max_edge / 5)`. For `local_radius`, use
   `exp(-length_ij**2 / (2*r_i*r_j))`, where r_i is the kth nonself feature
   neighbor distance. Normalize edge weights within each row to obtain P.
3. The global term is the nonself mean of
   `((d_ij/s - T_ij/s) / max(T_ij/s, 0.01))**2`.
   The local term is `mean_i(-sum_j P_ij * log(Q_ij))`, with
   `Q_i = softmax(-d_i/gamma)` over **all nonself cells**.
   The study uses `gamma=0.1*2/3` and local weight 0.03.
4. Initialize with classical scaling of the transformed mutual graph target,
   scaled to maximum initial disk radius 0.2. The study adds normal jitter
   with standard deviation 0.0002. Optimize tangent coordinates u with Adam
   (`learning_rate=0.02*2/3`), mapping them to
   `x=tanh(norm(u)/2)*u/norm(u)` in the disk.

Local radii are bounded below by the median positive radius times 1e-6. The
diagnostics record zero radii and Gaussian underflow. A row with no positive
teacher mass is rejected; the implementation does not insert artificial edge
weights to make it pass. The rule can fail on extreme inputs. All-zero
distances and other degenerate initialization cases require attention rather
than an inferred biological layout.

The geometric distance is hyperbolic for `geometry="poincare"` and Euclidean
for `geometry="euclidean"`. Native Euclidean coordinates are not disk points.
The controls share the initial tangent array, targets, temperature and update
budget; they do not have exactly equal initial pairwise distances or equal
physical step lengths.

Row chunks retain every column and accumulate both endpoint contributions
before each update. They change fitting workspace, not the objective or its
quadratic target storage. No negative sampling is used.

## Teacher diagnostics

Preparation includes `teacher_entropy_mean` (nats), entropy and incoming-mass
quantiles at `teacher_diagnostic_quantiles=[0,.1,.5,.9,1]`, maximum asymmetry and
maximum row-normalization error. The keys are `teacher_entropy_quantiles`,
`teacher_incoming_mass_quantiles`, `teacher_max_asymmetry` and
`teacher_max_row_sum_error`. These describe the prepared teacher, not biological
balance; the weights, initializer, targets and fit are not changed.

Row-normalized symmetric affinities generally produce a reversible but
asymmetric teacher. Mean entropy lower-bounds local cross entropy; subtracting
it yields teacher-specific KL, not a common biological score across teachers.
The [latest upstream adoption report](astra-mechanism-adoption.md) records the
analytic checks and the 37-execution research evidence, including failed
promotion criteria and cancelled work. No candidate target rule was promoted.

## Results, progress and state recovery

The returned `coordinates` select the lowest **training** objective, including
the initialization. `current` is the last iterate. `loss_trace` includes epoch
0; `best_epoch`, `best_loss` and `current_loss` distinguish selection from
termination. An epoch limit is not evidence of convergence. Do not compare raw
loss values across different teachers or temperatures: teacher entropy and
gradient scale change.

`result.state` contains optimizer state and provenance. Resume with the same
prepared target, geometry, seed, chunk size, configuration and Torch version;
`epochs` is the **total** target
epoch count, not the number of additional steps:

```python
first = fit_graph_stress(setup, epochs=100, random_state=0)
continued = fit_graph_stress(
    setup, epochs=200, random_state=0, resume=first.state,
)
```

A `callback` receives `GraphStressProgress` at initialization, each
`checkpoint_every` interval and the last step. It includes raw current/best
coordinates and state. Persist a consistent state/coordinate pair using an
atomic write. Keep input, preprocessing, target, teacher, initialization,
configuration and source hashes with the checkpoint. A disk-recentered display
is not an optimizer checkpoint.

This Adam-state resume contract is separate from the existing RFA fitter's
coordinate-only warm start. The later [display API extraction](display-layout-deprecation.md)
changes the `_disk.py` source fingerprint; pre-extraction states are not silently
migrated, even though retained fitting/geometry calculations pass parity checks.
Caller checkpoint I/O and external storage remain
the caller's responsibility; a progress callback is not durable storage.

## Resource limits and reproducibility

Preparation stores dense all-pairs targets and uses a dense classical-scaling
initialization. The shared `max_dense_bytes` preflight defaults to 512 MiB,
estimated as four float64 N-by-N arrays. This is not a process RSS ceiling.
Chunking does not remove O(N²) storage or the initialization's computational
cost. **Do not apply the exact development configuration to full 21,612-cell
Planaria.** The study's approximate large-data paths were separate research
code, and their lost observations do not validate this API at that scale.

The package API contains no historical deadline, container path, dataset name,
reference coordinates, or label-based condition selection. The source snapshot
used for the 18 development fits is distinct from the maintained package port;
numerical port checks establish computation agreement, not new biological
validation. Review input hashes, package versions, graph construction, update
counts and stopping state when reproducing results.

Adding teacher metadata at upstream `9f369a9` changes `experimental.py` source,
metadata and checkpoint digests. Historical states still need their original
sources; numerical equality does not bypass strict resume checks. Separate
main before/after processes matched 262 array/tensor/scalar checks on full Paul
PCA20 over 20 updates in both geometries/bandwidths. These are software regression
checks, not new independent biological fits.

The earlier source branch reported that its port verification used the full Paul CSV,
explicit label exclusion and PCA20.
For both bandwidth rules, the first 20 updates matched the frozen research
implementation exactly in coordinates, initialization and loss trace (maximum
absolute difference 0.0); input, target and teacher hashes also matched. The
43 experimental tests cover an independent objective, both geometries, chunk
gradients, persisted Adam resume, input validation and CLI checkpoint recovery.
The source branch's complete suite passed 207 tests; one optional Scanpy test
was skipped because Scanpy was unavailable. These are historical source-branch
claims, not the current main suite count. In this integration, both bandwidth
rules also match the frozen branch exactly over 20 updates on the fixed Paul
subset (coordinates, initial arrays, current/best and loss trace). The
[bounded MU scale-up study](mu-scaleup-study.md) defines the frozen protocol and
records its completed results, including exact control continuations. Neither
set of numerical checks by itself proves biological superiority.

For long jobs, follow the [recovery protocol](benchmarking.md#long-run-preparation-and-recovery).
Archive externally accessible copies of inputs or their pinned download
manifest, fitted feature matrices, raw states, source/configuration hashes and
partial/failed runs. Verify an actual empty-directory restore and re-evaluation.
Local ignored folders and ZIPs inside the same temporary container do not
protect against environment replacement.
