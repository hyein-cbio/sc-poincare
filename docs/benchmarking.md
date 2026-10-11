# Small-data benchmark protocol

The current priority is quality and efficiency on small datasets, before
introducing an approximate large-N method. **Superiority over the authors'
PoincaréMaps implementation has not yet been established.** Improvements over
our own previous code are not evidence of superiority over that reference.

## Scope and provenance

- Begin with ToggleSwitch (200 cells) and MyeloidProgenitors (640 cells).
- Provide datasets locally. Do not vendor them or reference-code implementations
  into this MIT package; keep inputs, embeddings and reports under ignored `data/`.
- Do not inspect, copy, or adapt the Facebook/CPM implementation for development.
  An independent benchmark can consume externally supplied reference results,
  with their provenance and license respected.
- Labels are evaluation-only. Graph construction and fitting never use them.
- Compare learned coordinates with all display transforms disabled. Recentring
  is an isometry; jitter, radial repel and fan layouts are not benchmark inputs.

## Reproducible runs

Install `pip install -e '.[torch,benchmark]'` and run from the repository root with local CSV inputs:

```bash
python examples/benchmark_small.py data/myeloid/MyeloidProgenitors.csv \
  --n-neighbors 30 --sigma 2 --gamma 2 --epochs 200 --seeds 0 1 2 \
  --terminal-labels monocyte neutrophil erythrocyt megakaryoc \
  --output data/benchmarks/myeloid_spectral.json

python examples/benchmark_small.py data/toggleswitch/ToggleSwitch.csv \
  --n-neighbors 15 --sigma 1 --gamma 2 --epochs 200 --seeds 0 1 2 \
  --terminal-labels branch1 branch2 \
  --output data/benchmarks/toggle_spectral.json
```

Use `--initialization random` for an initialization ablation, not as a proxy
for the authors' implementation. `--sigma median` explicitly chooses a bandwidth
from the positive graph-edge distances; the JSON records the resolved numeric
sigma and target diagnostics. Changing sigma changes the target: do not compare
raw KL losses across bandwidths. Use a fixed input-distance/graph metric instead.
The actual initialization and fallback reason are recorded per run.
Try `--batch-size 1`, `256`, and the full cell
count to check invariance of our row-chunked exact optimization. Defaults limit
BLAS and PyTorch to one thread for reproducibility; record hardware, dependency
versions and Git revision separately.

The script saves JSON metrics and unmodified learned embeddings (apart from
optional root recentering) as NPZ files. Fixed epoch budgets disable early stopping
for these runs. The returned coordinates still select the best observed global
loss checkpoint, including the initial coordinates.

## What to measure

1. **Quality:** local graph-neighbor recall using hyperbolic distances, reported
   both at 15 retrieved cells and at each node's own graph degree. Fixed-15
   recall cannot reach 1 for degree >15; degree-matched recall avoids that cap.
   Also report terminal label-neighbor fractions and within-terminal distances. These are diagnostics,
   not a complete test of topology. Compact terminals alone do not demonstrate
   correct branch separation or preservation of global relationships.
2. **Optimization:** global symmetric KL at the returned coordinates, convergence
   curves, and variation across seeds. Compare losses only for identical targets
   and loss definitions, including temperature and numerical stabilization.
3. **Efficiency:** graph/RFA preparation time, fitting time, and whole-process
   peak resident memory. `ru_maxrss` includes imports, metric calculation and
   earlier runs in the process; use a fresh process per seed when comparing peaks.

For a reference comparison, match input checksums and preprocessing, record all
method settings, and report quality-versus-wall-time as well as equal-budget
results. An "epoch" can mean different numbers of optimizer updates in different
implementations. A lower loss on our own target, one attractive picture, or one
faster run is insufficient for a general performance claim. Add a global
structure metric (e.g. rank preservation of graph-geodesic distances), and
separate small-data conclusions from any later scalability claims.

## Exact-mode changes in this revision

- kNN edge lengths, rather than an eagerly stored all-pairs feature-distance matrix.
- Exact component-MST bridges using blocked Borůvka rounds. Tied shortest bridges
  may differ, but minimum total bridge length is preserved. Search time can still
  be quadratic per round; this is a memory improvement, not an atlas-scale claim.
- One sparse factorization for RFA with blocked identity-column solves. The dense
  inverse is retained, with no sparsification, truncation or landmark approximation.
- Row-chunked global KL and gradient accumulation. All cells remain in the softmax
  denominator, including contributions through the column coordinates. Clipping
  and updating happen once after all row gradients have been summed.
- Early stopping evaluates the same global objective at the checkpoint it stores.
  It starts after burn-in and tracks cumulative improvements relative to the
  last significant checkpoint, separately from the best observed coordinates.
  Stop reason is recorded, not equated with a convergence guarantee.
- Direct Euclidean separations avoid cancellation and the old constant-distance,
  zero-gradient zone for close points. Torch and NumPy use the same metric.
- Per-point clipping uses the Riemannian gradient norm (`max_grad_norm=2.5`), after
  row accumulation; non-finite gradients abort the update.
- Valid root-translated points are no longer clipped to the optimizer's safety
  radius. Such clipping can severely distort hyperbolic distances even though
  the coordinate change looks small in a Euclidean scatter plot.
- Bridges use their actual Gaussian weights, like local edges. Numerical
  disconnection raises; a mostly self-dominated RFA target warns. No feature
  rescaling or bandwidth change is automatic.
- A pre-allocation dense working-array budget, and a tiny-input-only exact NumPy
  fallback instead of silently optimizing a different conditional objective.

The tests compare bridges to a small dense Kruskal reference, RFA to a direct
matrix inverse, and chunked losses, gradients and trajectories to full-batch
calculations. These verify numerical correctness; biological superiority still
requires the reference benchmark described above.
