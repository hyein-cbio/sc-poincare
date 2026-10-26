# Small-data benchmark protocol

The historical `benchmark_small.py` and objective-ablation drivers deliberately
remain explicit RFA reference tools; they do not silently switch to the public
AnnData MU default. Use the MU drivers/migration guide for that method. This
preserves their historical numerical contract and prevents mixed objectives.

The current priority is quality and efficiency on small datasets, before
introducing an approximate large-N method. **Superiority over the authors'
PoincaréMaps implementation has not yet been established.** Improvements over
our own previous code are not evidence of superiority over that reference.

The source branch's benchmark documentation is retained in the
[historical snapshot](history/graph-stress-remaining-tasks/benchmarking.md);
its [embedding study](embedding-study.md) and
[follow-up](embedding-followup.md) are historical evidence, not a revision of
the protocols or results below.

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

## Post-fit population and input-neighbor diagnostics

`sc_poincare.evaluation.evaluate_neighborhoods` is an observational evaluator,
not a new objective or a checkpoint-selection rule. Supply aligned cells in
three inputs: the numeric feature representation actually fitted (without
labels or IDs), raw N-by-2 disk coordinates, and nonmissing population labels.
It uses Euclidean feature distances and hyperbolic embedding distances; no
normalization, PCA, sampling, root selection or display transform is implicit.

For each k, exclude the cell itself and retrieve k neighbors among all other
cells. Exact ties are broken by ascending original row index. Report:

- **Overall purity:** mean same-label neighbor fraction over all cells.
- **Population purity:** the same fraction averaged within each population.
- **Macro purity:** an unweighted mean of population purities, including singletons.
- **Input-neighbor recall:** intersection of feature and disk neighbor sets / k,
  reported per cell, per population and overall. This is not recall of the
  stitched training graph and not the paper's Qlocal/Qglobal implementation.
- **Counts and purity ceilings:** a population of size m can reach at most
  min(m-1,k)/k. Do not call small-population mixing a regression without checking
  this ceiling, label granularity and expected continuity of differentiation.

The evaluator defaults to k=15/30 and requires 1 <= k < N. The small CSV benchmark
now adds this summary under each record's `neighborhood_evaluation`; for tiny
inputs it explicitly uses the distinct values min(15,N-1) and min(30,N-1),
recorded as the actual `ks`. Evaluation runs after fitting on its returned raw
coordinates and does not influence the optimizer or checkpoint selection.

To evaluate an existing NPZ (arrays `features`, `coordinates`, `labels`, and
optionally aligned `before_coordinates`), run:

```bash
python examples/evaluate_neighborhoods.py data/run.npz \
  --ks 15 30 --output data/neighborhood-evaluation
```

The output directory must be new. It receives `summary.json`, `manifest.json`,
`cell_scores.npz` (per-cell purity and input recall in input row order), and
`diagnostics_k*.png`. The manifest stores content hashes, versions, actual
imported evaluator source hash and CLI settings, not source snapshots.
Figures show label colors, per-cell input recall on a fixed 0–1 scale, and
population purity with counts, ceilings, overall and macro means. With before
coordinates, identical axes, labels and point sizes are used; no alignment is
performed. Coordinates/features themselves are not copied into the output.

### Checked before/after comparison and global safeguard

Pass a separate `--baseline baseline.npz` to compare metrics, not just pictures.
Both files must contain unique string `cell_ids` honestly describing each row's
identity. Exact feature/label/order and evaluation-policy matching is required;
no automatic reordering, subsetting or label remapping is performed. The CLI
prefers `raw_coordinates` if present, so benchmark display coordinates are not
accidentally scored. It writes `comparison.json` with after-minus-before deltas
for overall/macro purity, input recall and every population, and plots paired
population bars. These comparisons never select checkpoints or a winning method.
The Python equivalent is `compare_neighborhoods(before_report, after_report)`.
An embedded `before_coordinates` array without these identity checks remains an
unchecked visual panel, explicitly marked in the manifest, not a metric comparison.

The summary also includes `global_feature_rank`: choose up to 128 anchor cells
without labels (seed 714 by default), correlate each anchor's Euclidean feature
distances to all other cells with its hyperbolic distances using Spearman ranks,
and average the defined per-anchor correlations. The anchor indices, seed and
valid/undefined counts are explicit. Constant rows are undefined, not zero;
comparison deltas are null if either result has any undefined anchor. Change
`--global-anchors`/`--global-seed` only as a separately declared sensitivity check.
This measures direct feature-distance rank, not developmental order, graph
geodesics or an independent biological target. Labels do not choose anchors.

Use one latest report/figure directory for small development checks, rather than
accumulating full source copies and per-step records. Keep input artifacts or a
deterministic reconstruction available separately when exact reruns matter.
The [first graph-topology experiment](algorithm-experiment-01.md) froze sigma,
starts, inputs and evaluation before its explicitly authorized 12-fit run. Its
[results](graph-topology-results.md) retain unfavorable neighborhood changes.

For new matched NPZ inputs (`features`, `labels`, unique string `cell_ids`), the
paired driver resolves sigma once from mutual edges and creates the same mutual
spectral initial array for both graph modes per seed:

```bash
python examples/benchmark_graph_modes.py data/input1.npz data/input2.npz \
  --seeds 0 1 2 --epochs 500 --neighbors 15 --ks 15 30 \
  --threads 1 --wall-seconds 600 --output data/new-graph-comparison
```

The new output directory holds one shared input per dataset, paired coordinates
and losses per seed, a compact atomic summary (including failed/timeout state),
and figures for the first declared seed only. No labels enter graph/initializer/
fitting, no dense target caches are saved, and no winner is selected. A Unix
interval timer bounds the entire run after setup, including evaluation/figures.
The ordinary `benchmark_small.py --graph-mode union` flag is available too, but
separate runs with graph-specific median sigmas or spectral starts are not this
controlled causal comparison. Training-graph recall or raw KL across different
targets cannot stand in for common-input accuracy.

Distance computation is exact and row-blocked, with quadratic distance work
and O(block_size*N) distance/sort workspace; this is not an atlas-scale
approximation. Inputs must share cell order; shape checks cannot verify cell
identity. Use feature/input identity checks in comparisons. Do not interpret
training-input preservation as held-out prediction, confidence intervals or
biological accuracy. Known development data and short fixed-budget regression
fits remain development evidence. Freeze evaluation conditions before changing
algorithms and retain rare-population results, not just favorable averages.

### Screen visibility versus intrinsic geometry

For an already evaluated embedding, `evaluate_visibility(report)` from
`sc_poincare.visibility` uses a fixed disk diameter of 600 pixels, marker diameter
4 pixels, macro population 90%-radius width, nearest-center distances, strict
marker-center overlap and occupied 10-pixel grid bins. It also reports intrinsic
nearest-other-population distance. Display spread, less center overlap and
biological separation are different endpoints; single-label cross-population
metrics are undefined (null), not zero. See the
[fixed visibility protocol and additional-case results](graph-visibility-study.md).

The post-processing CLI accepts directories from `benchmark_graph_modes.py`:

```bash
python examples/evaluate_graph_visibility.py data/paired-study \
  --output data/new-visibility-report
```

It audits saved feature/label/ID/coordinate hashes, never fits or chooses new
conditions, reports all saved seeds, and plots the predetermined seed 0. The
individual PNG has exact 600-by-600 pixel disk axes at native 100 dpi with 4-pixel
circular markers. The overview is a resized thumbnail, not the scoring canvas.
Marker-center overlap is not raster occlusion: clipping, antialiasing, alpha,
point order, screen scaling and human interpretation are not scored. Broader
population width can represent mixing; occupied-bin counts depend on cell count,
orientation and grid. Compare only matched inputs with the same display policy.
Failed preparation cases remain in the summary/overview; never silently change
sigma just to turn a failed fixed-condition test into a successful fit.

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

## Long-run preparation and recovery

1. Verify `pwd`, `git rev-parse --show-toplevel`, `git status -sb`, the recent
   commits and `git worktree list`. Confirm the Python import source with an
   editable install or `PYTHONPATH=src`. A directory name is not proof that a
   particular revision or environment is being used.
2. Prefer the generic [objective comparison driver](objective-validation.md#reproduce-comparisons-on-your-own-inputs)
   for new ablations. Freeze the protocol, input hashes, feature/PCA choices,
   graph, initialization, seeds, update counts, primary evaluation and thread
   policy before fitting. Save the source revision, local diff and runtime
   dependency versions; use a fixed source snapshot for long jobs.
3. Reassess available RAM, runtime and worker limits for the current machine.
   Agree on limits before execution. The dense-array budget is not an RSS cap;
   monitor process RSS and system memory/swap separately. Past deadlines and
   successful runs do not authorize new jobs or establish their safety.
4. Keep inputs, references, raw current/best checkpoints, loss traces and partial
   or failed results under ignored `data/`. Use new output paths instead of
   overwriting previous runs. Do not restart an old controller until its source,
   inputs, scheduling and output behavior have been checked.
5. Inspect archived JSON/checkpoint metadata for obsolete absolute paths after
   moving repositories. Preserve historical provenance; add a relocation mapping
   or root-relative resolution in the new consumer rather than replacing every
   historical path. A fresh clone does not include ignored artifacts.
6. Follow the [low-level progress and warm-start contract](../README.md#long-fit-progress-and-warm-starts).
   Warm starts use raw training coordinates and start a new run, without restoring
   burn-in position, patience or best-checkpoint history. A constant post-burn-in
   RSGD update can continue statelessly from the same raw checkpoint only with a
   matched target and update policy; this is not general optimizer-state recovery.
   Record the original/continuation mapping, durable checkpoint, recomputed
   updates, thread changes and the best checkpoint across stages separately.
7. Report improvements, failures, unexecuted checks and selection bias together.
   Keep seed consistency separate from accuracy and do not infer global or
   biological improvement from a decreasing training loss alone.

## Historical resource observations

These measurements came from the earlier full Planaria scale-stability study
on a machine with 48 GiB RAM and 15 logical CPUs. The fitting runs used the former
probability-floor policy, not the current log-domain or conditional objective.
They are sizing context, not current performance guarantees or equal-compute
reference comparisons. Archived evidence lives under ignored
`data/scale_stability/` (`REPORT.md`, `final_audit.json`, source/harness snapshots
and `runtime_environment.json`). The runtime-environment record describes the
actual computation; a manifest writer's Python version alone does not.

| Observation | Historical measurement |
|---|---:|
| Full Planaria graph, 21,612 cells | about 3.9 s |
| RFA build plus cache write | about 10.4 s |
| One float64 target matrix | 3.48 GiB |
| Seed0, 1,000 fitting updates, 12 threads | about 2 h 37 min |
| Seed0 peak process RSS | about 5.10 GiB |
| Full-data fitting peak RSS range, including additional experiments | about 4.8–5.5 GiB |
| Maximum RFA row-mass error | 4.44e-16 |
| Sampled inverse residual (8 columns) | 1.33e-15 |

The sampled residual does not independently certify every inverse entry.
System-wide swap grew to about 3.8–4.3 GiB; other applications and OS effects
were not separated. Absence of OOM or a resource-limit stop is not evidence of
absence of memory pressure. That study used an 8 GiB per-worker RSS limit, a
3 GiB available-memory reserve and an explicit 16 GiB dense-array budget; those
were run-specific policies, not recommended defaults. Additional two-worker
runs with 6 threads each are not directly comparable to serial throughput.

A sigma2 control took about 913 s to reach its initial checkpoint. A separate
spectral probe with `tol=1e-4, maxiter=1000` took about .387 s with maximum eigen
residual about 9.24e-6; `tol=1e-6, maxiter=1000` failed to converge after about
4 s. These bounded settings were **not applied to the main initializer**.
Initial-distance structure, seed stability and final quality still need checking
before adopting them; a faster initialization is not necessarily equivalent.
