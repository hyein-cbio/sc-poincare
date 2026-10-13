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
