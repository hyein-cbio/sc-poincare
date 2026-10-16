# First isolated algorithm experiment: mutual versus union neighborhoods

**Status: implemented opt-in and executed after explicit user approval.**
The 12-fit, 500-update bounded probe completed. See the
[results and test report](graph-topology-results.md). Mutual remains the default;
this result does not authorize a default change, further experiments, a commit
or a push.

## Why this is the first change

The development study on `origin/feat/experimental-graph-stress` identified
substantial fragmentation before mutual-kNN stitching. Our 384-cell regression
subsets also have 13 components/11 isolated cells (Moignard) and 40/34 (Paul)
before stitching. These subset counts do not describe the full datasets.

Test the input graph before redesigning the objective or optimizer. Mutual kNN
keeps i-j only when each selects the other; union kNN keeps it when either does.
Union may reduce bridge dependence, but can also join unrelated populations or
increase hub influence. More neighbors can improve training-related metrics
without improving biology. There is no presumption that union is better.

## One experimental axis

The API exposes `graph_mode='mutual'` (default) or experimental `'union'`.
The completed controlled comparison uses:

- **A: corrected main** — mutual kNN + existing exact component-MST stitching.
- **B: candidate** — union kNN + the same stitching rule when needed.

Retain actual Gaussian weights on every edge, RFA `(I+L)^-1`, self-inclusive
`symmetric_kl`, float64 disk geometry, Riemannian updates, clipping and checkpoint
selection by each fit's own training objective. No graph stress, conditional KL,
local-radius bandwidth, Adam, negative sampling, display forces or dataset-name
branches enter this experiment. Changed degree/total affinity is part of the
graph change; do not silently renormalize it away.

## Freeze before fitting

Initial bounded development probe (not a biological validation study):

| Item | Fixed choice |
|---|---|
| Data | Same 384-cell Moignard/Paul subsets used in diagnostics |
| Subset rule | Uniform, label-independent seed 714; original row order |
| Features | Moignard's supplied 42 features; Paul subset full-SVD PCA20 after label exclusion |
| Cell identity | Original CSV row IDs, pinned input content and fitted-feature hashes |
| Graph | k=15 in A and B; only mutual/union membership changes |
| Sigma | Resolve positive-edge median **once on A**, then use that identical numeric sigma in A and B |
| Initialization | Generate from A's connectivity graph for each seed; pass the exact same initial array to A and B |
| Seeds | 0, 1, 2; paired comparisons, not confidence intervals |
| Fitting | gamma=2, learning_rate=.01, max_grad_norm=2.5, batch_size=128 |
| Schedule | 500 total updates, burn-in=5, patience=500; initial checkpoint eligible |
| Resources | One CPU thread; default 512 MiB dense preflight; no full Planaria |
| Evaluation | k=15 primary, k=30 sensitivity; 128 global feature-distance anchors, seed 714 |
| Display | Same cell order, colors, axes, size; no recentering, alignment, fan, repel or jitter |

The authorized run was limited to 12 fits (2 datasets × 2 graphs × 3 seeds),
not an open-ended search, with a 600-second whole-run Unix timer and one CPU
thread. All 12 completed in approximately 26 seconds. A 500-update
budget does not prove convergence; report initialization provenance, actual
updates, best epoch/loss if available, final stopping state and runtime. Abort on
invalid/nonfinite data, numerical graph disconnection or resource refusal; do
not silently relax the protocol or substitute a favorable partial run.

Shared starts and sigma are essential: allowing each graph to pick its own
median sigma or spectral start would confound this first causal comparison.
Do not compare raw KL values as accuracy across different RFA targets.

## Read the results without choosing a flattering view

- Compare on the **same frozen features**, not each method's own training graph.
  Primary: input-neighbor recall at k15. Report k30 and row-wise global
  feature-distance rank correlation, including undefined-anchor counts.
- Report overall and macro purity plus every population's purity/recall delta,
  counts and size ceilings. Flag negative deltas rather than filtering them out.
  Populations of size <=15 deserve particular attention, but larger groups and
  legitimate continuous transitions must also remain visible.
- Show paired seeds and per-dataset results, not only a pooled winning average.
  Fewer bridges, lower training loss or a prettier disk is not a promotion gate.
- A gain in local recall with lower global rank, macro purity or harmed populations
  is a trade-off to explain, not an unconditional success. Undefined global
  comparisons cannot pass a global safeguard. No automatic winner is selected.
- None of these metrics validates branch order, root pseudotime or marker
  continuity. Those checks and genuinely independent measured data with a
  frozen input-only rule are required before considering a default change.

## Keep the preparation lightweight

Use `evaluate_neighborhoods(..., cell_ids=...)` for both results and
`compare_neighborhoods(before, after)` for checked after-minus-before deltas.
The CLI `--baseline` compares NPZs with matching feature matrices, labels and
unique string cell IDs/order; it prefers `raw_coordinates` if present.

Keep one shared input/feature artifact if needed, paired raw coordinates and a
latest summary/figure set under ignored `data/`. Do not accumulate full source
copies, dense RFA caches, successful-run duplicates or per-step logs for this
small probe. Small manifests record content hashes, versions and evaluator
policy. Keep failed/partial outcomes in the summary. External recovery plans
are still needed for genuinely long or non-regenerable jobs.

## Preparatory work available now

- Graph component/isolation/bridge diagnostics without changing graph output.
- Overall/population/macro purity, size ceilings, and input recall at k15/30.
- Fixed-anchor global feature-distance rank diagnostic with explicit undefineds.
- Checked feature/label/cell-order matching and per-population comparison deltas.
- Matched before/after plots and compact CLI provenance; no automatic tuning.
- Synthetic regression coverage, wheel/import smoke checks, and GitHub Actions
  configuration for full CPU-Torch tests plus a separate Torch-free test path.
  See the [Actions configuration](../.github/workflows/tests.yml) for automated checks.
