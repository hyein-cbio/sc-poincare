# Astra follow-up: deduplicated adoption and bounded controls

## Reference and scope

A fresh fetch still resolves `feat/experimental-graph-stress` to
`3489e586caaa5b0408a0751220d6349426457bfe`; the starting main revision is
`843138f7a65cb5beccb9efe7a2f2a7d796758d2b`. No new upstream implementation was
silently merged. `experimental.py` and `examples/run_graph_stress.py` were already
byte-identical to that branch, so they were not copied or implemented again.

The branch's working MU model and its roadmap are different categories. This
follow-up turns selected roadmap mechanisms into explicit research controls and
adds missing observational resolution; it does not promote every suggestion to
a default or claim independent biological validation.

## Delta against the existing integration

| Astra item | Before this follow-up | Now |
|---|---|---|
| Mutual-global/union-local targets, fixed/local radius, tangent Adam, exact row chunks | Already implemented and tested | Unchanged; no duplicate implementation |
| Adam recovery, identity-bound AnnData wrapper, native Euclidean controls | Already implemented; full controls completed | Unchanged |
| Population, expression-average, visibility and direct-feature diagnostics | Already implemented | Reused, not replaced |
| Input-only forest propagation time tau | Proposed research only | Explicit experimental RFA `forest_tau`; default 1 unchanged |
| Spectral tol/maxiter and truthful fallback | Proposed, with historical timing observation | Explicit RFA controls, solver/residual/convergence metadata and seeded warned fallback |
| Gene-level losses hidden by averages | Not exported by the evaluator | Opt-in per-gene output with indices, optional names and zero-variance undefineds |
| Developmental-order diagnostics | Root-distance pseudotime existed, but no checked stage-order evaluator | Read-only evaluator requiring a caller-declared order; no inferred biology/root search |
| Preserved Astra aggregate evidence | Linked but not shipped locally | Immutable aggregate JSON retained separately from current reruns |
| Local research/environment exclusions | Base ignores existed | Upstream `.venv-*/` and `/work/` exclusions added |

## RFA controls, not default changes

```python
import sc_poincare as dr

dr.tl.poincare_map(
    adata, method="rfa", forest_tau=2.,
    spectral_tol=1e-4, spectral_maxiter=1000,
)
```

These values illustrate the API, not a recommended universal recipe.

- `forest_tau` forms `(I + tau L)^-1`; it does not rescale features, change sigma,
  insert bridge floors or select a graph. It must be finite and positive.
- `forest_tau=1` keeps the previous target and numeric branch exactly.
- `spectral_tol=0, spectral_maxiter=None` keeps the old dense <=2000 / sparse
  >2000 policy and the previous default ARPACK call. Nondefaults explicitly
  request ARPACK, even on smaller graphs. They cannot silently apply to a
  supplied coordinate warm start or a random start without connectivities.
- Inspect `initialization`, `initialization_reason` and
  `initialization_diagnostics`: requested settings, actual solver, convergence,
  returned/checked eigenpair counts and residuals are distinct facts. Solver
  convergence is not a guarantee of useful coordinates or biological quality.
- Non-convergence warns and records partial eigenpairs before a seeded fallback;
  partial eigenvectors are not silently substituted for a converged layout.
- Controls are RFA-only in AnnData; passing them under MU raises migration advice.
  The low-level `forest_accessibility` and `fit_disk_embedding` APIs expose the
  same controls. H5AD stores plain metadata, not optimizer state.
- Maxiter bounds ARPACK iterations, not wall time or O(N^2) RFA target memory.
  Dense guards remain enabled. No multiscale/degree-aware automatic tau rule is
  introduced.

### Source boundary and exact default parity

New RFA controls change the whole-file hashes of `_proximity.py` and
`_fitting.py`. MU checkpoint guards hash those files even though the default MU
calculations do not use the new controls. Pre-follow-up states therefore require
their original source/environment; hashes are not rewritten or whitelisted.
New-state resume remains checked. Low-level MU code and initialization jitter
0.0002 are unchanged.

A frozen pre-change comparison matched default RFA targets, five-update
coordinates/loss and a 2001-cell sparse spectral initialization exactly. Both MU
bandwidth rules also matched initial/best coordinates and loss traces exactly
on the fixed small input. Source-version compatibility and numerical default
parity are reported separately.

## Granular observational checks

```python
from sc_poincare.evaluation import evaluate_expression_neighborhoods, evaluate_ordering

report = evaluate_expression_neighborhoods(
    expression, raw_coordinates, per_gene=True,
    gene_names=gene_names, cell_ids=cell_ids,
)
order = evaluate_ordering(
    pseudotime, stage_labels, stage_order=["early", "middle", "late"],
    cell_ids=cell_ids,
)
```

Gene indices always refer to original expression columns. Duplicate supplied
names are retained by index; unnamed columns are not given fabricated marker
names. Zero-variance genes have explicit null scores rather than disappearing.
The default aggregate expression output is unchanged. Checksums record the
expression/coordinates and optional names/IDs; input-order honesty remains the
caller's responsibility.

Ordering is a descriptive Spearman association with an explicitly supplied
stage sequence. Missing/unknown stages and duplicate equivalent order categories
are rejected. Constant order or pseudotime is undefined, not zero. The function
never guesses an ordering, chooses roots from favorable results, or enters the
fitter. Marker continuity, lineage correctness and independent biology remain
separate validation work.

## Fixed bounded probes

The protocol was frozen before execution: two one-thread workers, 360-second
stage budget, 45 seconds per owned child, sampled aggregate RSS ceiling 6 GiB,
minimum available memory 2 GiB. All **30 jobs completed in 62.1 seconds**.

- **Tau:** Moignard/Paul existing 384-cell development inputs; tau .1/1/10;
  seeds 0/1/2; 100 RFA updates, fixed shared MU-classical starts, median sigma,
  gamma 2, lr .01, burn-in 5, exact row chunks 128. No condition retuned.
- **Spectral:** 2001 cells sampled from each full input with input-only seed 714,
  to exercise the >2000 sparse path; seeds 0/1; defaults versus tol 1e-4 or
  1e-6 with maxiter 1000; 20 RFA updates. All twelve eigensolvers converged.
- Input/ID/source hashes, all population outcomes, residuals, initial/final
  diagnostics and failures/time-stop slots are retained. No labels enter fitting.
- These are new short-budget controls, not additions to the original 76-condition
  study or convergence checks. Fitting timings include initialization and updates;
  they are not isolated eigensolver latency benchmarks. As-run source hashes
  are retained; a subsequent guard for oversized nonrepresentable tolerances is
  recorded separately, with default parity rechecked on the final source.

Tau .1 improved aggregate small-input macro purity and expression MSE relative
to tau 1 in these 100-update runs, while tau 10 was worse. This is not a default
selection: Moignard PS purity lost .0434 in seed 2, and Paul 10GMP/6Ery retained
losses under tau .1. Improvements over another short-run target also do not mean
that the input-neighbor recall of the shared initialization was preserved.

Bounded spectral settings were not universally better. Moignard's tol 1e-4
macro purity was slightly lower in both seeds. Paul improved in aggregate under
that setting, but 19Lymph lost .1014 purity in seed 1. Default residuals were
about 1.7e-15 to 1.8e-14; tol 1e-4 residuals were about 7.0e-6 to 7.8e-5.
Tolerance therefore changes the starting structure/trajectory, not just speed.
Defaults remain unchanged.

### Post-hoc gene reports, no refitting

All ten completed full-study models were reevaluated at gene level. Aggregate
MSE matched saved scores within 5e-15, and no fitting was repeated. Against the
shared-init RFA baseline, fixed/local-radius Poincare MU had **419/410 worse genes
out of 1000** in Paul and **9/8 worse genes out of 42** in Moignard, despite better
aggregate MSE. These counts describe reconstruction errors, not significance or
validated marker behavior. Input artifacts lack gene names, so reports retain
column indices rather than guessing genes.

Astra's reported real-data developmental rank was not recomputed: the aggregate
evidence does not provide a complete reproducible stage-order/root recipe. The
new evaluator was tested with declared synthetic orders; no real-data stage
mapping was guessed to reproduce a favorable number.

## Upstream evidence and what is still not applied

[astra-development-results.json](astra-development-results.json) is a verbatim
copy of the upstream aggregate at pinned 3489e58 (SHA256
`15a31060fef60b946a6bf7fbf756675d2c1e4ee28be856312e3770971decc70f`).
Its 18 rows are **two published-reference rows plus sixteen standard MU rows**;
the separate rejected-temperature entries summarize two computationally
completed but quality-failing controls. They are not eighteen additional runs
performed here. The original narrative reports 18 fits including those two
rejected controls. Neither source rows nor published coordinates are independent
holdout evidence or substitutes for upstream cell-ID verification.

Still deferred/rejected, with reasons:

- **Multiscale/degree-aware tau rules:** no frozen validated weighting rule;
  exposed scalar tau is not such a rule.
- **Independent data, marker/branch/pseudotime validation:** no predeclared unseen
  biological protocol is established by the known development inputs.
- **Arbitrary MU initialization, more seeds, causal optimizer/objective ablation:**
  existing classical+jitter seeds and fixed updates do not answer those questions.
  Coordinate-only MU warm starts remain a new condition needing a distinct contract.
- **Sparse/landmark/matrix-free scaling and full Planaria:** approximate targets,
  normalization, gradients and initialization need reference checks; do not disable
  the memory guard or infer results from lost runs.
- **Lorentz/intrinsic adaptive optimizers, curvature/dimension/global scale:**
  numerical range and biological quality require separate controlled work.
- **Failed row-temperature rule:** its population/expression counterexamples
  remain evidence against promotion, not an invitation to retune until it passes.
- **Durable off-machine backups:** no external destination/access was supplied;
  local atomic checkpoints and a fresh-directory restore do not prove persistence.
- **Historical 494367a-specific review and publication eligibility:** current
  tests/reviews do not retroactively cover that scope or establish release criteria.

Local verification: **505 passed** with two existing exact-zero-target warnings;
Torch-free checks **293 passed / 54 optional skips**. Installed-wheel MU resume,
RFA controls/H5AD and gene/order diagnostics also passed. The scoped CodeRabbit
review completed with zero findings; later tolerance-guard coverage is included
in the final local suite. No new remote CI run is claimed for uncommitted work.

Local artifacts remain one latest tree under `data/graph_diagnostics_integration/`:
new compact probes/gene reports supplement the preserved study, without copies of
raw inputs or dense targets. No new commit/push is implied by this follow-up.
