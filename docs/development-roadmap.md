# Development roadmap

This document separates implemented software, measured evidence and proposed
research. It does not authorize experiments, carry forward old runtime budgets,
or imply that local tests are remote CI or biological validation.

## Current public baseline

The operator selected **MU fixed** as the provisional AnnData default after the
[MU study](mu-scaleup-study.md). Local-radius is selectable, not automatically
chosen by dataset. `method="rfa"` preserves the corrected legacy defaults and
conditional-KL / union-only variants. This selection is not a claim of unseen
performance, biological equivalence, or hyperbolic-specific superiority.

- [Migration/compatibility](mu-default-migration.md): intentional default change,
  method-specific defaults and old-pipeline preservation.
- [MU API](graph-stress.md): target/optimizer definitions and unchanged bare-state recovery.
- [RFA objectives](objectives.md): legacy formulas and log-domain numerical policy.
- [MU results](mu-scaleup-study.md): small/full comparisons and exact resumed controls.
- [Astra adoption status](astra-adoption-status.md): completed, deferred and rejected work.
- [Benchmark/recovery protocol](benchmarking.md): resource, provenance and evaluation contracts.

## Completed software integration

- [x] Preserve the corrected RFA numerical path behind an explicit selector.
- [x] Integrate mutual-global / union-local stress, tangent Adam, fixed and
  local-radius teacher rules, and native Euclidean controls.
- [x] Retain exact all-column row chunks, gradient checks, best/current separation,
  source/target integrity and persisted Adam-state recovery.
- [x] Add graph fragmentation, local/population/global-feature and screen/intrinsic
  diagnostics; do not treat reconstructed tie choices as certified fitted edges.
- [x] Expose independent AnnData progress snapshots and cell-order-bound MU
  recovery without putting Torch optimizer tensors into H5AD uns.
- [x] Deprecate display jitter/fan/repel in one removable legacy module, retain numerical
  initialization jitter, and explicitly record the checkpoint source boundary.
- [x] Record method/graph semantics, effective parameters and raw/display provenance.
  Preserve old files; never fabricate freshness for missing provenance.
- [x] Configure [CPU-Torch / Torch-free / wheel CI](../.github/workflows/tests.yml)
  and local install/serialization tests. Consult workflow results for actual CI status.
- [x] Complete the bounded MU study and its separately budgeted optimizer-state
  continuation; keep initial resource stops as history, not failed geometry evidence.
- [x] Expose experimental scalar forest tau and bounded spectral controls without
  changing numerical defaults; record convergence/fallback and fixed probe losses.
- [x] Add optional per-gene reconstruction and predeclared-stage ordering diagnostics.
  Their software availability is not independent biological validation.
- [x] Audit upstream code/evidence before reapplication; see the
  [deduplicated Astra follow-up](astra-followup.md).

## P0 — Maintain a reproducible starting point

- [ ] Freeze new comparisons before fitting: input representation/IDs, primary
  and sensitivity metrics, own-objective selection and execution limits.
- [ ] Complete the specifically scoped external review of historical commit
  `494367a` if needed; a current uncommitted-diff review does not prove that older
  committed scope was reviewed. Preserve that distinction.
- [ ] Keep one latest small result tree with checksum/config manifests rather
  than source/cache duplicates. Long or unique jobs still need external backup
  and an actual restore/re-evaluation check.

## P1 — Independent biological validation

- [ ] Freeze input-only rules and test measured datasets not used in development.
  No labels enter fitting, initialization, bandwidth/condition selection.
- [ ] Inspect every rare/outlier population, marker continuity, branching order,
  root distances and pseudotime. Purity and graph scores are not sufficient.
- [ ] Compare arbitrary initialization policies and more seeds, with uncertainty
  estimates distinct from small jitter around one classical layout.
- [ ] Compare convergence/compute budgets, not only equal update counts. Retain
  population losses and native Euclidean successes as well as improvements.
- [ ] Check sensitivity to graph k/anchors, representation, bandwidth, marker size
  and declared display roots. Do not choose a favorable picture after fitting.

The earlier RFA conditional-KL work remains experimental. Its pending fixed-rule
corrected-self-inclusive comparison, longer Planaria runs and isolation checks
have not been satisfied merely by selecting MU as the new provisional default.

## P2 — Bound initialization and scale-up cost

- [x] For RFA, expose explicit spectral `tol` / `maxiter` controls with recorded
  convergence/residuals and seeded fallback; complete the fixed small quality probe.
- [ ] Extend initialization/convergence budgets on independent inputs before
  changing spectral defaults. Short-run gains and residuals are not equivalence.
- [ ] For MU, benchmark alternatives to dense classical scaling and dense global
  targets. Validate landmark/matrix-free/sparse approximations against the exact
  few-thousand-cell reference, including normalization and endpoint gradients.
- [ ] Treat full Planaria as a separate protocol. No exact large-data claim follows
  from the source branch's lost observations or this study's n<=3934 results.
- [ ] Do not disable the default memory guard to claim scalability. Measure process
  RSS and preparation, fitting, evaluation and checkpoint I/O separately.

## P3 — Research mechanisms, not dataset-specific patches

These are not default changes and require their own controlled experiments:

- [x] Expose scalar RFA forest time `P_tau=(I+tau L)^-1` distinctly from sigma;
  probe fixed .1/1/10 conditions without selecting a new default.
- [ ] Validate fixed multiscale mixtures or degree-aware rules separately.
- [ ] Hybrid graph/bridge and local/global trade-offs; do not optimize only a
  training-derived metric or maximize visual spread without semantic checks.
- [ ] Objective-versus-optimizer ablation, intrinsic adaptive updates, alternative
  coordinate charts/Lorentz geometry, curvature or dimension. Numerical range
  and biological quality are separate questions.
- [ ] Temperature/global-scale changes under a frozen input-only rule. The failed
  source row-temperature alternative is not promoted because one score improved.

## P4 — Software/publication readiness

- [ ] Keep policy/mechanism changes separate from numerical fixes and data parsing.
- [ ] Review privacy, licenses, installation, documentation and reproducibility
  before external publication; package MIT does not relicense input artifacts.
- [ ] Check any publication eligibility against current rules when submitting.
- [ ] Obtain explicit approval for commits/pushes and new expensive experiments.
  Earlier approval does not silently authorize future jobs.

The next algorithm-scale priority is an independently validated approximation
against the exact MU reference, not unlimited brute-force N-by-N storage.
The next scientific priority is independent biology/initialization validation.
