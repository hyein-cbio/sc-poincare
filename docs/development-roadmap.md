# Development roadmap

This roadmap separates proposed work from validated behavior. It does not
authorize new experiments, resume previous jobs, or carry forward their runtime
budgets. Agree on a protocol and resource limits before starting long runs.

The [comparison with main](main-comparison.md) records which of the items below
already exist on `main` at `1e724d8` and are not reimplemented here. The
[frozen conditional-KL protocol](frozen-conditional-kl-protocol.md) is the
starting-point gate. [Further work](additional-work.md) is not part of this
checklist.

## Current baseline and evidence

The default remains corrected, self-inclusive `symmetric_kl`.
`conditional_symmetric_kl` is an opt-in experimental model, not merely a numerical
fix. The existing evidence is promising but does not establish unseen-data
performance or biological equivalence to the reference method.

Use these documents as the sources of truth rather than duplicating their
results or API details here:

- [README](../README.md): supported APIs, numerical safeguards, progress callbacks
  and warm starts.
- [Objective definitions](objectives.md): default/conditional formulas and the
  exact-zero numerical policy.
- [Objective validation](objective-validation.md): fixed-budget ablations,
  historical failed approaches, integration checks and selection limitations.
- [Benchmark protocol](benchmarking.md): evaluation, safe run preparation and
  historical resource observations.
- [Reference-condition search](reference-parity.md): historical dataset-specific
  recipes, separate from the default and conditional objectives.
- [Graph stress API](graph-stress.md): opt-in input-only mutual-global/union-local
  stress and local-distribution fitting, including local-radius bandwidth and
  Adam-state recovery. It leaves the RFA defaults unchanged.
- [Measured-data study](embedding-study.md): preserved 18-fit evidence, failed
  alternatives, native Euclidean controls, display/intrinsic distinctions and
  population-level counterexamples.

The [controlled mechanism follow-up](embedding-followup.md) completed the local
teacher protocol and closed the global-target protocol at the user's request.
No local-teacher candidate met promotion criteria; the partial global-target
study shows endpoint gains and Paul counterexamples, without completed
selection or transfer. Teacher entropy/incoming-mass diagnostics are now
observable in prepared metadata. The unresolved trade-offs and independent-data
gates below remain open; this does not validate forest time-scale mixtures.

## P0 — Establish a reproducible starting point

- [x] Establish the current repository, Python import source and artifact paths,
  then freeze a new comparison protocol using the
  [long-run preparation and recovery checklist](benchmarking.md#long-run-preparation-and-recovery).
  Recorded in the [frozen conditional-KL protocol](frozen-conditional-kl-protocol.md).
  Freezing the protocol does not start the fit.
- [ ] Complete the external review of the benchmark/docs changes in `494367a`
  when review access is available. The original review was rate-limited; local
  checks are not evidence of a completed external review. CodeRabbit CLI 0.9.0
  is installed here and is not authenticated, so this review was not run.

## P1 — Validate conditional KL before considering a default change

- [ ] Extend full Planaria from 300 to 1,000 updates under fixed conditions and
  multiple seeds. Check whether global quality degrades while training loss
  continues to decrease.
- [x] Compare against the **corrected self-inclusive default**, not only the
  historical probability-floor baseline. On ToggleSwitch the driver ran
  corrected `symmetric_kl` and `conditional_symmetric_kl` from the same
  spectral start. The historical floor was not the baseline. Losses are not
  compared across objectives. See [toggle-evaluation.md](toggle-evaluation.md).
  Full Planaria is a separate unchecked item.
- [x] Select checkpoints by the training objective, not evaluation scores.
  Preserve raw current/best coordinates, loss, gradient norm, radius and stop
  reason throughout the trajectory. The fitter returns the lowest training
  loss, including initialization. Each ToggleSwitch run stores 200
  pre-update Riemannian gradient norms and radius quantiles. The full
  Planaria trajectory is still unrun.
- [x] Check evaluation graphs at k=15/20/30, alternative anchors and direct
  feature-distance diagnostics. Declare primary and sensitivity checks before
  fitting rather than selecting favorable evaluations afterward. Primary was
  k=15 and anchor seed 714. Sensitivity was k=20, k=30, anchor seed 716, and
  feature distance. Symmetric KL stays slightly above conditional KL at every
  one of those checks. See [toggle-evaluation.md](toggle-evaluation.md).
  That does not replace the unrun full-Planaria check.
- [x] Inspect isolated, rare and outlier populations for artificial mixing or
  loss of terminal-state information when the RFA diagonal is discarded.
  On Paul, with the rare set fixed before fitting as labels of at most 31
  cells, 200-update conditional KL lowers all-cell K15 purity in all five
  seeds and does not lower the rare-label mean. 17Neu stays near 0.02 either
  way. See [rare-population-diagonal.md](rare-population-diagonal.md). The
  representation is the 1,000 gene columns, not the historical PCA20 hash.
- [x] Evaluate seed consistency separately from accuracy; plan more than three
  seeds and uncertainty estimates. The Paul diagonal and root-distance checks
  use seeds 0–4. Report the min–max range. It is not a confidence interval
  and not a Planaria result. See
  [rare-population-diagonal.md](rare-population-diagonal.md) and
  [root-pseudotime.md](root-pseudotime.md).
- [x] Evaluate biological markers, branch continuity, root distances and
  pseudotime. Good graph scores alone do not establish central branching.
  With the unique `root` cell fixed first, its disk radius stays near 0.72
  while the median cell radius is near 0.83. Ery label integers are perfectly
  anti-aligned with root distance in all five seeds. See
  [root-pseudotime.md](root-pseudotime.md).
- [x] Apply a frozen hyperparameter rule to independent biological data not used
  in prior development. Keep labels out of fitting, initialization and condition
  selection; use them only for evaluation and optional root selection.
  pbmc3k raw counts were not in the Paul, Moignard, Olsson, ToggleSwitch,
  Myeloid, krumsiek, or Planaria comparisons. The rule is normalize, log1p,
  1,000 HVGs, PCA 20, and corrected symmetric KL for 200 updates. Processed
  labels are scored only after fitting. Dendritic-cell K15 purity stays near
  0.476 while overall purity is near 0.882. See
  [pbmc3k-frozen.md](pbmc3k-frozen.md).

**Decision gate:** explain global/local/biological/stability trade-offs and unmet
criteria under a fixed protocol. Reaching an epoch limit is not sufficient.
Changing the public default requires a separate decision after these checks;
conditional KL remains experimental in the meantime.

## P2 — Bound spectral initialization cost

- [ ] Consider generic, explicit `tol` and `maxiter` controls and record the actual
  initialization policy and provenance. `main` already has these controls at
  `3cbdd5f`. They are not copied onto this branch.
- [x] Record and warn on non-convergence, preserving a seeded fallback rather
  than silently lowering accuracy. `_seed_from_connectivities` already does
  this on this branch. Explicit `tol` / `maxiter` controls are a separate item
  and already exist on `main`; they are not copied here.
- [x] Compare initial distance structure, seed stability and final quality, not
  just initialization speed. On ToggleSwitch (200 cells, k=15, sigma=1, 200
  updates, seeds 0–2) spectral initialization keeps feature-distance rank near
  0.801 and best loss near 354.6. Random initialization is faster, starts near
  rank 0, and finishes with losses 446–614. See
  [spectral-init-quality.md](spectral-init-quality.md). This does not adopt the
  faster `tol` / `maxiter` setting from `main`.

The [historical initialization probe](benchmarking.md#historical-resource-observations)
identified a runtime bottleneck and a faster bounded setting. That setting has
not been applied to the main initializer; faster does not imply equivalent quality.

## P3 — Investigate general mechanisms, not dataset-specific patches

The graph stress/local-distribution experiment is now implemented separately
from the default RFA pipeline. Its measured-data evidence supports limited
visibility and neighborhood gains, with substantial counterexamples. The
pbmc3k local-radius run does not clear a promotion gate.

- [x] Provide an explicit experimental input-only graph stress implementation
  with mutual global and union local targets, fixed/local-radius bandwidth,
  tangent Adam, exact row chunks and checked optimizer-state resume.
- [x] Compare the same development targets with native Euclidean controls and
  retain failed temperature calibration and population-level losses.
- [x] Validate a frozen rule on genuinely independent measured data before
  considering broader API promotion or a default change. The same pbmc3k
  PCA used for symmetric KL was fit with local-radius graph stress for 1,500
  updates and seeds 0–2. Overall K15 purity is about 0.896 and dendritic cells
  stay near 0.525. The best epoch is the budget limit. No promotion. See
  [pbmc3k-graph-stress.md](pbmc3k-graph-stress.md).
- [ ] Resolve rare-population/macro-purity and developmental-order trade-offs;
  do not optimize only training-related graph metrics.

Remaining research candidates are unvalidated:

- [x] A fixed equal mixture of forest times 0.1 and 10 is available as
  `multiscale_forest_accessibility` on one affinity. It differs from
  `(I+L)^{-1}`, stays row-stochastic, and is not the `estimate_proximity`
  target. Tau is not sigma: repeating tau=1 recovers the unit target, while a
  different sigma changes that target. No Planaria-specific tau was selected.
- [x] Degree-aware widths ``sigma_ij = sigma * sqrt(deg_i deg_j) / mean(deg)``
  are available as `degree_aware_affinity`. Equal degrees reproduce the plain
  Gaussian. Unequal degrees change the edge weight. `local_affinity` and
  `estimate_proximity` do not use the rule. The public scalar `forest_tau`
  control is on `main` (`3cbdd5f`) and is not copied here.
- [ ] Test robustness to mutual versus union kNN, long MST bridges, local
  bandwidth and rare-population coverage. Do not normalize provided PCA scores
  as if they were raw gene counts. This branch's follow-up disqualified
  `union_log` and left `mutual_raw` incomplete. `main` records the finished
  global protocol (`36e88a1`); neither variant is promoted. It is not rerun.
- [x] Compare tangent Adam with plain SGD under the same graph-stress objective
  and the same initial tangent coordinates. Epoch-0 losses match. The default
  remains Adam, and an Adam checkpoint does not resume as SGD. The native
  Euclidean chart keeps that tangent array and does not have the same initial
  pair distances.
- [x] Compare a Lorentz spatial chart with the disk on the same graph-stress
  objective and the same initial disk points. Epoch-0 losses match because the
  hyperboloid map preserves those distances. Later tangent-Adam steps are not
  the same physical steps. The default geometry remains `poincare`. A lower
  Lorentz loss is not a reason to switch the default. Float64 range and
  optimization remain separate questions.
- [x] Investigate temperature/global scale. The input-calibrated row-temperature
  alternative failed on both development datasets and is not promoted. See
  [embedding-study.md](embedding-study.md).
- [ ] Dimension or curvature changes are not justified by that negative result
  and are not started here. `main` does not contain them either.
- [x] Explore matrix-free, row-solve or sparse approximations for scalability.
  Validate against an exact reference because approximation can change global
  normalization and gradients. `sc_poincare.landmark` keeps a sparse teacher
  and landmark-column distances. When every cell is a landmark, paths, scale,
  initialization, teacher and a short Adam trajectory match exact graph stress.
  Fewer landmarks change the endpoint gradient. `max_dense_bytes=None` is
  rejected. This is the synthetic exact reference, not a Paul or Planaria claim.

## P4 — Software and publication readiness

- [ ] Keep objective redesign separate from numerical corrections. Do not add
  dataset-name branches to core code.
- [x] Test this branch and inspect privacy, artifact and license implications.
  `python3 -m pytest tests` passed 211 tests. The published result JSON files
  contain no raw cells and no home-directory paths. `data/` stays gitignored.
  Follow the [provenance constraints](benchmarking.md#scope-and-provenance);
  local inputs and reference artifacts are not covered by MIT. An external
  review of historical commit `494367a` is still the open P0 item.
- [ ] Establish CI, installation, documentation and benchmark reproducibility
  before considering JOSS or similar publication. Check eligibility against the
  rules in force at submission time rather than promising it now. CPU-Torch,
  Torch-free and wheel CI already exist on `main` (`7285e4b`) and are not
  added again here.
- [ ] Obtain explicit approval for new commits or pushes; previous approval does
  not automatically cover future changes or expensive experiments.

The next scientific priority is a fixed-condition comparison of corrected
self-inclusive and conditional KL, followed by long-run, independent-data and
isolation/branch-continuity validation. Preserve unsuccessful and partial runs
alongside improvements; do not substitute attractive display layouts for learned
embedding quality.
