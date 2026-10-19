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
- [ ] Compare against the **corrected self-inclusive default**, not only the
  historical probability-floor baseline.
- [ ] Select checkpoints by the training objective, not evaluation scores.
  Preserve raw current/best coordinates, loss, gradient norm, radius and stop
  reason throughout the trajectory. The fitter already returns the lowest
  training loss, including initialization. `gradient_norm_trace` now stores
  the max per-point Riemannian norm before each update, aligned with
  `loss_trace`. The full Planaria trajectory is still unrun.
- [ ] Check evaluation graphs at k=15/20/30, alternative anchors and direct
  feature-distance diagnostics. Declare primary and sensitivity checks before
  fitting rather than selecting favorable evaluations afterward. The public
  objective driver now records direct feature-distance Spearman on the same
  anchors, and it already accepts several `--evaluation-k` and `--anchor-seed`
  values. That software does not replace the unrun full-Planaria check. The
  declared metrics are in the
  [frozen protocol](frozen-conditional-kl-protocol.md).
- [ ] Evaluate seed consistency separately from accuracy; plan more than three
  seeds and uncertainty estimates.
- [ ] Inspect isolated, rare and outlier populations for artificial mixing or
  loss of terminal-state information when the RFA diagonal is discarded.
- [ ] Evaluate biological markers, branch continuity, root distances and
  pseudotime. Good graph scores alone do not establish central branching.
- [ ] Apply a frozen hyperparameter rule to independent biological data not used
  in prior development. Keep labels out of fitting, initialization and condition
  selection; use them only for evaluation and optional root selection.

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
- [ ] Compare initial distance structure, seed stability and final quality, not
  just initialization speed.

The [historical initialization probe](benchmarking.md#historical-resource-observations)
identified a runtime bottleneck and a faster bounded setting. That setting has
not been applied to the main initializer; faster does not imply equivalent quality.

## P3 — Investigate general mechanisms, not dataset-specific patches

The graph stress/local-distribution experiment is now implemented separately
from the default RFA pipeline. Its measured-data evidence supports limited
visibility and neighborhood gains, with substantial counterexamples. It does
not satisfy the independent-data or general biological decision gates above.

- [x] Provide an explicit experimental input-only graph stress implementation
  with mutual global and union local targets, fixed/local-radius bandwidth,
  tangent Adam, exact row chunks and checked optimizer-state resume.
- [x] Compare the same development targets with native Euclidean controls and
  retain failed temperature calibration and population-level losses.
- [ ] Validate a frozen rule on genuinely independent measured data before
  considering broader API promotion or a default change.
- [ ] Resolve rare-population/macro-purity and developmental-order trade-offs;
  do not optimize only training-related graph metrics.

Remaining research candidates are unvalidated:

- [ ] Forest time scale `P_tau = (I + tau L)^-1`: distinguish the propagation
  range controlled by tau from relative edge weights controlled by kernel sigma.
  Consider fixed multiscale mixtures or degree-aware rules; freeze input-only
  rules and control protocols before searching for a Planaria-specific best tau.
  The scalar control itself is on `main` (`3cbdd5f`) and is not copied here.
  Fixed multiscale mixtures are still unvalidated on `main`.
- [ ] Test robustness to mutual versus union kNN, long MST bridges, local
  bandwidth and rare-population coverage. Do not normalize provided PCA scores
  as if they were raw gene counts. This branch's follow-up disqualified
  `union_log` and left `mutual_raw` incomplete. `main` records the finished
  global protocol (`36e88a1`); neither variant is promoted. It is not rerun.
- [ ] Compare optimizers and coordinate charts under the same objective and
  initial distances. Intrinsic adaptive updates, Lorentz coordinates and
  Euclidean parameterizations are candidates; distinguish float64 representation
  limits from optimization and do not assume Lorentz is always better.
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
