# Astra adoption status and remaining work

Initial source reference: branch `feat/experimental-graph-stress`, pinned commit
`3489e586caaa5b0408a0751220d6349426457bfe`. Latest audited addition: `9f369a9`;
see the [deduplicated mechanism adoption](astra-mechanism-adoption.md). Implemented code, development evidence
and speculative roadmap items are deliberately distinguished. This inventory
does not authorize more experiments, source pushes or continuous future work.
The [subsequent deduplicated follow-up](astra-followup.md) records the fresh
source comparison, newly exposed controls and their bounded checks.

## Implemented and integrated

| Idea | Current status |
|---|---|
| Mutual-global / union-local targets | Implemented in the unchanged low-level MU API and selected as provisional AnnData default |
| Tangent-coordinate Adam | Implemented; not mislabeled as Riemannian Adam |
| Fixed local bandwidth | Provisional default, input-only and dataset-independent |
| Local-radius bandwidth | Explicit option; no automatic dataset-specific selection |
| Classical-scaling initialization | Implemented, with actual target/initializer provenance |
| Exact row chunks retaining all comparison columns and both endpoints | Implemented and tested against full loss/gradient |
| Own-objective best/current/epoch-0 separation | Implemented; evaluation does not choose checkpoints |
| Optimizer-state recovery | Bare MU API retained; AnnData adds an outer cell-identity-bound schema |
| Native Euclidean controls | Implemented/evaluated without disk projection; all full measured controls now finished |
| Independent feature/population/expression/visibility checks | Implemented, with explicit biological and held-out limits |
| Retain counterexamples and resource failures | Initial underflow failures/time stops remain recorded, not replaced by favorable retries |
| Bounded resource measurement | Cold pilots, supervised child RSS/time limits, shared inputs and compact summaries |
| Explicit scalar forest tau | Experimental RFA-only control; tau=1 keeps the previous target |
| Explicit spectral tol/maxiter | Opt-in ARPACK controls, residual/convergence metadata and warned seeded fallback; defaults unchanged |
| Gene-level and declared-stage diagnostics | Implemented as read-only software; not biological validation or root/order inference |
| Upstream development aggregate | Pinned, separately retained JSON; not counted as current reruns |
| Teacher entropy/incoming-mass/normalization | Latest observational metadata ported with upstream analytic tests and main integration checks |
| Latest balanced/multiscale/global-target evidence | Original upstream outcomes/closure retained separately from later user-authorized receiver completion |
| Global completion and stage3/4 experiments | Receiver completion and longer-budget/sensitivity studies finished; no candidate met joint promotion/transfer criteria, no default change |
| Raw 37-fit receiving audit | Transport intact; separate state supplement qualifies CPU roundoff; fresh K metrics and fresh-score local selector agree, with no global-selector execution |
| Explicit Moignard stage mapping | Newly supplied recipe used post-hoc on existing main coordinates with the saved root verified |
| Remote CPU-Torch/Torch-free CI | Passed on main 3cbdd5f; that run does not certify subsequent uncommitted changes |

The two full Moignard Euclidean controls originally stopped at the 45-minute
budget. A later operator-authorized **20-minute continuation budget** restored
the original Adam states (epochs 900 and 0) to the same total target 1500. Both
finished in about 536 seconds. No objective/seed/teacher/learning-rate/source or
chunk changed, and no partial scores were treated as finished comparisons.
They are completed original conditions, not two new hyperparameter experiments.

## Additional compatibility/observability work completed

These integrations do not redesign the low-level objectives or optimizers:

- `method="mu"` is the public provisional default; `method="rfa"` retains the old
  numerical defaults/options. Unsupported RFA-only arguments under MU fail with
  a migration hint instead of being silently ignored. See the
  [explicit compatibility boundary](mu-default-migration.md).
- At default-switch integration, the four MU numerical/checkpoint source files
  were preserved and RFA solver parameter values were unchanged. The subsequent
  [display-layout extraction](display-layout-deprecation.md) deliberately changes
  the whole-file `_disk.py` hash without migrating old checkpoint hashes; numerical
  parity is checked separately from source-version compatibility.
- AnnData progress snapshots carry raw current/best points and optional identity-
  bound checkpoints. Copy mutation has no numerical effect; exceptions and
  externally changed feature/cell identity cannot silently overwrite results.
- H5AD uns stores plain metadata/hashes/loss arrays, not Torch optimizer tensors.
  Restoring nondefault bandwidth/seed/representation is explicit and tested.
- MU's directed teacher and symmetric display connectivity have distinct keys
  and semantics. Switching back to RFA cleans only the previously owned teacher.
- Read-only graph topology inspection flags cutoff ties and distinguishes raw
  reconstructed topology from the actual weighted teacher. No artificial edge
  floor, sigma change or inferred biological repair is applied.
- Root display transforms and derived pseudotime/lineage coordinate provenance
  are recorded. Old files remain readable; absent provenance is unknown, not
  fabricated. Re-fitting does not delete old columns or pretend to refresh them.
- Display jitter/fan/repel are deprecated and isolated in `_legacy_display.py`, ready
  for later deletion with the marked tl export block. Initialization jitter is
  retained. All three display APIs share the same removable legacy module.
- The demo exposes the method selector and does not pass its historical RFA
  presets into MU. Synthetic tests, installation, serialization and CI checks
  cover both methods without downloading private inputs.

## Remaining: research, not a safe drop-in

| Item | Why it is not silently adopted |
|---|---|
| Independent measured-data validation | Current datasets are known development inputs; labels/plots informed earlier research |
| Biological marker/branch/order validation | Per-gene/order evaluators now exist, but named markers, truth and independent protocols remain required |
| Arbitrary initialization / larger seed studies | Existing ranges mainly reflect small jitter around one layout, not confidence intervals |
| Objective-versus-optimizer attribution | MU fixed is a bundled objective/chart/optimizer change; equal updates do not mean equal convergence |
| Global/local and rare-population trade-offs | Some fidelity/purity losses remain; bigger spread is not always more information |
| Multiscale/degree-aware RFA rules | Scalar tau is exposed and probed; no validated mixture or adaptive weighting rule exists |
| Universal bounded spectral defaults | Explicit controls are implemented/probed; quality changes and population losses prevent automatic replacement |
| Matrix-free / row-solve / sparse / landmark scale-up | Can change normalization/gradients and initialization; validate against the exact reference first |
| Intrinsic adaptive optimizers, Lorentz/chart changes | Numerical range and optimization/biology are different questions |
| Curvature, dimension, temperature/global scale | Not established universal improvements; avoid data-name-specific tuning |
| Full Planaria cold-start result | Exact default guard still refuses it; lost warm-start observations are not validated evidence |
| Coordinate-only MU warm starts in the AnnData API | Full state resume is supported; a new arbitrary initializer is a separately testable condition |
| Durable external storage / forced environment replacement | Local atomic checkpoints and successful directory restores do not guarantee survival of container loss |
| Publication readiness | Successful main CI is not biological validation, historical-scope review or publication eligibility |

## Deliberately not promoted

- Latest balanced/multiscale local teachers failed joint promotion/transfer
  criteria. Union-global improved Moignard ordering but lost Paul macro/rare
  purity. At upstream closure mutual-raw screening and global selection/transfer
  were incomplete. Later [authorized completion](algorithm-continuation-results.md)
  finished them with receiver-reconstructed new fits; neither candidate passed
  joint transfer. The [stage3/4 study](stress-validation-34.md) also found no
  consistent joint gain for power1 or k-sensitivity weighting. No candidate is
  automatically enabled, combined or promoted.
- The source input-calibrated row-temperature alternative failed real-data
  population/expression diagnostics despite improving a training-related score.
  It is not enabled as a hidden rule or rerun until a favorable picture appears.
- Lost-environment synthetic/large-data observations are not merged into new
  measured evidence. A 21,296-row artifact cannot be aligned to 21,612 rows by
  truncation. No such alignment or full-Planaria claim is introduced.
- No visual fan/repel/jitter is reported as a learned biological improvement.
- No radius cap change, label-based condition search, probability floor or
  disabled memory guard is used to force a passing result.

## Practical next order

1. Keep the provisional MU fixed default reproducible and explicit in pipelines;
   keep local-radius and RFA selectable rather than guessing a winner by data name.
2. Validate new biological data/markers and a convergence/initialization protocol.
3. Design one declared approximation against the exact 2,730/3,934-cell reference.
4. Only then consider broader scale/default or publication claims. The current
   exact implementation is not atlas-scale merely because a few thousand cells fit.

See [MU results](mu-scaleup-study.md), [migration](mu-default-migration.md) and
[the roadmap](development-roadmap.md) for evidence and execution boundaries.
