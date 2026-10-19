# Latest Astra mechanism follow-up: adoption without duplication

## Pin and existing work

Fresh upstream reference: `feat/experimental-graph-stress` commit
`9f369a9`, following the previously integrated `3489e58`.
The main base at integration was `3cbdd5f`. Existing metadata-readiness working
notes were retained, not overwritten or treated as changes from Astra.

The latest upstream code delta is observational teacher metadata and three
analytic test cases. The 37 quality-fit executions are upstream research
**evidence**, not production implementations or new fits performed on main.
Earlier MU fixed/local-radius, exact row chunks, Adam resume, Euclidean controls,
RFA tau/spectral controls, per-gene and declared-order evaluators already exist
on main. They were not reimplemented or rolled back to the upstream RFA default.

## Adopted software

The maintained `experimental.py` is byte-identical to the pinned latest Astra
file. Preparation now records:

| Field | Meaning |
|---|---|
| `teacher_entropy_mean` | Mean row entropy in nats |
| `teacher_entropy_quantiles` | Entropy distribution, zero mass contributes zero |
| `teacher_incoming_mass_quantiles` | Distribution of column sums |
| `teacher_max_asymmetry` | Maximum absolute elementwise P minus P transpose |
| `teacher_max_row_sum_error` | Maximum row normalization error |
| `teacher_diagnostic_quantiles` | Levels 0, .1, .5, .9, 1 |

Existing AnnData metadata propagation exposes these fields in
`uns['poincare']['proximity_diagnostics']`, and callbacks/checkpoints carry them.
No second metadata implementation or normalization pass was needed. Preparation
remains Torch-free. Targets, initialization jitter, teacher weights, optimizer,
loss and defaults are unchanged.

A row-normalized symmetric affinity is generally an asymmetric reversible
teacher; that is not evidence of invalid probabilities or asymmetric geometry.
Incoming mass measures normalization, not equal label/lineage importance. The
local CE derivative for a shared symmetric pair distance is
`(P_ij + P_ji - Q_ij - Q_ji)/(n*gamma)`. Even a doubly stochastic teacher weights
populations through their cell counts, not equally by population. Sparse teacher
support and full-support softmax also limit exact equality at finite distances.

Mean entropy is a lower bound on CE over distributions. CE minus entropy is
teacher-specific KL; neither raw CE nor its entropy-adjusted value becomes a
common biological quality score across changed targets. No automatic loss or
learning-rate rescaling is introduced.

### Tests and source boundary

Imported the three upstream analytic cases: uniform triangle under both
bandwidths and the closed-form asymmetric line. Added independent integration
coverage for >128-row chunking, read-only nonmutation, invalid teachers,
reversibility, CE lower bound, the symmetric-pair derivative, callback metadata
isolation and AnnData/H5AD round trips. Existing gradient/resume and provenance
tests remain; they were not replaced by the new analytic cases.

Separate-process before/after checks used frozen full Paul PCA20, both bandwidths,
both geometries and 20 updates. **262 array/tensor/scalar checks matched exactly**,
including prepared arrays, initial/current/best coordinates, loss histories,
best-state selection, tangent states and Adam state. This is a numerical
regression probe, not another 1500-update biological quality study.

`experimental.py` and metadata/checkpoint hashes intentionally change. Historical
states need their original implementation/environment; no hashes were migrated,
whitelisted or silently rewritten. Current-source new-state resume is checked by
the normal suite. Numerical parity does not waive source integrity.

## New source evidence and explicit non-adoptions

The immutable aggregate is [astra-mechanism-results.json](astra-mechanism-results.json),
verbatim from upstream (SHA256
`5b21ca5c3aa38a76819dbb48f0d5e55d335be5845d963a2012a284319ba264c3`).
It contains no raw cell matrices, private paths or optimizer payloads. Supporting
raw capsules and audit tools remained upstream at adoption time; their
arithmetic/recovery claims were then reported as upstream claims. A subsequent
[receiving-session capsule audit](astra-capsule-receiver-audit.md) now verifies
raw delivery/state lineage (with a cross-platform roundoff caveat) and freshly
recalculates all 37 per-run K metrics plus the local selector decision. It does
not claim new training or independent biological validation.

| Item | Latest upstream outcome | Main decision |
|---|---|---|
| Balanced teacher / symmetric Sinkhorn | Missed development targets and joint Olsson transfer criteria | Do not implement as a recommended/automatic teacher |
| k15/k30 multiscale local teacher | Guardrail violations and failed transfer criteria | Do not add another public bandwidth rule from negative evidence |
| `union_log` global target | Moignard ordering improved; Paul macro/rare-label protection failed | Preserve result and trade-off, not a universal default/graph change |
| `mutual_raw` global target | Seven of eight screening fits completed overall; one counterpart cancelled | Incomplete evidence, not a completed selection or transfer result |
| Follow-up confirmation/transfer after global screening | Not executed after user closure | Do not restart old queues or inherit old budgets |
| Paired derivatives, entropy, incoming mass | Input-only explanatory diagnostics with analytic checks | Adopted without changing the objective |
| Exact replay/recovery results | Reproduced computation, not independent biological replicates | Keep execution counts and provenance separate |

The local-teacher protocol has **30 completed fits**. The global-target study has
**7 additional executions**; four reused controls appear in its records but
are not new executions. Combined total is **37**, including four exact earlier
baseline replays, not 37 independent experiments. Global formal selection,
confirmation and Olsson transfer did not run. Missing work is cancelled, not a
negative-quality score. The observed `union_log` guardrail failures still count
as evidence against that rule, without inventing a completed global selector.

Specific evidence worth retaining:

- Moignard `union_log` stage-rank gains: about +.0979 Poincare and +.1232 native
  Euclidean at seed 0. The Euclidean gain prevents a hyperbolic-specific claim.
- Paul `union_log`: Poincare macro15 about -.00484 and Lymph purity -.02366,
  despite lower expression MSE. Native Euclidean macro/Lymph also decline.
- Balanced Olsson Poincare macro15 improves in all three seeds while MSE30
  worsens in all three; this fails the joint rule rather than proving that
  every individual endpoint failed.
- `mutual_raw` misses the stated Paul composite target; its Moignard Poincare
  MSE15 rise violates the guardrail. Its cancelled Euclidean counterpart and
  missing later phases do not provide seed or transfer evidence.

No target combinations, best-looking seeds, label-driven population weights,
floors, radius/display forces or new optimization schedules were introduced.
This focused port does not answer independent biology, arbitrary initialization,
convergence, out-of-sample prediction or large-data approximation questions.

## Reused main coordinates: newly specified stage mapping

Earlier main follow-up did not guess a Moignard stage order because the source
aggregate lacked a full recipe. Latest Astra text now explicitly gives
`PS=0, NP=1, HF=2, 4SG=4SFG=3`, with the PS input-feature medoid as root.
This permits an explicitly **post-hoc** calculation without new training.

The saved main root index 1736 was verified against the Euclidean sum-distance
medoid of the PS feature rows. We passed the tied numeric ranks to the existing
`evaluate_ordering(..., stage_order=[0,1,2,3])`; there is no second ordering API,
no root search based on the answer, and no display transform. Distances are raw
hyperbolic or native Euclidean root distances as appropriate.

| Existing main method | Stage Spearman |
|---|---:|
| RFA shared-init study baseline | .197082 |
| MU fixed, Poincare | .317099 |
| MU local-radius, Poincare | .318435 |
| MU fixed, native Euclidean | .298727 |
| MU local-radius, native Euclidean | .298414 |

This is one supplied-label order diagnostic on development data, not measured
lineage truth or unseen validation. It is not a rerun of upstream variant fits.
Paul upstream fitted-feature bytes differ from main's frozen PCA20 bytes;
Moignard feature bytes match, but recorded ordering values differ slightly.
Upstream scores and main scores therefore remain separate provenance records;
no exact upstream-quality replay or pooled seed interval is claimed.

## Local software verification

Complete suite: **517 passed**, with two existing exact-zero-target warnings.
A freshly created Torch-free wheel environment ran **301 passed / 58 optional
skips**; the previous scratch environment no longer existed, so no earlier result
was reused as a current run. Installed-wheel MU fitting/new-state resume, teacher
metadata and H5AD serialization passed outside the source tree. The scoped
CodeRabbit review completed with zero findings. These checks do not claim a
remote CI run for uncommitted work.

Local compact parity/order records live under
`data/graph_diagnostics_integration/astra_latest/`. Historical original study
counts, input arrays, coordinates and checkpoint tags are not overwritten.
No long quality fit, external backup, commit or push is implied by this update.
