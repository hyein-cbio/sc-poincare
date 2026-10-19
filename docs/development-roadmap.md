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
- [x] Integrate latest teacher entropy/incoming-mass/normalization observability
  and analytic/integration tests; retain failed mechanism criteria and historical
  global cancellation. The later [authorized completion](algorithm-continuation-results.md)
  finishes the global protocol with receiver-reconstructed new fits, not exact
  old-state resume; neither global variant passes joint transfer. No promotion.
- [x] Inspect common-objective/gradient mechanisms and try the single power2-to1
  stress pilot. Retain its mixed/negative outcomes.
- [x] Complete the separately authorized [stage3/4 small-input study](stress-validation-34.md):
  three seeds/1500 updates, raw/native views and one isolated k-sensitivity
  weighting rule. Neither candidate passes the necessary joint gate. This does
  not close the independent-data, full-size or arbitrary-initialization tasks.

## P0 — Maintain a reproducible starting point

- [x] Freeze the landmark-versus-exact reference check before it runs: input id
  and feature hash, primary and sensitivity metrics, the exact MU configuration,
  landmark count, seeds and the dense-byte limit. The check fits nothing
  (`max_epochs` must be 0). Historical benchmark drivers are unchanged and are
  not covered by this gate. See [landmark-reference.md](landmark-reference.md).
- [x] Review historical commit `494367a` from that commit's diff, recorded in
  [historical-494367a-review.md](historical-494367a-review.md). A review of the
  current tree does not replace it. This record is not a third-party human
  publication audit.
- [x] Write a landmark reference tree with a checksum manifest and require a
  successful local reread before the writer returns. The tested tree is synthetic.
- [x] Keep one latest Moignard landmark-reference tree with a checksum manifest
  and rerun the frozen check from the stored feature matrix. The restored
  primary value matched exactly (absolute difference 0). The feature matrix
  stays under ignored `data/` and is not vendored. Record:
  [moignard-landmark-restore.json](moignard-landmark-restore.json).
- [x] Long or unique jobs still need an off-machine copy of `arrays.npz`.
  The Moignard manifest was uploaded and the downloaded bytes match SHA256
  `262fa3960267a859eac9a29d3b47f3ba0246ae789eea339df6da63f0a535bf19`. The
  report upload is 2095 bytes, the same size as the local file. The feature
  matrix file hash `e244315a…` equals the published feature-file hash, so that
  matrix is rebuilt from the public CSV. The Drive folder is `sc-poincare backups`.
  Hex pieces `arrays.npz.hex.00` through `.30` are 8,000 bytes and `.31` is
  5,896 bytes. The earlier 7,999-byte chunk 13 upload and the 7,985-byte chunk
  08 upload stay in the trash and were not part of the restore. Each of the 32
  current Drive download responses was read back from the persisted tool
  transcript, base64-decoded, and matched the corresponding local hex file
  byte for byte. Concatenating those hex texts and decoding them produced
  126,948 bytes whose SHA256 is
  `75da11742aa4773b28f62463c3c6b63146024e812fb7aaaf494d2f864ad5ca17`, the same
  as the local `arrays.npz`. The zip contains `landmark_index.npy`,
  `exact_initial.npy`, and `landmark_initial.npy`. This copy is on the
  connected personal account `hc27oclock@gmail.com`. No shop Google Drive is
  visible to this connection, so a second copy was not written there.

## P1 — Independent biological validation

The [metadata/readiness audit](validation-readiness-audit.md) verifies Paul and
Moignard gene-header mappings and identifies LARRY as a candidate with no local
fit evidence, not a certified blind holdout. Pancreas and Nestorowa were already
used. Row-order, data-use, subset feasibility and external-backup gates remain
before a new long run.

- [x] Freeze one input-only recipe and run it on 10x pbmc3k, which is absent
  from the recorded studies: total-count normalization, log1p, 1000 Seurat
  HVGs, full-SVD PCA20, MU defaults, 20 updates, seed 0. Labels were not read.
  The feature hash is `076856d9…`. Update-20 loss is 0.7431. Predeclared marker
  Spearman values against origin pseudotime are in
  [pbmc3k-independent-probe.json](pbmc3k-independent-probe.json). This is not a
  certified blind holdout or biological validation. Paul, Moignard, Pancreas,
  and Nestorowa remain development data. LARRY was not fit.
- [x] After the frozen Moignard fits, retain purity and input recall for every
  label population: NP 552, PS 624, 4SG 770, 4SFG 983, and HF 1005. NP is the
  smallest. Marker continuity, branching order, and an unseen rare-population
  audit were not part of this run. Scores did not choose a default. Record:
  [moignard-controlled-probe.json](moignard-controlled-probe.json).
- [x] Compare classical jitter with an area-uniform random disk that does not
  start from the classical layout, seeds 0 and 1, on exact Moignard. Epoch-40
  Poincaré losses are 0.5046 and 0.5047 for jitter, and 0.7612 and 0.7626 for
  the random disk. The policy gap is separate from the tiny jitter difference.
- [x] On those same trajectories, record the loss at 20 and at 40 updates.
  Classical Poincaré loss falls from 0.695 to 0.505 and is still decreasing.
  Euclidean seed 0 at 40 updates has loss 0.557 and input recall 0.185, next
  to Poincaré recall 0.176. Equal update counts are not treated as convergence.
- [x] Predeclare k=15 versus k=30 at 20 updates, marker diameters 4 and 8 on the
  same coordinates, and the PS feature-space medoid (row 1736) as a post-fit
  root only. Pseudotime medians are 0.929 from the classical start and 0.640
  from the random disk. Representation was not varied, and no picture was
  chosen afterward.

The earlier RFA conditional-KL work remains experimental. Its pending fixed-rule
corrected-self-inclusive comparison, longer Planaria runs and isolation checks
have not been satisfied merely by selecting MU as the new provisional default.

## P2 — Bound initialization and scale-up cost

- [x] For RFA, expose explicit spectral `tol` / `maxiter` controls with recorded
  convergence/residuals and seeded fallback; complete the fixed small quality probe.
- [x] On independent pbmc3k features, record the default ARPACK spectral residual
  6.2e-15 and a separate call with tol 1e-8 and maxiter 5000, whose residual is
  1.3e-9. The looser tolerance is not a better initialization. Spectral
  defaults stay unchanged.
- [x] For MU, check one declared landmark-triangle approximation against the
  exact dense target on synthetic inputs: landmark columns, nonself-mean
  normalization, classical-scaling initialization and both-endpoint gradients.
  The exact guard stays on. No default changes. Details:
  [landmark-reference.md](landmark-reference.md).
- [x] Repeat that check on the exact Moignard 3,934×42 feature array
  (`feature_array_sha256` `674f8b38…`, CSV `3f450573…`). Thirty-two landmarks,
  seed 0, zero epochs, and the default 512 MiB guard. Landmark columns agree
  within 2e-14 absolute error. With that partial landmark set the column
  objective's endpoint gradient differs from exact MU (relative L2 0.875).
  Record: [moignard-landmark-reference.json](moignard-landmark-reference.json).
- [ ] Repeat that check on the exact Paul 2,730×20 PCA features. Public
  `Paul.csv` matches source `fed836ed…`. The recorded runtime is NumPy 2.5.4,
  SciPy 1.18.1, scikit-learn 1.9.1, and pandas 3.0.6, the same versions used
  here. One-thread `PCA(20, svd_solver="full")` on the 2,730×1,000 label-stripped
  matrix is deterministic on this host and hashes to `20a9f31a…`, not
  `0386248d…`. The default thread pool is also deterministic and hashes to
  `8efb90c0…`. Float32 input, whitening, covariance eigh, ARPACK, and `auto`
  miss the recorded hash as well. A NumPy SVD of the same centered matrix
  differs from the one-thread sklearn scores by at most 1.8e-12 and still has
  a different byte hash. All 1,048,576 column-sign patterns of the one-thread
  scores miss `0386248d…`. Torch 2.14.1 SVD with the V-based flip hashes to
  `c6e41671…`. OpenBLAS 0.3.34 one-thread kernels hash to
  `93eb6f3a…` (Haswell), `dc67f1b0…` (Sandybridge), `d5b1d157…` (Nehalem),
  and `b24766fa…` (Penryn). Their default thread pools hash to `89beec8b…`,
  `3603a5ed…`, `0fc2e924…`, and `b0fab7f4…`. SciPy `gesvd` on one SkylakeX
  thread hashes to `b96251a8…`. sklearn's full solver matches SciPy `gesdd`
  with the V-based sign flip. None of these bytes is the recorded array, and
  that array was not substituted.
- [x] Fit landmark columns without storing an N-by-N target. The teacher stays
  sparse and each update materializes one row-chunk of embedding distances.
  When every cell is a landmark, a 3-epoch synthetic trajectory matches exact
  MU. A 2-epoch Moignard run with 32 landmarks is a different objective
  ([moignard-landmark-column-fit.json](moignard-landmark-column-fit.json));
  it is not promoted. `max_dense_bytes=None` is rejected.
- [x] Keep full Planaria on a separate protocol. The default dense guard rejects
  21,612 cells, and no exact large-data claim is made from n<=3934 runs.
- [x] Keep the exact dense guard enabled. The landmark-column fitter records
  separate preparation, fitting, evaluation, and checkpoint RSS samples and
  rejects a disabled budget. On exact Moignard those current-RSS samples were
  353, 453, 458, and 459 MiB. A workspace that fits does not mean exact N-by-N
  MU was stored.

## P3 — Research mechanisms, not dataset-specific patches

These are not default changes and require their own controlled experiments:

- [x] Expose scalar RFA forest time `P_tau=(I+tau L)^-1` distinctly from sigma;
  probe fixed .1/1/10 conditions without selecting a new default.
- [x] Probe one fixed degree-aware teacher,
  `sigma_ij = sigma * sqrt(deg_i deg_j) / mean(deg)`, for 20 updates on exact
  Moignard. Its own loss at that budget differs from the default teacher by
  about 1e-5. The input is a development dataset, so the result is not
  promoted and the default teacher is unchanged.
- [x] Validate one fixed multiscale mixture separately: an equal row-normalized
  mix of forest times 0.1 and 10 on exact Moignard. It differs from `tau=1`
  by max absolute value 0.174 and stays row-stochastic. Five-update own losses
  are not compared across targets. The mixture is not promoted, and the
  default remains `forest_tau=1`. Record:
  [moignard-mechanism-probe.json](moignard-mechanism-probe.json).
- [x] Keep the recorded union-global and local/global trade-off results as
  negative promotion evidence. They are not rerun to search for a favorable
  score. See [graph-topology-results.md](graph-topology-results.md) and
  [astra-adoption-status.md](astra-adoption-status.md).
- [x] On exact Moignard, compare tangent Adam with plain SGD on the same MU
  objective, classical jitter seed 0, and learning rate .01333 for 20 updates.
  Both start at loss 1.0369. Adam reaches 0.6952 and SGD stays at 1.0368.
  SGD is not promoted.
- [x] On exact Moignard, compare Poincaré ball dimensions 2 and 3 for 20
  updates from classical scaling and the same jitter seed. Losses at update 20
  are 0.6952 and 0.6069. Macro purity is 0.546 and 0.543, so the lower 3D
  loss is not treated as a better embedding. The 2D default is unchanged.
  Predeclared marker Spearman values against PS-root pseudotime and the stage
  means are in [moignard-dimension-probe.json](moignard-dimension-probe.json).
  The predeclared order PS, 4SFG, 4SG, NP does not match the mean pseudotimes.
- [x] On exact Moignard, compare Poincaré curvatures 1 and 0.25 for 20 updates
  from the same classical jitter. Update-20 losses are 0.6952 and 0.6985.
  Macro purity is 0.546 and 0.545. Curvature 1 remains the default. Record:
  [moignard-curvature-probe.json](moignard-curvature-probe.json).
- [x] On exact Moignard, map the classical jitter start onto the Lorentz
  hyperboloid and take 20 Minkowski retractions at the disk learning rate.
  The initial loss matches the disk, 1.0369. At update 20 the Lorentz loss is
  1.0379, macro purity 0.528, and input recall 0.091, against the disk Adam
  values 0.6952, 0.546, and 0.136. The retraction is not promoted. Record:
  [moignard-lorentz-probe.json](moignard-lorentz-probe.json).
- [x] On exact Moignard, run bias-corrected Riemannian RMSProp for 20 updates
  from the same classical jitter start. The initial loss matches tangent Adam
  at 1.0369. At update 20 the intrinsic loss is 0.4936, macro purity 0.565,
  and input recall 0.213, against Adam's 0.6952, 0.546, and 0.136. One short
  development run does not change the tangent-Adam default. Record:
  [moignard-intrinsic-probe.json](moignard-intrinsic-probe.json).
- [x] Keep the failed source row-temperature alternative unpromoted. One
  improved score is not a reason to enable it. See
  [astra-adoption-status.md](astra-adoption-status.md).

## P4 — Software/publication readiness

- [x] Keep mechanism probes in `controlled_probe.py`, separate from the default
  MU and RFA numerical paths and from data parsing.
- [x] Review privacy, licenses, installation, documentation, and reproducibility
  in [publication-readiness-review.md](publication-readiness-review.md).
  Package MIT does not relicense input artifacts. This is not a submission.
- [ ] Check any publication eligibility against current rules when submitting.
- [ ] Obtain explicit approval for commits/pushes and new expensive experiments.
  Earlier approval does not silently authorize future jobs.

The next algorithm-scale priority is an independently validated approximation
against the exact MU reference, not unlimited brute-force N-by-N storage.
The next scientific priority is independent biology/initialization validation.
