# Lorentz distance and gradient isolation, 2026-11-03

The historical loss increase is reproduced with the old acosh distance and old gradient conversion. Correcting the gradient removes that increase under either distance expression. Retaining the old gradient with the current disk-distance expression also decreases loss. This fixed-seed comparison therefore isolates a distance-by-gradient interaction; a distance-independent explanation based only on the gradient switch would be incomplete.

All six factorial arms completed 20 updates. A separate verbatim historical fitter, including full backward, also completed 20 updates. These are Moignard development-data numerical checks. They do not establish improved biological performance, reproduce the original Poincare Maps algorithm's performance, or change a model default.

## Historical fidelity

The reference is the parent of the PR #3 numerical fix:

- [Pre-fix commit `cd6ae2d`](https://github.com/hyein-cbio/sc-poincare/commit/cd6ae2d6a9754b34e934d72328ac7cb9ec1a17d8); `controlled_probe.py` blob `71bab09d35ac5db49ae9ef801b4919c5b9e23d32`.
- [Fix `13ce854`](https://github.com/hyein-cbio/sc-poincare/commit/13ce854a00aedeb4f115749159ee96c955da0906).
- [`examples/fixtures/pre_pr3_lorentz.py`](../examples/fixtures/pre_pr3_lorentz.py) contains the original `fit_lorentz` function text verbatim. Its function-text SHA256 is `97bde9824263931ceea71f286eadc6cefd97c85d1c15b912817e2d0d36830d2a` (LF lines and final newline). The imported `lorentz_from_disk` and `_minkowski` helpers are text-identical to that commit. Current setup validation is retained.

| Control | What is restored | Fidelity limit |
|---|---|---|
| `disk_stable` | Current disk map and stable distance | Current implementation control |
| `acosh_exact` | Original `acosh(clamp(-inner, min=1))`, clone, then zero diagonal | Exact historical distance operations; factorial backward remains chunked |
| `acosh_masked` | Acosh with self/clamped coincident arguments masked before acosh, then zeroed | New stabilized reconstruction of backward behavior; never a historical replay |
| Separate historical fitter | Original complete fitting function, old gradient and full backward | Exact function replay on the shared prepared setup under the current runtime; not a restoration of all historical dependencies |

The stabilized reconstruction adds no epsilon to positive pair distances. It assigns zero derivatives to self/clamped coincident pairs before evaluating acosh. Its provenance remains distinct even when it gives identical measurements here. The exact control does not silently mask, floor, replace NaNs, or fall back to a different distance. The research runner records a nonfinite gradient as a failed arm at its last finite checkpoint rather than calling it a completed fit. Acosh boundary backward behavior can vary by Torch version.

## Shared preparation

The base is `d1ffcd8f7e188627ed8c623499789732f1827769`. Execution source is local commit `d827270d5eceb207076c702c22c049a9bdb0fb92`. Published source commit [`ceb03f4`](https://github.com/hyein-cbio/sc-poincare/commit/ceb03f4140b3d5a1c4c58db633515c25a9d5a3b7) has identical fitting/verification source fingerprints; its tests also include portable handling of historical boundary failures. The JSON fingerprints each relevant source file. Production fitting and geometry files are unchanged.

The input is the [authors' processed Moignard CSV at a pinned commit](https://github.com/facebookresearch/PoincareMaps/blob/7ffa518cabe6e6424a875d480c4298bda3bd9678/datasets/Moignard2015.csv). CSV and float64 feature hashes match the prior rerun. All 3,934 cells and 42 features are retained without scaling, PCA, filtering or reordering. The last column supplies post-fit evaluation labels only. The original data license applies; the input is not vendored.

One `GraphStressConfig()` setup and one seed-0 classical jitter are shared across every arm. Each fit starts afresh, uses the same target and teacher arrays, learning rate `0.013333333333333334`, curvature -1, retraction and 20-update budget. Factorial arms use chunks of 256 with chunk-wise backward. The shared dense guard is 512 MiB; this is not a bound on full-backward autograd memory.

| Shared array | SHA256 of C-contiguous float64 bytes |
|---|---|
| Initial disk coordinates | `672b16237a5324038fe60f5ad6e1e3f67c18eeffca8e10cb9ea5146cec98d60d` |
| Target distances | `1419437d1d13c4f267d9e0de1a7de21345f647387d00dbc5f9f263f7ec0fbfcb` |
| Teacher | `a95c1ca46913e1f48e6d919e906482ee8c393335a217ae954795e30e20cdbdb7` |

Runtime: Linux, Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, pandas 2.3.3, scikit-learn 1.8.0, Torch 2.15.0+cpu and threadpoolctl 3.6.0. Torch and BLAS/OpenMP use one thread; fitting uses CPU float64. The complete run took about 176 seconds on this host. The newly prepared initial coordinates differ from the saved October 30 coordinates by at most `5.46e-15`; they are not byte-identical across these runtimes. Within this experiment all arms share the exact same initial array.

## Measurements

Every arm starts at loss `1.0369251291534152`. Metrics use all cells, k=15, Euclidean feature neighbors and self-excluding Poincare embedding neighbors. Population purity is macro-averaged; recall is cell-averaged. Every population is retained in the JSON.

| Distance | Gradient | Final own loss | Increasing steps | Input-neighbor recall | Macro purity |
|---|---|---:|---:|---:|---:|
| disk_stable | corrected | 1.036791034182 | 0/20 | 0.093119810202 | 0.528934173291 |
| disk_stable | legacy | 1.036787350317 | 0/20 | 0.093119810202 | 0.528894225317 |
| acosh_exact | corrected | 1.036791034182 | 0/20 | 0.093119810202 | 0.528934173291 |
| acosh_exact | legacy | 1.037866719949 | 20/20 | 0.090713438400 | 0.527866085374 |
| acosh_masked | corrected | 1.036791034182 | 0/20 | 0.093119810202 | 0.528934173291 |
| acosh_masked | legacy | 1.037866719949 | 20/20 | 0.090713438400 | 0.527866085374 |

Under corrected conversion, changing the distance changes final loss by only `1.78e-15` and final Lorentz coordinates by at most `2.41e-13`. Under the old conversion, restoring acosh raises final loss by `0.001079369632` relative to the current distance. Exact and stabilized acosh produce byte-identical traces and final coordinates for each gradient choice on this input.

This is consistent with the distance formulas agreeing on the constrained manifold but having different ambient gradients. Correct metric conversion/projection removes the irrelevant normal component; the old conversion can depend on that extension. Synthetic tests compare the corrected tangent gradients directly and show the old converted gradients differ. Tangency alone is insufficient: the incorrect conversion can still yield tangent vectors while failing the metric-dual identity. Its slightly lower training loss under the current distance is not evidence that the old gradient is valid.

## Backward schedule control

The verbatim historical fitter completed a one-update finite preflight and a fresh 20-update full-backward run:

- Initial loss `1.0369251291534152`; final loss `1.0378667195859907`; all 20 steps increase loss.
- The archived historical final loss is `1.0378667202542602`, an absolute difference of `6.68e-10`. All aggregate and population neighborhood metrics match the archive exactly.
- Relative to old-distance/old-gradient chunk-wise backward, final loss differs by `3.63e-10`, the largest trace difference is `7.47e-10`, and the largest final-coordinate difference is `1.59e-6`. Neighborhood metrics remain exactly equal.

Thus backward scheduling produces small numerical differences here but does not explain the observed change from increasing to decreasing loss. This is a comparison under this runtime and budget, not a claim of bitwise historical replay or a universal bound on chunking effects. The invalid-gradient run is sensitive to small numerical changes.

## Numerical and saved-record checks

All recorded losses, gradients and coordinates are finite in the six factorial arms. Every recorded iterate remains in the forward sheet and open evaluation disk. The largest recorded hyperboloid constraint error is `5.56e-16`; the smallest disk margin is greater than `0.9599`. The historical full-backward losses and final coordinates are finite too.

Each final checkpoint was evaluated under all three distance implementations. Their objective values differ by at most `1.78e-15`, so the comparison is not an artifact of incompatible final scoring. The independent reread verifier reconstructs the setup, checks source/NPZ/shared-array hashes and initialization, verifies trace shape/endpoints and finite checkpoint status, recomputes every final objective, all population metrics and final stability. All reread objective errors were zero on this host; every metric and stability result matched exactly. It also checks the saved historical preflight and replay. It does not refit or recompute each interior-epoch objective from coordinates; intermediate coordinates are not archived. The per-epoch geometry/gradient diagnostics are observations made during fitting.

Regression coverage includes verbatim function-text fidelity, literal historical distance/backward parity, equivalence to the existing two-arm fitter, full/chunked backward, corrected-gradient chart agreement, finite stabilized coincident gradients, explicit failed-arm labeling, write-once output protection and saved-record corruption detection. The exact-distance tests preserve runtime-specific boundary behavior rather than requiring a particular NaN outcome.

Validation: the 27 new Lorentz tests plus 13 existing record tests passed (40 total); `compileall` and `git diff --check` passed. Before optional psutil installation, the full suite reported 753 passed, 18 skipped and 2 warnings. With psutil installed, the full suite reported 767 passed, 4 failed and 2 warnings. Three failures are `psutil.NoSuchProcess` in unchanged landmark resource tests; one supervisor test observes zero RSS. The container exposes different PID values through `os.getpid()` and `/proc/self/stat`, and psutil can resolve an unrelated process. This is consistent with the three PID lookup errors, but those three failures did not reproduce in a targeted main run. A full run of unchanged main reported 743 passed and the same supervisor zero-RSS failure. These local resource-accounting failures remain unresolved and are not reported as passing checks. Standard-runner CI is a separate check.

## Reproduce

Use an environment with the package's torch and benchmark dependencies. No private data is needed. Obtain the pinned `SOURCE_URL` in `src/sc_poincare/_probe_record.py` as `data/inputs/Moignard2015.csv`; the loader verifies both input hashes before preparation. Use new output names because the runner refuses existing JSON/NPZ files and symlinks.

```sh
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python examples/run_moignard_lorentz_probe.py \
  --source data/inputs/Moignard2015.csv \
  --output docs/moignard-lorentz-factorial-NEW-RUN.json \
  --distance-ablation --historical-replay
python examples/verify_moignard_lorentz_factorial.py \
  --source data/inputs/Moignard2015.csv \
  docs/moignard-lorentz-factorial-NEW-RUN.json
python -m pytest tests/test_lorentz_ablation.py tests/test_probe_record.py
python -m pytest tests
```

Without `--historical-replay`, the same six arms run without the more memory-intensive verbatim full-backward fitter. Without `--distance-ablation`, the existing current/gradient-only driver behavior remains available. Filesystem containment guards still apply.

## Records and limits

- [Measurements and provenance](moignard-lorentz-factorial-2026-11-03.json), with [coordinate/trace NPZ](moignard-lorentz-factorial-2026-11-03.npz).
- [Preservation and reread audit](moignard-lorentz-factorial-audit-2026-11-03.json).
- [Prior revalidation](moignard-pr3-revalidation-2026-10-30.md) is retained as the historical account of what was known before this experiment.

All 80 preexisting tracked files under `docs/`, including historical research results, remain byte-for-byte unchanged. This experiment adds records with separate names. Independent datasets, additional seeds, longer convergence budgets, marker continuity, branch order, pseudotime and unseen rare populations remain unvalidated. All conditions retain `biological_validation=false` and `promoted=false`. The next biological-performance comparison still needs an independently specified endpoint and data; these short numerical probes do not supply it.
