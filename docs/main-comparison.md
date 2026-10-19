# Comparison with main

Fetched on 2026-10-19. This branch is `feat/experimental-graph-stress` at
`9f369a9`. `origin/main` and local `main` are `1e724d8`. The merge base is
`9b05bc6`. Nothing below reimplements a behavior that `main` already has.

## What this branch added

`3489e58` adds the opt-in graph-stress embedding and the measured-data study.
`9f369a9` adds teacher entropy, incoming-mass and normalization metadata, plus
the controlled follow-up record. `src/sc_poincare/experimental.py` is identical
to `main`. The graph-stress API and teacher diagnostics are already on `main`
(`a264823`, `01b1a89`). They are not copied again.

`main` also contains later work this branch does not have: MU as the
provisional AnnData default (`619425f`), CPU-Torch / Torch-free / wheel CI
(`7285e4b`), explicit spectral `tol` / `maxiter` (`3cbdd5f`), scalar forest
time `P_tau = (I + tau L)^-1` (`3cbdd5f`), and the completed global-target
and stress-weighting records (`36e88a1`). Those stay on `main`.

## Checklist disposition

| Written item | Disposition |
|---|---|
| Graph-stress API, Euclidean controls, Adam resume, teacher diagnostics | Already on this branch and on `main`. No further implementation. |
| Spectral non-convergence warning and seeded fallback | Already in `_seed_from_connectivities` on this branch. |
| Explicit spectral `tol` / `maxiter`, with a small quality probe | On `main` (`3cbdd5f`). Not ported. |
| Scalar forest time, distinct from sigma | On `main` (`3cbdd5f`). Not ported. |
| Mutual versus union global-target study | Partial on this branch; the joint transfer conclusion is on `main` (`36e88a1`). Not rerun. |
| CI, install and wheel checks | On `main` (`7285e4b`). Not added here. |
| Temperature / global scale | Investigated here. The row-temperature control failed and is not promoted (`embedding-study.md`). |
| Freeze repository, import source and a comparison protocol | Done in `frozen-conditional-kl-protocol.md`. The fit is not started. |
| External review of `494367a` | Not done. CodeRabbit CLI 0.9.0 is installed and not authenticated. A local reread is not that review. |
| Full Planaria, 300 to 1,000 updates, corrected self-inclusive versus conditional KL | Not on `main`. Not run here. See the resource gate in the frozen protocol. |
| More than three seeds, rare-population and marker checks, independent data | Not on `main`. Not run here. |
| Direct feature-distance diagnostic in the public objective driver | Not on `main`. Added here. The full-Planaria check that must use it is still unrun. |
| Initial-distance quality of a faster spectral tolerance | Not established here. `benchmarking.md` still says the faster setting was not applied to this initializer. `main` has the controls and its own small probe; that probe is not copied. |
| Matrix-free or sparse approximation validated against an exact reference | Not on `main`. Added here as `sc_poincare.landmark`, with 16 exact-reference tests. The public Paul feature hash still does not match sklearn 1.9.1 PCA20, so that measured matrix was not used. |
| Dimension, curvature, optimizer and coordinate-chart comparisons | Not on `main` and not started here. The failed temperature result does not justify them. |
| Resolve rare-population and developmental-order trade-offs | Unresolved. No candidate was promoted. |
| Privacy, license and test inspection of this branch's public files | Tests: 211 passed (`python3 -m pytest tests`). Result JSON has no raw cells and no home-directory paths. `data/` stays gitignored. Package MIT does not relicense inputs. |
| Obtain approval before new expensive runs | This comparison and the frozen protocol are the approved record. They do not approve a Planaria run or a new default. |

## Import and artifacts recorded with the protocol

- Repository root: `/workspace`, branch work based on `9f369a9`.
- `pip show sc-poincare` reports version 0.1.0, MIT, editable location `/workspace`.
- `sc_poincare.__file__` is `/workspace/src/sc_poincare/__init__.py`.
- Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Torch 2.14.1+cpu, Scanpy 1.12.4, pytest 9.1.1.
- `data/` is gitignored and was absent. `src/`, `docs/` and `examples/` are present.
- One worktree. Machine: 15 GiB RAM, about 4.8 GiB available, no swap.

## Paul feature hash

Public `Paul.csv` from PoincaréMaps `7ffa518` matches source SHA256
`fed836ed28f148d9ee8efd8ec13ebd468f06ef9b92eeb6f9e218235d4ad2f21d`
(31,121,352 bytes). Dropping the `labels` column leaves 2,730×1,000 finite
values. scikit-learn 1.9.1 `PCA(20, svd_solver="full")` hashes to
`8efb90c0b6145365…`, not the recorded feature SHA256
`0386248d2c02c99f82cda2eb18b8e357b8e0399b10aba7e67575599fb115d90b`.
Centered NumPy SVD, whitening and z-scored PCA also miss. That matrix was
not substituted. The CSV was not committed.
