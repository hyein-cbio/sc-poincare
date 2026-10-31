# PR #3 Moignard revalidation, 2026-10-30

Both authorized 20-epoch experiments ran successfully on the checksummed public 3,934 x 42 matrix. Corrected Lorentz loss decreases slightly; curvature loss trajectories reproduce the historical values. Curvature-aware evaluation changes neighborhood scores slightly at c=0.25. These are development-data numerical checks, not independent biological validation. No default changed.

## Revision and preservation

- Base: remote `origin/main`, `cbb75d99f31f29b7474dbd0f812ca5bcab2df22a`. Local main matched the fetched remote. PR #3 merge is `3769326580def3648353f2606808cbd73ef1ff25`; current core source is unchanged from that merge.
- Work branch: `codex/moignard-pr3-validation-20261030` in a separate worktree. The original checkout remains clean on main. All edits are unstaged; no commit or push was made.
- All 26 preexisting tracked JSON files under docs, including both historical probe records, match HEAD byte for byte. Historical prose and research archives are unchanged. The audit JSON records their SHA256 values.
- Updated drivers accept `--source` and `--output`, use a separate dated default output, refuse existing JSON/NPZ and symlinks, and retain closed-research path guards. NPZ files retain initial/final coordinates and every recorded loss.

## Input and execution

Public source: [PoincareMaps CSV at pinned commit](https://github.com/facebookresearch/PoincareMaps/blob/7ffa518cabe6e6424a875d480c4298bda3bd9678/datasets/Moignard2015.csv). This is the authors' public processed matrix, not a reconstruction of upstream raw single-cell preprocessing. The data license remains the upstream license.

```text
CSV SHA256:     3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1
Feature SHA256: 674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0
```

The last CSV column is excluded from fitting. No row filtering/reordering, feature scaling, or PCA is applied. The feature hash covers raw C-contiguous float64 bytes, without a shape/dtype header. Population counts are NP 552, PS 624, 4SG 770, 4SFG 983 and HF 1005. The temporary input path in the old scripts was absent; the pinned CSV was downloaded into the new worktree's ignored active input folder and both hashes verified before fitting.

Both runs use the existing `GraphStressConfig`, classical jitter seed 0, 20 updates, chunks of 256, learning rate 0.013333333333333334 and the 512 MiB dense guard. Computation uses CPU float64 and one OMP/BLAS/MKL thread. No dependencies were installed. The two drivers produced identical initial arrays; two executions of each driver produced exactly equal saved arrays. This checks repeatability on this host, not variation across seeds.

Runtime: Python 3.14.10, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0, scikit-learn 1.9.1 and Torch 2.14.1. This does not reproduce every historical package version. The JSON records full configuration, runtime/thread information, code hashes and array hashes.

## Historical and current measurements

Neighborhood metrics use all cells, k=15, Euclidean feature neighbors, and self-excluding hyperbolic embedding neighbors. Macro purity weights populations equally; aggregate recall weights cells equally. These are descriptive scores, not biological correctness.

| Condition | Historical loss 0 -> 20 | Current loss 0 -> 20 | Macro purity, old -> new | Input recall, old -> new |
|---|---:|---:|---:|---:|
| Lorentz | 1.036925129 -> 1.037866720 | 1.036925129 -> 1.036791034 | 0.527866085 -> 0.528934173 | 0.090713438 -> 0.093119810 |
| Curvature c=1 | 1.036925129 -> 0.695174801 | 1.036925129 -> 0.695174801 | 0.546047413 -> 0.546047413 | 0.136451449 -> 0.136451449 |
| Curvature c=0.25 | 1.037585148 -> 0.698535527 | 1.037585148 -> 0.698535527 | 0.545134963 -> 0.545121882 | 0.136739536 -> 0.136688697 |

## Separating the Lorentz changes

The current conversion first changes the sign of the Euclidean gradient's time component, then projects using the Minkowski inner product. Existing regression tests check tangency, the metric-dual identity, descent, finite self/coincident gradients, and chunked/full-objective agreement. The current distance objective maps hyperboloid points back to the disk and uses the stable disk-distance implementation; PR #3 also changed backward execution to one chunk at a time.

The corrected 20-update trajectory decreases by 0.000134094972. With only the old, incorrect gradient conversion restored, while retaining the current distance objective and chunking, loss also decreases to 1.036787350317; recall is 0.093119810202 and macro purity is 0.528894225317.

Therefore the historical increase cannot be attributed solely to the gradient conversion from this comparison. In this fixed-distance control, the incorrect conversion even has a slightly lower final loss, by about 3.68e-6. That is not evidence that the old gradient is mathematically valid. Changing distance expressions can change their ambient gradients away from the constrained manifold; the incorrect projection can interact with that change. This is an interpretation, not an isolated measurement of the former distance implementation.

The complete old acosh-distance implementation was not replayed for 20 epochs. This run validates the corrected implementation and isolates the gradient switch under the current distance objective; it does not quantify every pre-fix component separately. Lorentz uses a different optimizer from tangent Adam, so its small loss reduction does not establish chart inferiority, convergence, or an equivalent compute budget.

## Separating curvature fitting and evaluation

For curvature magnitude c, `d_c(x,y) = d_1(sqrt(c)*x, sqrt(c)*y) / sqrt(c)`. The positive factor does not change neighbor ranks. The corrected evaluation scales coordinates by sqrt(c); the legacy control evaluates the exact same final coordinates without scaling. It performs no additional optimization.

The c=1 loss/metrics reproduce the archived result to floating-point precision. For c=0.25, the legacy evaluation exactly reproduces every archived population score and aggregate score on the new coordinates, while the loss matches within floating-point precision. Thus the observed neighborhood-score differences are directly isolated to evaluation geometry in this run.

At c=0.25, matching geometry changes macro purity by -0.000013080816 and aggregate recall by -0.000050838841. Across the two current curvature fits, c=0.25 minus c=1 gives purity -0.000925531027 and recall +0.000237247924. There is no joint aggregate improvement. Own-objective losses across different curvatures are not a common model-selection score.

## Every population

Each cell shows historical -> current. Labels entered only post-fit evaluation. NP is the smallest recorded population; this is not a dedicated unseen rare-population or outlier audit.

| Population | n | Lorentz purity | Lorentz recall | c=1 purity | c=1 recall | c=0.25 purity | c=0.25 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| NP | 552 | 0.284420 -> 0.283575 | 0.099155 -> 0.099275 | 0.307488 -> 0.307488 | 0.147585 -> 0.147585 | 0.304227 -> 0.303986 | 0.149155 -> 0.149034 |
| PS | 624 | 0.407479 -> 0.413675 | 0.095620 -> 0.100962 | 0.437073 -> 0.437073 | 0.151816 -> 0.151816 | 0.436645 -> 0.436752 | 0.152885 -> 0.152671 |
| 4SG | 770 | 0.545628 -> 0.545368 | 0.069437 -> 0.068658 | 0.571688 -> 0.571688 | 0.101558 -> 0.101558 | 0.571342 -> 0.571342 | 0.101818 -> 0.101905 |
| 4SFG | 983 | 0.949067 -> 0.948321 | 0.122889 -> 0.128654 | 0.945066 -> 0.945066 | 0.182028 -> 0.182028 | 0.945202 -> 0.945337 | 0.181485 -> 0.181350 |
| HF | 1005 | 0.452736 -> 0.453731 | 0.067861 -> 0.068856 | 0.468922 -> 0.468922 | 0.102952 -> 0.102952 | 0.468259 -> 0.468192 | 0.102886 -> 0.102952 |

Lorentz aggregate gains are mixed by population: NP, 4SG and 4SFG purity decline; PS and HF purity rise. 4SG recall declines. For c=0.25, corrected evaluation lowers NP/HF purity and raises PS/4SFG purity; 4SG purity is unchanged. These population losses are retained.

## Numerical stability and reread verification

| Condition | Loss-increase steps / 20 | Max unit-ball radius | Min 1 - radius squared | Max Lorentz constraint error |
|---|---:|---:|---:|---:|
| Corrected Lorentz | 0 | 0.200173950 | 0.959930390 | 6.661e-16 |
| Old-gradient-only control | 0 | 0.200179572 | 0.959928139 | 6.661e-16 |
| Curvature c=1 | 0 | 0.361222218 | 0.869518509 | n/a |
| Curvature c=0.25 | 0 | 0.185976267 | 0.965412828 | n/a |

Every recorded loss and final coordinate is finite. Lorentz final time coordinates are positive. No exception or numerical-stability intervention occurred. These checks cover the recorded trajectory and final geometry; synthetic regression tests separately exercise finite self/coincident gradients. No full intermediate-coordinate trajectory was retained.

`verify_moignard_probe_record.py` reread both NPZ files, checked hashes and initialization, reconstructed final objectives, and recomputed all aggregate/population metrics and stability diagnostics without fitting. Maximum loss error was 1.11e-16; every neighborhood and stability result matched exactly. Two same-seed executions also produced exactly equal saved coordinates and traces.

- Active test suite: 654 passed, 41 warnings, including write-once output protection and updated CLI containment regressions. The suite emitted 41 warnings; the saved test log retains them. No test failed.
- `git diff --check` passed. Current core model files match both HEAD and the PR #3 merge. Both historical probes and all other preexisting tracked docs JSON files are unchanged.
- CodeRabbit reviewed four tracked files and flagged the helper as missing because it did not include untracked files in its review. The helper exists in this worktree and is included in the delivery bundle; both drivers and all tests imported it successfully. The finding is resolved in the complete local file set, but this is not an automated review of the new untracked files. New helpers/results were inspected and executed directly.
- Semantic indexing was unavailable because its embedding backend exhausted MPS memory. Source inspection and searches used local files; this did not affect the CPU experiments.

## Reproduce

From the new worktree, select an existing Python environment containing the dependencies declared in pyproject.toml plus the torch/benchmark extras. Use fresh output names. The original names below intentionally refer to the new dated rerun records and must not be reused if those files already exist.

```sh
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Use the chosen environment's Python; do not install packages silently.
python examples/run_moignard_lorentz_probe.py --source data/inputs/Moignard2015.csv --output docs/moignard-lorentz-probe-NEW-RUN.json --gradient-ablation
python examples/run_moignard_curvature_probe.py --source data/inputs/Moignard2015.csv --output docs/moignard-curvature-probe-NEW-RUN.json
python examples/verify_moignard_probe_record.py --source data/inputs/Moignard2015.csv docs/moignard-lorentz-probe-NEW-RUN.json docs/moignard-curvature-probe-NEW-RUN.json
python -m pytest
```

Download the pinned `SOURCE_URL` from `src/sc_poincare/_probe_record.py` into the active input path if absent. Drivers verify both checksums before preparing data. Their original hard-coded /tmp input path is no longer required. Inputs remain ignored and are not vendored.

## Records and limits

- [Corrected Lorentz and gradient-only control](moignard-lorentz-probe-rerun-2026-10-30.json), with same-stem NPZ.
- [Curvature fits and evaluation-only control](moignard-curvature-probe-rerun-2026-10-30.json), with same-stem NPZ.
- [Preservation and coordinate-verification audit](moignard-pr3-validation-audit-2026-10-30.json).

Independent data, additional seeds, longer optimization budgets, marker continuity, branching order, pseudotime and unseen rare/outlier populations were not validated here. All four fitted conditions retain `biological_validation=false`; promotion remains false. No claim of a new biological method or a better default follows from these runs.
