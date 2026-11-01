# Moignard saved-record verifier follow-up — 2026-11-01

Base: `main` at `231fec9fdc6bf299d3f8d1e90fc15bccb9a3792e` (merged PR #4).
Delivery branch: `fix/moignard-record-verification-20261101`. No PR was requested
for this follow-up; the merged PR's review threads have not been resolved.

## Confirmed defects and changes

- [Unchecked saved traces](https://github.com/hyein-cbio/sc-poincare/pull/4#discussion_r4429548966):
  the baseline verifier accepted an edited intermediate JSON loss for all four
  conditions, even with endpoints and summary stability diagnostics unchanged.
  It now reads the independent NPZ traces, checks finite, nonempty 1D shape and
  exact JSON agreement, checks the initial/final loss summaries, and recomputes
  stability from the NPZ trace. Successful reports include `loss_trace_exact`.
- [Checks removed by Python optimization](https://github.com/hyein-cbio/sc-poincare/pull/4#discussion_r4429548972):
  the baseline also accepted incorrect sidecar/source/initial hashes, population
  metrics and stability summaries when its assertions were optimized away.
  All verifier checks now explicitly raise `ValueError` on mismatch, including
  initialization, legacy curvature evaluation and final objective checks.

| Condition | Coordinates | Independent NPZ trace |
| --- | --- | --- |
| Lorentz | `current` | `loss_trace` |
| Gradient-only ablation | `legacy_gradient_current` | `legacy_gradient_loss_trace` |
| Curvature 1.0 | `current_c1_0` | `loss_trace_c1_0` |
| Curvature 0.25 | `current_c0_25` | `loss_trace_c0_25` |

## Validation

The new tests build small, deterministic records from fixed coordinates and
synthetic loss histories. They do not call any optimizer. Each condition is
tested with JSON-only trace changes, NPZ trace changes with an updated sidecar
hash, malformed trace shape, hash mismatches, initialization changes, population
and stability edits, endpoint edits, and a coordinated trace/final-loss edit
that only the independently reconstructed objective can reject. Legacy
curvature metrics are tested separately. Fresh subprocesses repeat all valid
and invalid cases under both `python -O` and `PYTHONOPTIMIZE=1`, using explicit
checks in the subprocess harness itself.

- Before the fix: the targeted reproduction had 4 valid cases pass and all 6
  integrity regression cases fail (four accepted JSON edits and two optimized
  subprocesses accepting tampering). A separate optimized baseline diagnostic
  confirmed all four conditions accepted each of the three hash edits,
  population edits and stability edits.
- After the fix: **87 passed** across `test_verify_moignard_probe_record.py`,
  `test_probe_record.py`, and `test_controlled_probe.py`; this includes **56 new
  tests** in the verifier-specific module.
- Active suite: **706 passed, 3 skipped, 1 failed**, reproduced on a second run.
  The failure is the existing
  `tests/test_mu_study.py::test_supervisor_runs_owned_child_and_collects_result`:
  `observed_peak_rss_bytes` is zero. It passes in isolation. The untouched base
  commit has the same failure in its full suite (**650 passed, 3 skipped,
  1 failed**). No supervisor changes are included here.
- `git diff --check` passed.
- Historical tiny synthetic suite: **10 passed**.
- Both committed rerun NPZ SHA256 hashes match their JSON records. All four
  committed traces have shape `(21,)` and exactly match the JSON traces.

Commands (with the project's dependencies, including Torch, installed):

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests/test_verify_moignard_probe_record.py tests/test_probe_record.py tests/test_controlled_probe.py
python -m pytest
python -m pytest examples/historical/tests
```

## Existing cross-environment verification limit

A full `python -O examples/verify_moignard_probe_record.py --source
data/inputs/Moignard2015.csv ...` attempt on the committed records stopped at the
existing exact initializer comparison. The untouched base verifier fails at
the same comparison. In this Linux environment, the recreated initial array
differs from the macOS record by at most `5.45657269e-15` (7,667 of 7,868 values
are not bitwise equal). This is consistent with numerical environment
differences; it is not a newly introduced validation failure. The exact check
has not been relaxed, and no claim of complete saved-coordinate reverification
in this environment is made.

Local versions: Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, pandas 3.1.0,
scikit-learn 1.8.0, Torch 2.15.0+cpu. The saved record reports macOS arm64,
NumPy 2.5.4 and Torch 2.14.1. The pinned public CSV download passed its existing
hash checks. It can be downloaded again from `SOURCE_URL` in
`src/sc_poincare/_probe_record.py`; it is not a new experiment artifact.

## Preservation and handoff

All preexisting experiment JSON and NPZ files, package source files, and both
Moignard run drivers are unchanged from the base commit. No Moignard fitting
was repeated. The only delivered changes are the verifier, its regression
tests, and this note.

The branch preserves the implementation, reproduction method, validation
results and limitations. Deleting the originating chat does not lose these
deliverables. The conversation, raw local command logs, temporary test records,
downloaded CSV and temporary Python environment are not part of the delivery;
their relevant findings and reproduction commands are recorded above.
