# Paul landmark / held-out branch: critical review follow-up

Reviewed branch: `cursor/paul-landmark-heldout-1d40` at `bf52ac7`.
Integration target: `main` at `6569b57`. Their merge tree has no textual
conflicts. No merge, commit, push, biological rerun, or default change is
part of this follow-up.

## Corrections

- **Held-out fit acceptance:** a finite point inside the unit disk is not
  proof of convergence. Reject unsuccessful/nonfinite optimizer results and
  points in the objective's numerical boundary penalty region
  (`radius_squared >= 1 - 1e-8`). Preserve the training coordinates. Regression
  tests cover an actual iteration-limit failure, reported success in the
  penalty region, nonfinite/outside points, and a valid nonduplicate fit.
- **Landmark diagnostic integrity:** reject fractional, boolean, empty,
  duplicate, and out-of-range landmark indices; reject nonreal, negative,
  nonfinite, incorrectly shaped, or nonzero-self distances. Revalidate the
  prepared setup and its positive dense budget before allocating tensors.
  Cross-check gradients at neighboring row-block sizes rather than forcing
  a full-N autograd batch.
- **Scope of the gradient conclusion:** the saved Moignard schedule tested
  32, 128, 512, 2048, and 3934 landmarks. Only its tested all-cell case met the
  tolerance. This is not a theorem that every partial set fails. With N−1
  landmarks there is only one omitted cell and no omitted nonself pair.
  Given exact distances and the exact `global_mean`, both reductions recover
  the exact gradient in the diagnostic. The matrix-free preparation uses a
  column mean that need not equal the exact mean at N−1. Synthetic regression
  tests cover that distinction and the counterexample. It does
  not establish a useful sparse approximation: all pairwise work remains.
- **Paul PCA reconstruction:** matching recomputed scores are valid, not a
  reason to abort. Both matching and nonmatching reconstructions retain the
  independently pinned `features.npy` bytes; the reconstruction is diagnostic
  only. Synthetic driver tests cover both cases without fitting.
- **Best checkpoints:** regression tests reconstruct the recorded objective
  from the returned best coordinates, including zero epochs, and verify that
  best/current arrays do not alias. Saved optimizer-screen deltas and
  guardrails are checked against the driver without rerunning the screen.
- **Optimizer-screen schema:** preserve `populations_k15` inside each arm's
  scores, matching the committed record. Previously the driver removed both
  that field and its duplicate arm-level copy, losing all population scores
  on a fresh run. A stubbed driver test checks the complete written arm schema
  and overwrite refusal without executing any fits.

Existing experiment JSON/NPZ files and checkpoint-hashed model sources are
unchanged. These fixes do not retrospectively regenerate or reinterpret saved
measurements as a new fit.

## Replaying the existing Moignard records after integration

The Moignard verifier intentionally hashes its implementation sources.
This branch changes `controlled_probe.py` and `evaluation.py`, so verifying
the old rerun records directly with the integrated sources fails closed with
`Source SHA256 mismatch`. That is expected provenance enforcement, not a
reason to edit the archived hashes or disable checks.

Use the complete source snapshot at `6569b57` to replay those records: it
contains both the original fitting/evaluation sources and the follow-up
verifier that keeps integrity checks active under optimized Python. For
example, from the repository root, with the project dependencies available:

```sh
snapshot=$(mktemp -d)
git archive 6569b57 | tar -x -C "$snapshot"
# Supply an absolute path to the original, checksummed public CSV.
source_csv=/absolute/path/to/Moignard2015.csv
(
  cd "$snapshot"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src python -O examples/verify_moignard_probe_record.py \
    --source "$source_csv" \
    docs/moignard-lorentz-probe-rerun-2026-10-30.json \
    docs/moignard-curvature-probe-rerun-2026-10-30.json
)
```

The snapshot is temporary source material, not a new worktree, experiment, or
backup. Verification still requires the original numerical environment where
bitwise initialization/metric equality is part of the saved-record contract.
