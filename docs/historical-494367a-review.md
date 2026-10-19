# Scoped review of commit 494367a

This review covers only `494367a68e3940d228e10f82c7b45dff06c6405e`
(`docs: publish generic objective ablations and validation limits`). The
diff adds 377 lines in five files and does not touch the later MU fitter:

- `examples/benchmark_objectives.py` (new)
- `tests/test_benchmark_objectives.py` (new)
- `docs/objective-validation.md` (new)
- `docs/reference-parity.md` (10 lines)
- `README.md` (5 lines)

It was read with `git show 494367a`. Later containment of the legacy floor,
closed-research path checks, and the landmark reference check are outside
this scope. A review of the current working tree does not replace this one.
This is not a third-party human publication audit.

## What the commit actually binds

The driver reads one local CSV, hashes those bytes, drops one label column
before PCA or fitting, and runs the corrected symmetric KL and conditional KL
from a shared spectral initialization. The optional `1e-12` floor is a separate
function reached only with `--include-legacy-floor`. The dense-array budget
stays on (`--max-dense-mib` defaults to the shared 512 MiB guard). Outputs
refuse to overwrite. The notes in the JSON say that equal epoch limits are not
convergence, that losses are not comparable across objectives, and that the
anchor graph is not a biological holdout.

The commit's own test checks three facts: the three modes share initial
coordinates, changing the label column does not change coordinates, and the
legacy floor's chunk size does not change its two-epoch trajectory.

## Findings inside that diff

- Legacy and corrected traces both append only after epoch 0 (`if epoch: trace.append`
  in the legacy loop; `if epoch > 0` in `fit_disk_embedding`). `epochs_run` is
  therefore the number of updates, and epoch 0 can still win `best`. That
  matches the fitter. The legacy path hard-codes `stop_reason='epoch_limit'`
  even though it always walks the requested epoch count; the corrected path can
  report `patience` because the driver passes `patience=epochs` and `min_delta=0`.
- The process RSS field is `ru_maxrss` for the whole process. The commit's note
  already says it includes earlier modes, imports, and evaluation, and that the
  dense budget is not an RSS cap.
- `spearmanr(...).statistic` is written through `float` with no finite check.
  A constant distance vector becomes `NaN` in the JSON. The smoke test does not
  cover that input. This remains in the maintained driver; it is a reporting
  failure on a degenerate evaluation graph, not a change to the fitting objective.
- Labels are stored in the NPZ and are not arguments to `fit_disk_embedding`
  or the legacy loop. The label-swap test locks that for a 12-by-3 synthetic CSV.

No numerical change was made as part of recording this review.
