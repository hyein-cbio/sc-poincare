# Legacy, historical examples and closed research boundaries

## What remains active

MU fixed/local-radius, explicit RFA/conditional-KL/union options, forest tau and
spectral controls, Euclidean controls, initialization jitter and current
observational diagnostics remain active. Not choosing an option as the default
is not the same as retiring it. No algorithm, optimizer or scientific default is
changed by this containment work.

## LEGACY blocks: retained, not silently removed

- `src/sc_poincare/_legacy_display.py`: all deprecated jitter/fan/repel mechanics
  and wrappers lie between BEGIN/END LEGACY DISPLAY IMPLEMENTATION markers.
- `tl.py`: the existing callable aliases and `__all__` additions lie in a marked
  BEGIN/END LEGACY DISPLAY EXPORTS block. Calls still emit the same warning,
  including zero-strength calls; initialization jitter is unaffected.
- `_legacy_numerics.py`: two retired NumPy probability-space wrappers,
  `_row_softmax_neg_distance` and `_sym_kl`, are kept in their own LEGACY block.
  `_fitting.py` no longer defines or reexports them. No production module imports
  that legacy numerical module. Direct tests preserve its ordinary arithmetic.
- `examples/benchmark_objectives.py`: the old probability-floor reference is
  explicitly contained in a LEGACY FLOOR CONTROL block and reached only through
  `--include-legacy-floor`. It remains an opt-in historical control, not the
  corrected objective or its default. The useful current benchmark is retained.

`_torch_sym_kl` is not an unused function: historical examples and independent
reference/gradient tests still call it. It is not removed with the unused NumPy
wrappers. Neither are substantive mathematical tests deleted as cleanup.

## Historical examples are not normal examples

The old geodesic fitter, dataset-specific recipe runner and their dedicated
objective/chunk/input tests all live under `examples/historical/` (tests in its
`tests/` subdirectory). No root-path redirect/shim is left behind. Default
`pytest tests` does not discover that historical-specific suite (it still checks
boundary/acknowledgement contracts). CI invokes the historical arithmetic tests
in a separate explicit step on tiny synthetic inputs. Recipe execution requires
`--acknowledge-historical` before reading a CSV or fitting; this flag does not
reopen a closed research input. Sibling loading is explicit and independent of
the caller's current working directory.

Current commands are documented separately from historical reconstruction.
See [historical usage](../examples/historical/README.md). Frozen old numerical
policies/source commits remain part of the evidence; using the new file location
with current dependencies is not exact replay of every old artifact.

## Closed research is preserved and removed from ordinary replay paths

The following completed local directories were moved byte-for-byte, not copied
or scientifically edited:

| Former path under `data/` | Current path under `data/research_archive/` |
|---|---|
| `graph_diagnostics_integration/algorithm_followup/` | `graph_diagnostics_integration/algorithm_followup/` |
| `graph_diagnostics_integration/stress_validation_34/` | `graph_diagnostics_integration/stress_validation_34/` |
| `public_tree_2026/research_mechanisms/` | `public_tree_2026/research_mechanisms/` |

The destination root contains `data/research_archive/CLOSED_RESEARCH.json`,
covering every relocated subtree, in addition to the old-location markers.
An inventory/relocation map records **1814 preserved files**, their sizes/hashes
and old/new roots. Their old locations now contain only ARCHIVED.md and
CLOSED_RESEARCH.json, not runnable Python scripts. Scientific hashes, states,
results and original logs stay unchanged. Historical absolute paths in those
records remain provenance, resolved by `data/research_archive/RELOCATION.json`,
not blindly rewritten. Current result-index pointers are updated separately.

The immutable Astra handoff remains at `data/astra37_handoff/`; no source,
checkpoint or transfer-manifest byte inside its capsule is edited. A closure
marker sits outside the capsule payload at the handoff root. Audit/evaluation
reads and the original transport checks remain possible.

### Enforced maintained-CLI convention

`_research_policy.ensure_active_paths` checks lexical and symlink-resolved
ancestors for CLOSED_RESEARCH.json. Maintained fitting/demo/benchmark CLIs check
inputs, optional resume states, output paths and declared supervisor job inputs
before starting training or creating outputs. The Scanpy demo also checks
`sc.settings.figdir`, its secondary figure destination. A symlink does not silently bypass
the convention. No override/environment flag is added to normal commands.

Read-only evaluators do not apply the fitting prohibition. To conduct a new
approved experiment, establish a new protocol and active fitted inputs/output;
do not remove archive markers or relaunch closed queues as an implicit resume.

This is an operational exclusion, **not an OS sandbox**. Array-based APIs cannot
infer the filesystem origin of an array, and deliberate direct execution of
immutable archived Python is outside this convention. We do not modify historical
source bytes to inject guards or claim that arbitrary code execution is impossible.

## Checkpoint source boundary

Moving the unused functions out of `_fitting.py` changes that whole-file source
fingerprint, which MU resume guards record. Old-source states therefore still
need their original implementation/environment; no digest is rewritten,
whitelisted or declared compatible merely because calculations are equal.
The new `_legacy_numerics.py`, `_research_policy.py` and historical examples are
outside the optimizer's source-hash set. Future removal of those isolated
modules need not change numerical source files solely for cleanup.

Small fixed before/after checks separately verify that RFA and both MU recipes'
coordinates, initialization and losses are unchanged. Such numerical checks are
not claims of old-checkpoint source compatibility.

## Git and preservation

All archived research, closure markers, inventories, local results and raw
capsules stay under ignored `data/`. They are not package wheel contents or Git
push targets. Versioned legacy/historical code remains visibly contained;
failed experimental training rules were not imported into production.
No scientific result, raw data or original capsule is deleted by this cleanup.
Local preservation still does not constitute independent off-machine backup.

## Local verification

- Active/boundary tests: **527 passed** (two existing exact-zero-target warnings).
- Explicit historical arithmetic suite: **10 passed**; combined total **537**.
- Torch-free lane: **318 passed / 61 optional skips**.
- Installed wheel: retained legacy calls/warnings, active MU/RFA, new-source
  resume/H5AD and rejection of the real closed capsule input passed.
- Frozen ten-update before/after RFA and both MU recipes matched exactly.
- Original Astra transport: **2810 file hashes passed** after containment.
- Archive relocation: **1814 original file sizes/hashes** reverified at new roots.

CLI checks cover closed inputs/resume/output roots, symlink aliases, derived
benchmark outputs and Scanpy's secondary figdir before fitting. No expensive
archived experiment was restarted. These are local checks, not a new remote CI
result or old-source checkpoint-compatibility guarantee.
