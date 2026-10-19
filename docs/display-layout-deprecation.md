# Deprecated display jitter, fan/spread and repel

## Decision and scope

Keep the following APIs callable for now, but deprecated:

- `tl.poincare_jitter(..., jitter=...)`
- `tl.poincare_fan(..., spread=..., groupby=...)`
- `tl.poincare_repel(..., repel=...)`

Each call emits `DisplayLayoutDeprecationWarning`, a visible `FutureWarning`,
including zero-strength identity calls. There is no import-time warning and
normal fitting does not call these helpers. Public signatures, default output
keys, copy behavior and coordinate transformations are retained until removal.
No removal release/date is promised; removal is a separate explicit cleanup.
These APIs remain callable within explicit BEGIN/END LEGACY implementation and
export blocks; the [containment policy](research-containment.md) also isolates
retired numerical helpers and historical examples.

**Initialization jitter is not deprecated.** `GraphStressConfig.jitter` remains
0.0002 and still perturbs the initial disk coordinates before tangent Adam.
The RFA initialization policy is unchanged too. Repel joins the same deprecation:
its API remains callable, but its radial displacement is only a display layout.

## One disposable implementation file

`src/sc_poincare/_legacy_display.py` owns all three public wrapper implementations,
their angular/radial algorithms, fan-only circular helpers and warning category.
Only the isolated deprecated-export block in `tl.py` imports it. Core geometry,
AnnData fitting and initializers do not import it. The legacy module depends on
core geometry, not vice versa.

The former private `_disk.jitter_angles`, `_disk.fan_dendrogram_angles`,
`_disk.repel_points` and `_anndata` display-wrapper imports are removed, not
kept as additional redirect shims. Backward compatibility for those private
locations and old checkpoint hashes is deliberately not maintained.

These are layouts, not improvements to the learned objective. Jitter changes
individual angles; fan/spread rotates label groups. Both can change pairwise
hyperbolic distances, neighbors and angular lineage results despite preserving
radius/current-origin distance. Repel changes radii and can therefore alter
origin-distance pseudotime as well as pairwise distances. The standard warning recommends raw coordinates
plus marker size/alpha, density or zoom rather than artificial coordinate spread.
If a zero-strength call was used only to copy an embedding, use the array's
`.copy()` directly rather than a deprecated layout API.
Old stored layout coordinates remain plain readable obsm arrays; loading a file
does not require the retired helper that originally produced them.

## Efficient source/checkpoint bookkeeping

Extraction changes the whole-file `_disk.py` hash even though the retained
geometry and fitting calculations are unchanged. MU checkpoint validation hashes
that file. Therefore earlier bare/AnnData optimizer checkpoints may be rejected
with a source-mismatch error under this source revision. This is intentional.

- Do not rewrite checkpoint hashes, whitelist old source sets or pretend they
  were produced by the new source.
- Completed coordinates, losses and historical study records retain their old
  provenance; their existence is not a claim of current-code resume compatibility.
- Use the original source revision/environment if old-state recovery is essential,
  or start a new run under current code. New checkpoints validate normally.
- The compact local `display_layout_deprecation.json` records before/after source
  hashes, the affected checkpoint-source file and small parity checks. The repel
  extension records its subsequent `_disk.py` hash boundary in that same record. No
  duplicated datasets, source archives or full experiment reruns are required.

Future removal of this legacy module and its `tl.py` export block need not edit
`_disk.py` again. The current source-hash set does not include `tl.py` or the
legacy module; deleting the deprecated feature need not invalidate new numerical
checkpoints solely because of those files.

## Future removal checklist

1. Delete `src/sc_poincare/_legacy_display.py`.
2. Delete the marked deprecated import/`__all__` block at the end of `tl.py`.
3. Delete `tests/test_legacy_display.py`, remove its explicit Torch-free pytest
   path in `.github/workflows/tests.yml`, and update the deprecation docs/README.
4. Keep initialization-jitter tests in `tests/test_graph_stress.py`, core geometry,
   RFA/MU fitting and root isometries intact.
5. Run package import, fitting, installed-wheel and full regression checks.

A removal smoke test copies the package, removes just the module/export block,
and verifies import, MU preparation and explicit RFA fitting. Dependency tests
prevent fitting/core files from growing imports back into the legacy module.

## Verification boundary

Combined extraction checks: **443 local tests passed** (two existing exact-zero-target
warnings); the Torch-free lane had **231 passed / 54 optional skips**. An actual
installed wheel verified warning-free normal MU fitting and warnings from the
retained display APIs, with initialization jitter still 0.0002. These are local
checks, not a claim that remote CI ran.

Small fixed pre/post extraction checks compare MU fixed/local-radius initial
coordinates, returned coordinates and loss traces, RFA coordinates/loss, and the
moved display algorithms, including positive-strength repel. They are numerically
exact in the tested environment.
Independent geometry, warning/caller-location/copy and source-mismatch tests
remain separate. This code organization does not establish biological quality,
new large-data results, or old-checkpoint compatibility.
