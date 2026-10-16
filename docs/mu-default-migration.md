# Provisional MU default and RFA compatibility

The operator selected MU fixed as the provisional public default after the
bounded development study. This is an intentional numerical/default change,
not a claim that biological validation gates have been satisfied. Other changes
in this integration are opt-in or observational; no hidden dataset-specific
condition, bandwidth search or new approximation is introduced.

## Select the method explicitly

```python
import sc_poincare as dr

dr.tl.poincare_map(adata)                          # MU fixed, 1500 updates
dr.tl.poincare_map(adata, method="mu")             # same explicit selection
dr.tl.poincare_map(adata, local_bandwidth="local_radius")
dr.tl.poincare_map(adata, method="rfa")            # exact old defaults
```

For an existing pipeline, **add `method="rfa"`** to preserve its old fitter:

```python
dr.tl.poincare_map(adata, method="rfa", n_neighbors=15,
                    sigma=2., gamma=2., n_epochs=250,
                    learning_rate=.01, burn_in_epochs=30,
                    min_delta=1e-4, patience=40, max_grad_norm=2.5)
```

RFA keeps its self-inclusive symmetric KL, spectral initialization and recorded
fallback, graph mode, numerical policies, root/copy/storage semantics and
small-data NumPy fallback. Conditional symmetric KL and union-only RFA are still
explicit RFA options. No existing stored embedding is recomputed by loading it.

The MU default instead requires Torch, uses 1500 fixed updates, supports 2-D disk
output and Euclidean input distances, and uses gamma .1*2/3 and learning rate
.02*2/3. The exact memory preflight is unchanged. These differences are deliberate;
select the old method rather than expecting identical coordinates or runtime.

## Parameters are not silently reinterpreted

| Parameters | MU | RFA |
|---|---|---|
| `use_rep`, `n_pcs`, `root`, `copy`, `key_added` | existing conventions | unchanged |
| `n_neighbors` | overrides config, caps at n_obs-1; records requested/effective | old cap and semantics |
| `gamma`, `learning_rate` | explicitly override GraphStressConfig | old gamma/lr |
| `n_epochs` | default 1500; TOTAL target for resume | default 250; old patience |
| `batch_size` | exact all-column row chunk, default 256 | old exact chunk policy |
| `random_state` | integer/None/Generator; effective integer seed recorded | old behavior |
| `mu_config`, `local_bandwidth` | explicit MU recipe selection | rejected as MU-only |
| `sigma`, `objective`, `graph_mode`, `burn_in_epochs`, `min_delta`, `patience`, `max_grad_norm` | rejected with legacy-method hint | unchanged meanings/defaults |
| `initial_coordinates` | use full Adam resume instead | coordinate-only NEW run |
| `resume`, `checkpoint_every` | identity-bound full state | not optimizer resume |
| `metric`, `n_components` | Euclidean features / 2-D only | old other metrics/dimensions |

Omitted shared settings use method-specific defaults (`method_default` in the
signature). Explicit settings override `mu_config`; the caller's frozen config
is not mutated. Effective seed and full MU recipe are recorded. A copied or
pickled default sentinel preserves omission semantics. MU permits `n_epochs=0`
for an initialization/state snapshot; this is not a trained result. Demo CLI
fits require positive update counts.

Old calls supplying RFA-only knobs without a method must be updated. Raising an
informative error is intentional: silently ignoring `sigma` or interpreting a
RFA patience/clipping setting as an Adam option would be more dangerous. Shared
knobs such as explicit gamma also change the MU recipe; pin the method for
reproducibility rather than relying on defaults.

## Metadata, graphs and losses

Existing `obsm['X_poincare']`, `obsp['poincare_connectivities']` and
`uns['poincare']` remain. `key_added` changes all prefixes consistently; `copy=True`
returns a separate AnnData as before. New records have schema_version=2 and an
explicit method tag in the record/params.

RFA connectivity values remain the old Gaussian affinities. MU connectivity is
**the symmetric mean of its row-normalized union teacher**, not the old Gaussian
matrix. The directed teacher is separately stored in
`obsp['poincare_local_teacher']` (or the selected prefix). A successful switch
back to RFA removes only that previously owned MU teacher, not unrelated graphs.

MU diagnostics are method-specific: no fabricated RFA self-mass is stored. The
record includes global target/local teacher/initializer/source hashes, kernel
policy, effective neighbor count and an independently reconstructed unweighted
topology. Cutoff ties are explicit; reconstructed topology is not claimed to
certify exact fitted tie choices on every numerical backend. Weighted teacher
components are observed separately; global distance targets can retain structure
even when local-radius weights drop some teacher bridges.

For both AnnData methods, `loss` contains post-update evaluations, with epoch 0
in `initial_loss`. `best_epoch=0` remains eligible. MU low-level loss_trace still
includes epoch 0; its established API/checkpoint format is unchanged. Best and
current losses/epochs, raw-best versus display coordinate hashes and source cell
order are recorded. An epoch limit is not a convergence or biological verdict.

## Safe progress and MU resume

`progress_callback` receives `tl.PoincareProgress`: method, epoch, current loss,
best loss/epoch, independent raw `current`/`best` coordinate copies, initialization
provenance and optional checkpoint. Aliases `coordinates`/`best_coordinates`
refer to the raw snapshots. A callback never receives root-translated display
coordinates. Returning a value is ignored; exceptions propagate before result
slots are written. Mutating copies has no effect; externally changing fitted
features/cell identity during the call is detected, not silently relabeled.

```python
import os
import torch

saved = {}
def checkpoint(p):
    if p.method == "mu":
        temporary = "trusted_mu.pt.tmp"
        torch.save(p.checkpoint, temporary)
        os.replace(temporary, "trusted_mu.pt")
        saved["checkpoint"] = p.checkpoint

dr.tl.poincare_map(adata, n_epochs=100, local_bandwidth="local_radius",
                    checkpoint_every=25, progress_callback=checkpoint)
state = torch.load("trusted_mu.pt", weights_only=True)
dr.tl.poincare_map(adata, resume=state, n_epochs=200)
```

Resume requires an explicit total epoch target greater than the saved epoch.
If omitted, the MU recipe, integer seed and representation cap are restored from
the checkpoint; explicitly conflicting settings fail rather than retune. The
original unique/nonempty obs_names, order and exact fitted features must match.
Duplicate-name inputs can fit, but cannot be certified for high-level resume;
make IDs unique before the original run if recovery matters. IDs must honestly
describe upstream identity; checksums cannot certify biological cell matching.

The outer AnnData checkpoint binds observation/feature provenance to the existing
bare Adam-state integrity hash. It is not stored inside H5AD uns. The low-level
`experimental.fit_graph_stress(..., resume=state)` format/contract is unchanged.
At default-switch integration its four source files were preserved; the later
[display-layout extraction](display-layout-deprecation.md) intentionally changes
`_disk.py`'s whole-file hash, so pre-extraction states are not auto-migrated.
The high-level wrapper also refuses bare states without identity evidence.
Matching numerical source/Torch/runtime policy still matters. The subsequent
[Astra control follow-up](astra-followup.md) also changes `_proximity.py` and
`_fitting.py` source hashes; default numerical parity does not waive those checks.

RFA progress has checkpoint=None. Its coordinate warm start is a new schedule,
not exact optimizer/patience recovery. Do not pass a root-translated display to
an optimizer. Atomic local saves are not durable external backup; load only
trusted states and follow the [recovery protocol](benchmarking.md).

## Derived annotations and old files

```python
from sc_poincare.diagnostics import result_status
print(result_status(adata))
```

Root helpers record display isometries without changing raw training checkpoint
identity. Pseudotime/lineage helpers record the coordinate checksum they used and the
checksum of the values they generated.
The read-only status reports `current`, `stale`, `missing` or `unknown`; it never
fabricates freshness for old files. Re-fitting keeps pre-existing observation
columns for compatibility but does not recompute them: rerun downstream helpers.
External coordinate changes and subsets can invalidate checksum evidence, not
necessarily the biological interpretation of a copied annotation. Inspect the
context rather than deleting data automatically. Display jitter/fan/repel are
[deprecated](display-layout-deprecation.md), isolated non-isometric layouts and
must not be scored as learned embeddings. Initialization jitter is retained.

## Evidence and limitations

The migration is covered by high-level-to-low-level MU parity, persisted AnnData
Adam recovery, H5AD round trips, independent callback snapshots and legacy
RFA regressions. Frozen pre-switch RFA checks match coordinates, loss, sparse CSR
and old parameter values exactly for symmetric/conditional KL, union and
cosine/3-D conditions. Tests are not independent biological validation.

Default-switch validation before the display extraction: **432 local tests passed** (two existing exact-zero
warnings); the isolated Torch-free path had **222 passed / 52 optional skips**.
An actual installed wheel was smoke-tested outside the source tree for default
MU fitting, H5AD/annotation integrity and identity-bound resume; explicit RFA
also worked in a separate Torch-free installed environment. These are local
software checks, not a claim of remote CI execution or biological accuracy.

The [MU study](mu-scaleup-study.md) and [Astra adoption status](astra-adoption-status.md)
separate the operator's provisional default choice from remaining evidence gaps.
