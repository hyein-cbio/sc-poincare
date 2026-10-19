# sc-poincare

Poincaré-disk embeddings for hierarchical single-cell structure, with a Scanpy /
AnnData interface.

**The provisional default is now MU**: a mutual-graph global distance target plus
a union-graph local distribution, fitted with tangent-coordinate Adam. The
previous corrected Relative Forest Accessibility (**RFA**) implementation is
retained explicitly as `method="rfa"`, with its original numerical defaults.
This default choice is for exploratory use; it is not a claim of universal
biological or hyperbolic-specific superiority.

## Install

```bash
pip install -e '.[torch]'
```

MU requires PyTorch. Without Torch, explicit `method="rfa"` retains the slow
NumPy finite-difference fallback for at most 128 cells. Install
`pip install -e '.[torch,benchmark]'` for study/evaluation drivers and their
resource supervisor. Scanpy is an optional extra: `.[scanpy]`.

## Scanpy-style usage

```python
import scanpy as sc
import sc_poincare as dr

adata = sc.read_h5ad("cells.h5ad")
# Choose appropriate preprocessing explicitly for your input.
# Do not normalize supplied PCA scores as if they were raw counts.
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50)

dr.tl.poincare_map(adata, n_neighbors=15)  # method="mu", fixed local bandwidth
# Optional second MU recipe:
# dr.tl.poincare_map(adata, local_bandwidth="local_radius")

dr.pl.poincare_map(adata, color="cell_type")
dr.tl.poincare_recenter(adata, root="HSC_1")
dr.tl.poincare_pseudotime(adata)
dr.tl.poincare_lineages(adata, n_lineages=4)
sc.pl.embedding(adata, basis="poincare", color="poincare_pseudotime")
```

High-dimensional inputs expect `X_pca` unless `use_rep` is explicit. Labels and
roots do not enter MU fitting or initialization; a root is an optional post-fit
isometry. Pseudotime is a root-distance diagnostic, not validated biological time.

### Use the previous method unchanged

```python
dr.tl.poincare_map(adata, method="rfa", n_neighbors=15, sigma=2., gamma=2.)
# Previous experimental conditional KL and union-only RFA remain available:
# dr.tl.poincare_map(adata, method="rfa", objective="conditional_symmetric_kl")
# dr.tl.poincare_map(adata, method="rfa", graph_mode="union")
```

| Setting | MU default | Explicit RFA legacy default |
|---|---|---|
| Objective | relative global stress + local cross-entropy | self-inclusive symmetric RFA KL |
| Graphs | mutual global / union local | mutual, stitched by component MST |
| Initialization | classical scaling + fixed-seed jitter | spectral, with recorded fallback |
| Optimizer | tangent Adam | Riemannian gradient updates |
| Updates | 1,500, fixed limit | 250, patience 40 after burn-in 30 |
| Gamma | .1*2/3 | 2.0 |
| Learning rate | .02*2/3 | .01 |
| Local bandwidth | input-derived fixed rule; optional local-radius | sigma 2.0; optional `"median"` |

**Pin the method in reproducible pipelines.** The default switch intentionally
changes coordinates, runtime and Torch requirements. RFA-only arguments such
as `sigma`, `objective`, `graph_mode`, `patience`, `burn_in_epochs`, `min_delta`
and `max_grad_norm` require `method="rfa"`; MU never silently ignores or
reinterprets them. RFA also retains its other metrics/dimensions. MU currently
uses Euclidean input distances and 2-D disk output. See the
[migration and compatibility guide](docs/mu-default-migration.md).

## Storage and provenance

Both methods keep the existing slots and `copy` / `key_added` conventions:

| Slot | Contents |
|---|---|
| `adata.obsm["X_poincare"]` | learned disk coordinates, optionally root-translated |
| `adata.obsp["poincare_connectivities"]` | symmetric sparse graph, with method-specific semantics |
| `adata.uns["poincare"]` | method, effective settings, post-update loss trace and provenance |
| `adata.obs["poincare_pseudotime"]` | optional distance from the current root/origin |
| `adata.obs["poincare_lineage"]` | optional angular-clustering labels |

For MU the connectivity slot is the symmetric mean of its row-normalized union
teacher; the exact directed teacher is stored separately in
`obsp['poincare_local_teacher']`. It is **not** the old RFA Gaussian-affinity
matrix. RFA retains its original affinity values and self-mass diagnostics.
`uns['poincare']['params']['method']` makes the distinction explicit.

MU provenance records input/cell-order hashes, requested/effective neighbor
counts, raw versus display coordinates, target/teacher/source hashes and best
versus current losses. `loss` excludes epoch 0 for the AnnData interface;
`initial_loss` records it separately and `best_epoch=0` is eligible.

```python
from sc_poincare.diagnostics import graph_topology, result_status

print(adata.uns["poincare"]["params"])
print(result_status(adata))  # current / stale / unknown derived annotations
# Optional input-only topology reconstruction; no fitting or kernel tuning:
# print(graph_topology(adata.obsm["X_pca"], n_neighbors=15))
```

Graph reconstruction labels cutoff ties explicitly; it is not a certificate of
identical fitted tie choices on every numerical backend. Old H5AD embeddings
remain readable, with missing provenance marked unknown rather than fabricated.
Re-fitting does not delete/recompute old observation columns; rerun downstream
helpers after changing the embedding. Root helpers update display provenance,
not optimizer state.

## Progress and verified MU recovery

```python
import torch

latest = {}
def checkpoint(progress):
    # Independent raw training snapshots, never root-translated display points.
    latest["checkpoint"] = progress.checkpoint

dr.tl.poincare_map(adata, n_epochs=100, checkpoint_every=25,
                    progress_callback=checkpoint)
torch.save(latest["checkpoint"], "trusted_mu_checkpoint.pt")

saved = torch.load("trusted_mu_checkpoint.pt", weights_only=True)
dr.tl.poincare_map(adata, resume=saved, n_epochs=200)  # TOTAL target, not +200
```

Persist files atomically and externally when needed; a callback is not a durable
backup. Resume requires the original unique `obs_names` / order, fitted features,
MU parameters, seed/chunk policy, numerical source and Torch environment. Callback
exceptions abort before output slots are changed, and mutating snapshot copies
does not modify the fit. Feature/cell changes made externally during a callback
are detected. Torch optimizer states are never stored in H5AD `uns`.

RFA progress uses the same `PoincareProgress` envelope but `checkpoint=None`.
Its optional `initial_coordinates` starts a **new** run; it does not restore
optimizer, patience or burn-in state. Bare low-level MU states keep their
existing `experimental.fit_graph_stress(..., resume=state)` contract; the
AnnData wrapper adds an identity-bound outer checkpoint schema.

## Numerical and resource limits

Both exact methods retain dense quadratic targets. The unchanged 512 MiB
`max_dense_bytes` preflight estimates four float64 N-by-N working arrays,
allowing at most 4,096 cells by default. It is **not** a process RSS ceiling.
Feature matrices, factorization/eigensolver work, autograd and imports add memory.
Row chunks retain every column and both endpoint gradients; they do not remove
quadratic storage or make the implementation atlas-scale.

Geometry/root translation never clips valid display points to the RFA training
margin; invalid coordinates or exhausted boundary precision raise. No hidden
feature scaling, artificial bridge floor or quality-dependent parameter choice
is introduced. MU rejects degenerate targets/empty teachers instead of inventing
a layout. A lower loss or an epoch limit alone does not prove convergence.

For details see [MU targets/recovery](docs/graph-stress.md),
[RFA formulas and numerical policy](docs/objectives.md), and the
[benchmark protocol](docs/benchmarking.md).

## Evaluation and evidence

`sc_poincare.evaluation.evaluate_neighborhoods` reports k15/30 population/macro
purity, size ceilings, input-neighbor recall and fixed-anchor feature-distance
rank. `compare_neighborhoods` checks inputs/IDs/policies before paired deltas.
Native Euclidean controls use `geometry="euclidean"`, never disk projection.
`evaluate_expression_neighborhoods` measures variance-normalized input-expression
reconstruction, not held-out prediction. `sc_poincare.visibility` separates
fixed-canvas marker-center crowding/width from intrinsic separation.

- [MU scale-up study](docs/mu-scaleup-study.md): both bandwidth rules, RFA and
  native Euclidean controls, full 2,730/3,934-cell results and explicit resource stops.
- [MU default / compatibility](docs/mu-default-migration.md): intentional default
  change, exact RFA path and verified AnnData checkpoint examples.
- [Astra adoption and remaining work](docs/astra-adoption-status.md): implemented,
  safe integration changes, unvalidated research and deliberately rejected ideas.
- [Deduplicated Astra follow-up](docs/astra-followup.md): opt-in RFA `forest_tau`
  and spectral budgets, per-gene/order diagnostics and fixed control probes.
  Defaults remain unchanged; current checkpoints use the new source fingerprints.
- [Latest Astra teacher/mechanism adoption](docs/astra-mechanism-adoption.md):
  teacher entropy/incoming-mass diagnostics and analytic tests; negative and
  cancelled mechanism evidence retained without new target-rule defaults.
- [Earlier union-only experiment](docs/graph-topology-results.md) and
  [visibility extension](docs/graph-visibility-study.md): historical, different
  objectives; neither is the MU algorithm's performance.
- [Historical objective validation](docs/objective-validation.md) and
  [reference-condition search](docs/reference-parity.md): separate numerical
  policies/recipes and their limitations, not default-MU validation.

Display usability is a separate endpoint, not proof of biological correctness.
Known development data, small init jitter and fixed budgets do not establish
independent-data performance, arbitrary-initialization robustness, branch order,
marker continuity or general superiority over the authors' implementation.

`tl.poincare_jitter`, `tl.poincare_fan(spread=...)` and `tl.poincare_repel`
are **deprecated** display layouts. They remain callable with a visible FutureWarning and live in one
[removable legacy module](docs/display-layout-deprecation.md), outside core
geometry/fitting. Initialization `GraphStressConfig.jitter` remains unchanged.
Do not score artificial layouts
as learned embeddings. Exact few-thousand-cell results are not full Planaria validation.

Legacy APIs and old probability wrappers are explicitly contained; old selected
recipes live only under `examples/historical/`. Completed local experiments are
preserved under ignored `data/research_archive/`, with closure-marker checks in
maintained training CLIs. See [containment policy](docs/research-containment.md).

The earlier display extraction intentionally changed the `_disk.py` checkpoint source hash;
old optimizer states are not automatically migrated or hash-rewritten. Historical
coordinates/results remain readable. See the deprecation guide for the boundary.

## Development and tests

```bash
python -m pytest tests
```

The [Actions workflow](.github/workflows/tests.yml) configures CPU-Torch full tests,
a Torch-free path and installed-wheel smoke checks without private datasets.
See the [roadmap](docs/development-roadmap.md) for research gates and follow-ups.
