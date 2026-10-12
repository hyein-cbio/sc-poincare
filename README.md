# sc-poincare

Poincaré-disk embeddings for hierarchical single-cell structure, written to drop into a Scanpy workflow.

The method follows Klimovskaia et al., *Poincaré maps for analyzing complex hierarchies in single-cell data* (Nat Commun, 2020): a connected neighborhood graph, Relative Forest Accessibility as a global proximity, and a two-dimensional hyperbolic embedding matched to those proximities by symmetric KL.

This package is an independent reimplementation of the paper's mathematics. It does not share code with the authors' release.

## Install

```bash
pip install -e .
```

Needs NumPy, SciPy, scikit-learn, AnnData, and Matplotlib. Install `pip install -e '.[torch]'` for PyTorch-based Riemannian updates. Without PyTorch, a slow finite-difference fallback approximates the gradient of the exact global objective; it is available only for at most 128 cells.

## Scanpy-style usage

```python
import scanpy as sc
import sc_poincare as dr

adata = sc.read_h5ad("cells.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50)

dr.tl.poincare_map(adata, n_neighbors=15, sigma=2.0, gamma=2.0)
dr.pl.poincare_map(adata, color="cell_type")

dr.tl.poincare_recenter(adata, root="HSC_1")
dr.tl.poincare_pseudotime(adata, root="HSC_1")
dr.tl.poincare_lineages(adata, n_lineages=4, root="HSC_1")

# spread=0 is exactly X_poincare. spread=1 fans dendrogram leaves.
dr.tl.poincare_fan(adata, groupby="cell_type", spread=0.0)
# repel=0 is exactly X_poincare.
dr.tl.poincare_repel(adata, repel=0.0)
# jitter=0 is exactly X_poincare. Positive values are radians.
dr.tl.poincare_jitter(adata, jitter=0.0)

sc.pl.embedding(adata, basis="poincare", color=["poincare_pseudotime", "poincare_lineage"])
```

### What gets written

| slot | contents |
| --- | --- |
| `adata.obsm["X_poincare"]` | disk coordinates (open unit ball) |
| `adata.obsp["poincare_connectivities"]` | local Gaussian affinities |
| `adata.uns["poincare"]` | hyperparameters and loss trace |
| `adata.obs["poincare_pseudotime"]` | hyperbolic distance from the root |
| `adata.obs["poincare_lineage"]` | angular slices after recentering |

`scanpy.pl.embedding(adata, basis="poincare")` works because Scanpy looks up `X_poincare`.

### Typical knobs

The paper's usual ranges are `k ∈ {15, 20, 30}`, `σ ∈ {1, 2}`, `γ ∈ {1, 2}`. **σ is in the units of the chosen feature distances**, not a universal setting across raw expression matrices or PCA scores. Riemannian updates default to `learning_rate=0.01`; inspect convergence rather than assuming a few hundred epochs are sufficient. If the feature dimension is above 100 and you do not pass `use_rep`, the tool expects `adata.obsm["X_pca"]`.

## Method, briefly

1. Build a **mutual k-nearest-neighbor graph** in the chosen feature space and stitch components with the shortest remaining links so the graph is connected.
2. Weight every edge, including added bridges, with the same Gaussian kernel of width `σ`. No artificial bridge-strength floor is applied.
3. Convert the graph into global proximities via the Relative Forest Accessibility matrix `P = (I + L)^{-1}`.
4. Initialize disk coordinates from a spectral layout of the local connectivity graph, scaled near the origin. Fit softmaxed hyperbolic distances to `P` by minimizing the global symmetric KL with Riemannian gradient steps.
5. Optionally translate the disk (an isometry) so a known root sits at the origin. Pseudotime is then the hyperbolic distance from that origin; lineages are agglomerative clusters of polar angle.

## Exact mode and memory limits

This is currently a **small-data, exact global-proximity implementation**, not an atlas-scale method. RFA is generally dense even when the graph is sparse, and storing it requires O(N²) memory. A float64 matrix alone uses 0.8 GB at 10,000 cells or 12.8 GB at 40,000 cells.

`max_dense_bytes` defaults to 512 MiB. Before feature densification or graph construction, the AnnData API checks a conservative budget of four float64 N×N working arrays. The default therefore allows at most 4,096 cells. This is **not** a bound on process RSS: sparse factorization fill-in, feature matrices, autograd, imports and other allocations consume additional memory. Subsample larger inputs, or explicitly raise this budget at your own risk (`None` disables the check).

Graph construction uses kNN edge distances. Disconnected components are bridged with exact, chunked component-MST searches rather than a globally sorted list of all cell pairs. RFA is computed using one sparse factorization and blocked identity-column solves; its final result is still dense.

`batch_size` now means **row workspace size**, not a conditional minibatch objective. Every row is compared with all cells, all gradient contributions are accumulated at the same coordinates, and one update is made per epoch. This reproduces the full global loss and gradient up to floating-point summation order. It saves fitting workspace but does not remove RFA's quadratic storage. The NumPy fallback likewise differentiates the global loss rather than a batch submatrix.

`uns['poincare']['loss']` records losses at post-update checkpoints. `best_loss` is the loss of the returned best checkpoint, including the initial coordinates if no update improves them. After burn-in, early stopping measures cumulative improvement since the last significant checkpoint, not just the last epoch. `stop_reason` is `patience` or `epoch_limit`; neither alone proves convergence. Geometry uses direct Euclidean separations inside the Poincaré distance formula, without an off-diagonal distance floor. Training updates project into a safe interior margin, but geometry and root translation never apply that training cap to valid points. They reject invalid coordinates; if translation exhausts floating-point boundary precision it raises rather than silently changing distances.

## Numerical safeguards and target diagnostics

- `max_grad_norm=2.5` clips the **Riemannian norm** of each accumulated gradient, not its Euclidean norm. `None` disables clipping. This changes optimizer behavior, not the KL objective or the embedding's radius limit. Both Torch and the NumPy fallback use the same rule. Non-finite gradients raise instead of becoming zero updates.
- Close but distinct points retain their distance gradients. Exactly coincident points use a finite zero subgradient; they are not artificially separated. The small-data NumPy fallback uses adaptive central-difference steps that stay inside the ball and avoid nearby distance kinks; perturbations are not projected. It remains an approximate gradient of the exact global objective.
- If Gaussian underflow disconnects the weighted graph, graph construction raises before RFA or fitting. Increase sigma or explicitly change feature preprocessing, rather than fabricating bridge weights. Underflow of nonessential edges produces a warning.
- If at least 10% of cells assign over 99% of RFA mass to themselves, `ProximityWarning` identifies a weak trajectory target. This heuristic is a diagnostic, not a biological quality test. No normalization or sigma tuning is done silently.
- Inspect `adata.uns['poincare']['proximity_diagnostics']` for edge-weight ranges, median edge distance relative to sigma, and RFA self-mass counts. Kernel and clipping settings are recorded in `params`.

For example, inspect the warning and feature units, then **explicitly** choose a better representation or bandwidth. An opt-in global bandwidth rule is available:

```python
dr.tl.poincare_map(adata, use_rep="X_pca", sigma="median")
print(adata.uns["poincare"]["params"]["sigma"])  # effective numeric bandwidth
print(adata.uns["poincare"]["proximity_diagnostics"])
```

`"median"` uses the median **positive** distance among the stitched graph's edges, including bridges. It does not standardize features or use cell labels. It is a heuristic, not a reproduction of the authors' settings or a guarantee of a good target. Very long bridges can still underflow; all-zero edge distances require an explicit positive numeric sigma. Numeric sigma values retain their previous meaning, and the default remains `2.0`. Requested and effective values are recorded separately as `sigma_requested` and `sigma`.

The actual initialization is recorded as `spectral`, `random` or `random_fallback`, with an `initialization_reason`. An eigensolver failure therefore cannot be mistaken for a completed spectral initialization.

The package does not claim that a particular rescaling reproduces the authors' preprocessing. To refuse suspicious targets entirely, promote the warning to an error:

```python
import warnings
from sc_poincare._proximity import ProximityWarning
warnings.filterwarnings("error", category=ProximityWarning)
```

## Probability calculation

The default remains symmetric RFA KL with self included. Fitting now uses
log-softmax and retains positive target probabilities instead of flooring them
to `1e-12`. Exact-zero target logs alone use a documented machine-tiny numerical
regularization, with a warning. Losses and stopping trajectories can change.
See [objective definitions and numerical policy](docs/objectives.md).

An opt-in **experimental** variant excludes self from both distributions while
retaining all other cells:

```python
dr.tl.poincare_map(adata, objective="conditional_symmetric_kl")
```

This is a different objective, not the paper's self-inclusive symmetric KL.
It can lose isolation information and has local/global trade-offs; it is not
an unconditional improvement or a claim of biological equivalence. The default
is still `objective="symmetric_kl"`. The selected objective and numerical policy
are saved in `adata.uns['poincare']['params']`; RFA self-mass diagnostics remain.

## Long-fit progress and warm starts

The low-level `sc_poincare._fitting.fit_disk_embedding` accepts an optional
`progress_callback` and `initial_coordinates`. A callback receives independent
copies of the evaluated current and best coordinates, their losses, epoch number
(including epoch 0), and initialization provenance. Mutating the copies or returning
a value does not alter fitting; exceptions abort the run. Checkpoint I/O is synchronous
and belongs to the caller, so include its overhead in timing measurements.

Warm-start coordinates must be finite, have shape `(n_obs, n_components)`, and fit
inside the optimizer's safety margin. Use raw training coordinates, not root-translated
display coordinates. A warm start is a **new run**: it does not restore patience,
burn-in position or prior best-checkpoint history. Explicitly configure the new
schedule. These options do not change the default Scanpy API or default fitting
trajectory, and do not remove dense RFA's quadratic memory requirement.

## Benchmarks

See the [objective ablation and its limitations](docs/objective-validation.md)
and `examples/benchmark_objectives.py` to compare both objectives with shared
initialization on your own inputs. An optional isolated historical-floor baseline
is for numerical diagnosis, not a production method or an external-code replay.

Install `pip install -e '.[torch,benchmark]'` for the benchmark's explicit dependencies. See [the small-data benchmark protocol](docs/benchmarking.md) and `examples/benchmark_small.py`. No general superiority over the authors' implementation has been established. Compare unmodified learned coordinates, not jittered or repelled display layouts.

A [dataset-specific reference-condition search](docs/reference-parity.md) found recipes that pass two graph-distance diagnostics on six datasets across three seeds. **Five recipes use an experimental geodesic-regularized objective**, not the paper's pure RFA KL. This is not full analysis equivalence or a generalization claim. `examples/run_selected_conditions.py` reproduces those opt-in conditions from local input CSVs; the default Scanpy API is unchanged.

## Tests

```bash
python -m pytest tests
```
