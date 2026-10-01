# sc-poincare

Poincaré-disk embeddings for hierarchical single-cell structure, written to drop into a Scanpy workflow.

The method follows Klimovskaia et al., *Poincaré maps for analyzing complex hierarchies in single-cell data* (Nat Commun, 2020): a connected neighborhood graph, Relative Forest Accessibility as a global proximity, and a two-dimensional hyperbolic embedding matched to those proximities by symmetric KL.

This package is an independent reimplementation of the paper's mathematics. It does not share code with the authors' release.

## Install

```bash
pip install -e .
```

Needs NumPy, SciPy, scikit-learn, AnnData, and Matplotlib. PyTorch is used for the Riemannian updates when it is already installed; otherwise a NumPy fallback is used.

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

The paper's usual ranges are `k ∈ {15, 20, 30}`, `σ ∈ {1, 2}`, `γ ∈ {1, 2}`. Riemannian updates default to `learning_rate=0.01` and a few hundred epochs: larger steps drive developed cells onto the disk boundary (`r → 1`) and freeze them there. If the feature dimension is above 100 and you do not pass `use_rep`, the tool expects `adata.obsm["X_pca"]`.

## Method, briefly

1. Build a **mutual k-nearest-neighbor graph** in the chosen feature space and stitch components with the shortest remaining links so the graph is connected.
2. Weight surviving edges with a Gaussian kernel of width `σ`.
3. Convert the graph into global proximities via the Relative Forest Accessibility matrix `P = (I + L)^{-1}`.
4. Place one point per cell in the Poincaré disk so that softmaxed hyperbolic distances reproduce `P`, by minimizing the symmetric KL with Riemannian gradient steps.
5. Optionally translate the disk (an isometry) so a known root sits at the origin. Pseudotime is then the hyperbolic distance from that origin; lineages are agglomerative clusters of polar angle.

## Tests

```bash
python -m pytest tests
```
