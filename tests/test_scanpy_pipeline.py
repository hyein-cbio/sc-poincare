from __future__ import annotations

import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")

from sc_poincare.tl import poincare_map, poincare_pseudotime
from tests.test_scanpy_api import _branching_anndata


def test_fits_after_standard_scanpy_preprocess():
    adata = _branching_anndata(n_per_arm=12, n_genes=40, seed=7)
    scanpy.pp.scale(adata, max_value=10)
    scanpy.pp.pca(adata, n_comps=8)
    poincare_map(adata, use_rep="X_pca", n_neighbors=6, n_epochs=30, patience=30, random_state=0)
    assert adata.obsm["X_poincare"].shape == (adata.n_obs, 2)
    assert adata.uns["poincare"]["params"]["use_rep"] == "X_pca"
    poincare_pseudotime(adata, root=0)
    # Scanpy's generic embedding plot looks up X_<basis>.
    ax = scanpy.pl.embedding(adata, basis="poincare", color="poincare_pseudotime", show=False)
    assert ax is not None
    assert np.isfinite(adata.obs["poincare_pseudotime"]).all()
