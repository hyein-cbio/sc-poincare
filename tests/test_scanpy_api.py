from __future__ import annotations

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from sc_poincare._disk import pairwise_disk_length
from sc_poincare.tl import poincare_lineages, poincare_map, poincare_pseudotime, poincare_recenter


def _branching_anndata(n_per_arm: int = 18, n_genes: int = 16, seed: int = 0):
    """Three arms that share a progenitor cloud — a tiny tree-like manifold."""
    rng = np.random.default_rng(seed)
    root = rng.normal(scale=0.15, size=(n_per_arm, n_genes))
    arms = []
    labels = ["root"] * n_per_arm
    directions = [
        np.linspace(0.0, 3.0, n_per_arm)[:, None] * np.eye(n_genes)[0],
        np.linspace(0.0, 3.0, n_per_arm)[:, None] * np.eye(n_genes)[1],
        np.linspace(0.0, 3.0, n_per_arm)[:, None] * np.eye(n_genes)[2],
    ]
    for name, shift in zip(["a", "b", "c"], directions):
        cloud = root + shift + rng.normal(scale=0.08, size=(n_per_arm, n_genes))
        arms.append(cloud)
        labels.extend([name] * n_per_arm)
    x = np.vstack([root, *arms])
    adata = anndata.AnnData(x)
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    adata.obs["branch"] = labels
    return adata


def test_layout_writes_scanpy_slots_and_stays_inside_the_disk():
    adata = _branching_anndata()
    out = poincare_map(
        adata,
        method="rfa",
        n_neighbors=8,
        sigma=1.0,
        gamma=1.5,
        n_epochs=80,
        patience=80,
        burn_in_epochs=5,
        random_state=0,
        verbose=False,
    )
    assert out is None
    assert "X_poincare" in adata.obsm
    assert "poincare" in adata.uns
    coords = adata.obsm["X_poincare"]
    assert coords.shape == (adata.n_obs, 2)
    assert np.all(np.linalg.norm(coords, axis=1) < 1.0)
    assert "poincare_connectivities" in adata.obsp
    assert adata.uns["poincare"]["params"]["use_rep"] == "X"
    assert adata.uns["poincare"]["loss"].ndim == 1
    assert len(adata.uns["poincare"]["loss"]) >= 1


def test_copy_flag_leaves_the_original_untouched():
    adata = _branching_anndata()
    clone = poincare_map(
        adata,
        method="rfa",
        n_neighbors=6,
        n_epochs=30,
        patience=30,
        copy=True,
        random_state=1,
    )
    assert "X_poincare" in clone.obsm
    assert "X_poincare" not in adata.obsm


def test_root_translation_and_pseudotime():
    adata = _branching_anndata()
    poincare_map(adata, method="rfa", n_neighbors=8, n_epochs=60, patience=60, random_state=2)
    root = "c0"
    before = pairwise_disk_length(adata.obsm["X_poincare"]).copy()
    poincare_recenter(adata, root)
    after = pairwise_disk_length(adata.obsm["X_poincare"])
    assert np.linalg.norm(adata.obsm["X_poincare"][0]) < 1e-8
    assert np.allclose(before, after, rtol=1e-5, atol=1e-6)
    poincare_pseudotime(adata, root=root)
    pt = adata.obs["poincare_pseudotime"].to_numpy()
    assert pt[0] < 1e-8
    # Terminal-ish cells on an arm should sit farther from the progenitor than the progenitor itself.
    root_mask = adata.obs["branch"].to_numpy() == "root"
    arm_mask = adata.obs["branch"].to_numpy() == "a"
    assert pt[arm_mask].mean() > pt[root_mask].mean()


def test_angular_lineages_are_written_as_categories():
    adata = _branching_anndata()
    poincare_map(adata, method="rfa", n_neighbors=8, n_epochs=50, patience=50, random_state=3)
    poincare_lineages(adata, n_lineages=3, root="c0")
    assert "poincare_lineage" in adata.obs
    assert str(adata.obs["poincare_lineage"].dtype) == "category"
    assert adata.obs["poincare_lineage"].nunique() == 3


def test_use_rep_reads_obsm_and_key_added_renames_slots():
    adata = _branching_anndata()
    adata.obsm["X_custom"] = adata.X.copy()
    poincare_map(
        adata,
        method="rfa",
        use_rep="X_custom",
        n_neighbors=6,
        n_epochs=25,
        patience=25,
        key_added="hyp",
        random_state=4,
    )
    assert "X_hyp" in adata.obsm
    assert "hyp" in adata.uns
    assert adata.uns["hyp"]["params"]["use_rep"] == "X_custom"


def test_plot_function_returns_axes_without_displaying():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from sc_poincare.pl import poincare_map as plot_map

    adata = _branching_anndata()
    poincare_map(adata, method="rfa", n_neighbors=6, n_epochs=20, patience=20, random_state=5)
    ax = plot_map(adata, color="branch", show=False)
    assert ax is not None
    assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal"


def test_gentle_schedule_keeps_a_radial_gradient():
    """A small learning rate should unfold the tree instead of parking every cell at r≈1."""
    adata = _branching_anndata()
    poincare_map(
        adata,
        method="rfa",
        n_neighbors=8,
        sigma=1.0,
        gamma=2.0,
        learning_rate=0.01,
        n_epochs=120,
        burn_in_epochs=20,
        patience=120,
        random_state=0,
    )
    poincare_recenter(adata, 0)
    radii = np.linalg.norm(adata.obsm["X_poincare"], axis=1)
    branches = adata.obs["branch"].to_numpy()
    root_r = np.median(radii[branches == "root"])
    leaf_r = np.median(radii[branches != "root"])
    assert root_r < leaf_r
    assert (radii < 0.95).mean() > 0.25
