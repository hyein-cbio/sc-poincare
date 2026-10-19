"""Frozen short probe of 10x pbmc3k, a measured set absent from the study records.

Labels are not read. The recipe below is fixed before fitting: total-count
normalization, log1p, 1000 Seurat HVGs, full-SVD PCA20, MU defaults, 20 updates,
seed 0. Spectral budgets are recorded and do not change the default solver.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scanpy as sc

from sc_poincare._disk import pairwise_disk_length
from sc_poincare._fitting import _seed_from_connectivities
from sc_poincare._proximity import local_affinity
from sc_poincare.controlled_probe import classical_jitter, fit_from_initial
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress

MARKERS = ("CD3D", "CD14", "MS4A1", "GNLY", "FCER1A", "PPBP")
DENSE_BUDGET = 512 * 1024**2


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def spectral_status(weights, *, tol=0.0, maxiter=None):
    status = {}
    _seed_from_connectivities(
        weights, 2, np.random.default_rng(0), status=status, spectral_tol=tol, spectral_maxiter=maxiter,
    )
    diagnostics = status.get("initialization_diagnostics", {})
    return dict(
        initialization=status.get("initialization"),
        reason=status.get("initialization_reason"),
        solver=diagnostics.get("solver"),
        converged=diagnostics.get("converged"),
        max_eigen_residual=diagnostics.get("max_eigen_residual"),
        spectral_tol=diagnostics.get("spectral_tol"),
        spectral_maxiter=diagnostics.get("spectral_maxiter"),
    )


def main():
    adata = sc.datasets.pbmc3k()
    if adata.n_obs < 1000 or adata.n_vars < 1000:
        raise SystemExit(f"Unexpected pbmc3k shape {adata.shape}.")
    genes = np.asarray(adata.var_names.astype(str))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    logged = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat")
    kept = np.asarray(adata.var["highly_variable"])
    matrix = logged[:, kept]
    from sklearn.decomposition import PCA
    features = np.ascontiguousarray(PCA(20, svd_solver="full").fit_transform(matrix), dtype=np.float64)
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    fitted = fit_from_initial(
        setup, classical_jitter(setup, 0), epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET,
    )
    times = pairwise_disk_length(np.zeros((1, 2)), fitted["current"])[0]
    marker_spearman = {}
    from scipy.stats import spearmanr
    for name in MARKERS:
        matches = np.flatnonzero(genes == name)
        if len(matches) != 1:
            marker_spearman[name] = None
            continue
        marker_spearman[name] = float(spearmanr(logged[:, matches[0]], times).statistic)
    weights, _lengths = local_affinity(features, 15, "median")
    payload = dict(
        input_id="pbmc3k-10x",
        used_in_recorded_studies=False,
        labels_entered_fitting=False,
        biological_validation=False,
        promoted=False,
        spectral_defaults_changed=False,
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        feature_shape=list(features.shape),
        feature_sha256=digest(features),
        recipe=dict(normalize_total=1e4, log1p=True, hvg=1000, flavor="seurat", pca=20, svd_solver="full",
                    epochs=20, seed=0, n_neighbors=15),
        loss_at_0=fitted["loss_trace"][0],
        loss_at_20=fitted["loss_trace"][20],
        root_rule="disk origin, applied only after fitting",
        marker_pseudotime_spearman=marker_spearman,
        spectral_default=spectral_status(weights),
        spectral_extended=spectral_status(weights, tol=1e-8, maxiter=5000),
    )
    Path("docs/pbmc3k-independent-probe.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(dict(shape=payload["feature_shape"], loss20=payload["loss_at_20"],
                          spectral=payload["spectral_default"], extended=payload["spectral_extended"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
