"""Frozen symmetric-KL fit on pbmc3k.

The rule is fixed here, before any score is read. Raw counts only. No cell
label, cluster, or mitochondrial filter enters the graph, PCA, or fit.
Evaluation labels from the processed file are applied after fitting and only
to cells present in both tables.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from sc_poincare._disk import pairwise_disk_length
from sc_poincare._fitting import _seed_from_connectivities, fit_disk_embedding
from sc_poincare._proximity import estimate_proximity
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sc_poincare._research_policy import ensure_active_paths


def feature_rank(coordinates: np.ndarray, features: np.ndarray, *, feature_distance=None) -> float:
    if feature_distance is None:
        feature_distance = cdist(features, features)
    embedding_distance = pairwise_disk_length(coordinates)
    mask = ~np.eye(len(coordinates), dtype=bool)
    return float(spearmanr(feature_distance[mask], embedding_distance[mask]).statistic)


def purity(coordinates: np.ndarray, labels: np.ndarray, k: int = 15) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    neighbors = nn.kneighbors(coordinates, return_distance=False)
    indices = np.asarray([row[row != i][:k] for i, row in enumerate(neighbors)])
    same = np.array([np.mean(labels[row] == labels[i]) for i, row in enumerate(indices)])
    scores = {"overall": float(same.mean())}
    for label in sorted(set(labels), key=str):
        scores[str(label)] = float(same[labels == label].mean())
    return scores


def prepare():
    datasetdir = Path(sc.settings.datasetdir)
    ensure_active_paths(datasetdir, datasetdir / "pbmc3k_raw.h5ad", datasetdir / "pbmc3k_processed.h5ad")
    raw = sc.datasets.pbmc3k()
    labels = sc.datasets.pbmc3k_processed().obs["louvain"].astype(str)
    sc.pp.normalize_total(raw, target_sum=1e4)
    sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=1000, flavor="seurat")
    from sklearn.decomposition import PCA
    matrix = raw[:, raw.var["highly_variable"]].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    features = np.asarray(PCA(20, svd_solver="full").fit_transform(matrix), dtype=np.float64)
    return raw.obs_names.to_numpy(), features, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensure_active_paths(args.output)
    names, features, label_table = prepare()
    proximity = estimate_proximity(features, 15, "median")
    feature_distance = cdist(features, features)
    runs = []
    for seed in args.seeds:
        status: dict[str, str] = {}
        initial = _seed_from_connectivities(proximity.weights, 2, np.random.default_rng(seed), status=status)
        started = time.perf_counter()
        report = fit_disk_embedding(
            proximity.accessibility,
            initial_coordinates=initial,
            n_epochs=args.epochs,
            random_state=seed,
            burn_in_epochs=30,
            temperature=2.0,
            learning_rate=0.01,
        )
        kept = np.array([name in label_table.index for name in names])
        order = [label_table.index.get_loc(name) for name in names[kept]]
        scores = purity(report.coordinates[kept], label_table.to_numpy()[order])
        runs.append({
            "seed": int(seed),
            "initialization": status["initialization"],
            "best_loss": float(report.best_loss),
            "stop_reason": report.stop_reason,
            "epochs_run": int(report.n_epochs),
            "fitting_seconds": time.perf_counter() - started,
            "feature_distance_rank": feature_rank(report.coordinates, features, feature_distance=feature_distance),
            "labeled_cells": int(kept.sum()),
            "purity_k15": scores,
        })
        print(seed, round(runs[-1]["best_loss"], 2), round(scores["overall"], 4), flush=True)
    args.output.write_text(json.dumps({
        "dataset": "scanpy pbmc3k raw counts",
        "rule": "normalize_total 1e4, log1p, 1000 seurat HVGs, PCA 20 full SVD, k=15, sigma=median, symmetric_kl, gamma=2, 200 updates",
        "labels": "pbmc3k_processed louvain, evaluation only, aligned by barcode after fitting",
        "n_cells": int(len(features)),
        "n_features": int(features.shape[1]),
        "sigma_effective": proximity.kernel_width,
        "runs": runs,
    }, indent=2))


if __name__ == "__main__":
    main()
