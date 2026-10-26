"""Compare spectral and random disk initializations on one local CSV.

Labels are dropped before the graph and the fit. The report stores distance
ranks, losses and times, not coordinates or cell labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

from sc_poincare._disk import pairwise_disk_length
from sc_poincare._fitting import _seed_coordinates, _seed_from_connectivities, fit_disk_embedding
from sc_poincare._proximity import estimate_proximity
from sc_poincare._research_policy import ensure_active_paths


def distance_rank(coordinates: np.ndarray, features: np.ndarray, *, feature_distance=None) -> float:
    """Spearman correlation of feature distances and hyperbolic distances."""
    if feature_distance is None:
        feature_distance = cdist(features, features)
    embedding_distance = pairwise_disk_length(coordinates)
    mask = ~np.eye(len(coordinates), dtype=bool)
    return float(spearmanr(feature_distance[mask], embedding_distance[mask]).statistic)


def compare(features: np.ndarray, *, n_neighbors: int, sigma: float, epochs: int, seeds: list[int]) -> dict:
    proximity = estimate_proximity(features, n_neighbors, sigma)
    feature_distance = cdist(features, features)
    rows = []
    for kind in ("spectral", "random"):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            started = time.perf_counter()
            status: dict[str, str] = {}
            if kind == "spectral":
                initial = _seed_from_connectivities(proximity.weights, 2, rng, status=status)
            else:
                initial = _seed_coordinates(len(features), 2, rng)
                status = {"initialization": "random", "initialization_reason": ""}
            initialization_seconds = time.perf_counter() - started
            started = time.perf_counter()
            report = fit_disk_embedding(
                proximity.accessibility,
                initial_coordinates=initial,
                n_epochs=epochs,
                random_state=seed,
                burn_in_epochs=30,
                temperature=2.0,
                learning_rate=0.01,
            )
            fitting_seconds = time.perf_counter() - started
            rows.append({
                "initialization": status["initialization"],
                "initialization_reason": status["initialization_reason"],
                "seed": int(seed),
                "initialization_seconds": initialization_seconds,
                "fitting_seconds": fitting_seconds,
                "initial_feature_distance_rank": distance_rank(initial, features, feature_distance=feature_distance),
                "final_feature_distance_rank": distance_rank(report.coordinates, features, feature_distance=feature_distance),
                "best_loss": float(report.best_loss),
                "stop_reason": report.stop_reason,
                "epochs_run": int(report.n_epochs),
            })
    return {"n_cells": int(len(features)), "n_features": int(features.shape[1]), "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensure_active_paths(args.csv, args.output)
    raw = args.csv.read_bytes()
    frame = pd.read_csv(args.csv)
    label = args.label_column or frame.columns[-1]
    features = frame.drop(columns=label).to_numpy(dtype=np.float64)
    result = compare(features, n_neighbors=args.n_neighbors, sigma=args.sigma, epochs=args.epochs, seeds=args.seeds)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["label_column"] = label
    result["sigma"] = args.sigma
    result["n_neighbors"] = args.n_neighbors
    result["epochs"] = args.epochs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
