"""Compare self-inclusive and conditional KL on rare Paul labels.

Labels are counted before fitting and are not used to build the graph,
initialization, bandwidth, or objective. The report stores aggregate purity,
not coordinates or expression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sc_poincare._fitting import _seed_from_connectivities, fit_disk_embedding
from sc_poincare._objectives import CONDITIONAL_OBJECTIVE, DEFAULT_OBJECTIVE
from sc_poincare._proximity import estimate_proximity
from sc_poincare._research_policy import ensure_active_paths


def neighbor_purity(coordinates: np.ndarray, labels: np.ndarray, k: int = 15) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    neighbors = nn.kneighbors(coordinates, return_distance=False)
    indices = np.asarray([row[row != i][:k] for i, row in enumerate(neighbors)])
    purity = np.array([np.mean(labels[row] == labels[i]) for i, row in enumerate(indices)])
    scores = {}
    for label in sorted(set(labels), key=str):
        mask = labels == label
        scores[str(label)] = float(purity[mask].mean()) if np.any(mask) else float("nan")
    scores["overall"] = float(purity.mean())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rare-max-count", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensure_active_paths(args.csv, args.output)
    raw = args.csv.read_bytes()
    frame = pd.read_csv(args.csv)
    label_name = frame.columns[-1]
    labels = frame[label_name].to_numpy()
    counts = {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    rare = sorted(label for label, count in counts.items() if count <= args.rare_max_count)
    features = frame.drop(columns=label_name).to_numpy(dtype=np.float64)
    started = time.perf_counter()
    proximity = estimate_proximity(features, 15, "median")
    preparation_seconds = time.perf_counter() - started
    runs = []
    for seed in args.seeds:
        status: dict[str, str] = {}
        initial = _seed_from_connectivities(proximity.weights, 2, np.random.default_rng(seed), status=status)
        for objective in (DEFAULT_OBJECTIVE, CONDITIONAL_OBJECTIVE):
            started = time.perf_counter()
            report = fit_disk_embedding(
                proximity.accessibility,
                initial_coordinates=initial,
                n_epochs=args.epochs,
                random_state=seed,
                burn_in_epochs=30,
                temperature=2.0,
                learning_rate=0.01,
                objective=objective,
            )
            purity = neighbor_purity(report.coordinates, labels)
            runs.append({
                "objective": objective,
                "seed": int(seed),
                "initialization": status["initialization"],
                "best_loss": float(report.best_loss),
                "stop_reason": report.stop_reason,
                "epochs_run": int(report.n_epochs),
                "fitting_seconds": time.perf_counter() - started,
                "overall_purity_k15": purity["overall"],
                "rare_purity_k15": {label: purity[label] for label in rare},
                "rare_mean_purity_k15": float(np.mean([purity[label] for label in rare])),
            })
            print(objective, seed, round(runs[-1]["fitting_seconds"], 1), flush=True)
    result = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "n_cells": int(len(features)),
        "n_features": int(features.shape[1]),
        "representation": "all numeric columns after dropping the last label column; no PCA",
        "rare_rule": f"label count <= {args.rare_max_count}, fixed before fitting",
        "rare_counts": {label: counts[label] for label in rare},
        "sigma_effective": proximity.kernel_width,
        "preparation_seconds": preparation_seconds,
        "epochs": args.epochs,
        "runs": runs,
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
