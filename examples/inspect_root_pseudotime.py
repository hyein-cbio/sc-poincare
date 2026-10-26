"""Root distance and pseudotime on Paul, with the root rule fixed first.

The root is the unique cell whose label is ``root``. Pseudotime is the
hyperbolic distance from that cell. Labels do not enter the graph, bandwidth,
initialization, or objective. Ery stage is the leading integer of labels that
end in Ery; that order is the label text, not a rank chosen after fitting.
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

from sc_poincare._disk import pairwise_disk_length
from sc_poincare._fitting import _seed_from_connectivities, fit_disk_embedding
from sc_poincare._objectives import CONDITIONAL_OBJECTIVE, DEFAULT_OBJECTIVE
from sc_poincare._proximity import estimate_proximity
from sc_poincare._research_policy import ensure_active_paths


def ery_stage(label: str) -> int | None:
    if not str(label).endswith("Ery"):
        return None
    digits = ""
    for character in str(label):
        if character.isdigit():
            digits += character
        else:
            break
    return int(digits) if digits else None


def summarize(coordinates: np.ndarray, labels: np.ndarray) -> dict:
    root = np.flatnonzero(labels == "root")
    if len(root) != 1:
        raise ValueError("Expected exactly one cell labeled root.")
    pseudotime = pairwise_disk_length(coordinates[root], coordinates)[0]
    medians = {}
    for label in sorted(set(labels), key=str):
        medians[str(label)] = float(np.median(pseudotime[labels == label]))
    stages, stage_time = [], []
    for label, median in medians.items():
        stage = ery_stage(label)
        if stage is not None:
            stages.append(stage)
            stage_time.append(median)
    return {
        "root_disk_radius": float(np.linalg.norm(coordinates[root[0]])),
        "median_disk_radius": float(np.median(np.linalg.norm(coordinates, axis=1))),
        "median_pseudotime": medians,
        "ery_stage_spearman": float(spearmanr(stages, stage_time).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensure_active_paths(args.csv, args.output)
    raw = args.csv.read_bytes()
    frame = pd.read_csv(args.csv)
    label_name = frame.columns[-1]
    labels = frame[label_name].astype(str).to_numpy()
    if int(np.sum(labels == "root")) != 1:
        raise SystemExit("Refusing to fit: the root rule requires one root cell.")
    features = frame.drop(columns=label_name).to_numpy(dtype=np.float64)
    proximity = estimate_proximity(features, 15, "median")
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
            summary = summarize(report.coordinates, labels)
            summary.update({
                "objective": objective,
                "seed": int(seed),
                "best_loss": float(report.best_loss),
                "stop_reason": report.stop_reason,
                "fitting_seconds": time.perf_counter() - started,
            })
            runs.append(summary)
            print(objective, seed, round(summary["root_disk_radius"], 4), round(summary["ery_stage_spearman"], 4), flush=True)
    args.output.write_text(json.dumps({
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "root_rule": "the unique cell labeled root; fixed before fitting",
        "pseudotime": "hyperbolic distance from that cell",
        "ery_order": "leading integer of labels ending in Ery",
        "representation": "1000 numeric genes, no PCA",
        "sigma_effective": proximity.kernel_width,
        "epochs": args.epochs,
        "runs": runs,
    }, indent=2))


if __name__ == "__main__":
    main()
