"""Frozen graph-stress fit on the same pbmc3k features as the RFA rule.

Preprocessing is the pbmc3k rule already fixed in inspect_pbmc3k_frozen.py.
The embedding rule is the experimental local-radius graph stress, k=15, for
1,500 updates. Labels are read only after fitting.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from sc_poincare.experimental import GraphStressConfig, fit_graph_stress, prepare_graph_stress


def _frozen():
    spec = importlib.util.spec_from_file_location(
        "inspect_pbmc3k_frozen",
        Path(__file__).resolve().parent / "inspect_pbmc3k_frozen.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = _frozen()
    names, features, label_table = frozen.prepare()
    setup = prepare_graph_stress(features, config=GraphStressConfig(local_bandwidth="local_radius"))
    runs = []
    for seed in args.seeds:
        started = time.perf_counter()
        result = fit_graph_stress(setup, epochs=args.epochs, random_state=seed, chunk_size=256)
        kept = [name in label_table.index for name in names]
        import numpy as np
        kept = np.asarray(kept)
        order = [label_table.index.get_loc(name) for name in names[kept]]
        scores = frozen.purity(result.coordinates[kept], label_table.to_numpy()[order])
        runs.append({
            "seed": int(seed),
            "best_loss": float(result.best_loss),
            "best_epoch": int(result.best_epoch),
            "current_loss": float(result.current_loss),
            "epochs_run": int(result.n_epochs),
            "stop_reason": "epoch_limit",
            "fitting_seconds": time.perf_counter() - started,
            "feature_distance_rank": frozen.feature_rank(result.coordinates, features),
            "labeled_cells": int(kept.sum()),
            "purity_k15": scores,
        })
        print(seed, result.best_epoch, round(result.best_loss, 4), round(scores["overall"], 4), flush=True)
    args.output.write_text(json.dumps({
        "dataset": "scanpy pbmc3k raw counts",
        "preprocessing": "same frozen normalize/log1p/1000 HVG/PCA20 full SVD as the symmetric-KL pbmc3k run",
        "rule": "GraphStressConfig local_radius, k=15, 1500 updates, tangent Adam, geometry poincare",
        "labels": "pbmc3k_processed louvain, evaluation only",
        "n_cells": int(len(features)),
        "runs": runs,
    }, indent=2))


if __name__ == "__main__":
    main()
