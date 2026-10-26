"""Frozen curvature probe on exact Moignard features. Curvature 1 stays the default.

Own losses are not a common score across curvatures. Labels are used only after fitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc_poincare.controlled_probe import classical_jitter, fit_from_initial, promotion_allowed
from sc_poincare.evaluation import evaluate_neighborhoods
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress
from sc_poincare._research_policy import ensure_active_paths

SOURCE = Path("/tmp/sc-poincare-inputs/Moignard2015.csv")
SOURCE_SHA = "3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1"
FEATURE_SHA = "674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0"
CURVATURES = (1., 0.25)
DENSE_BUDGET = 512 * 1024**2


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def main():
    output = Path("docs/moignard-curvature-probe.json")
    ensure_active_paths(SOURCE, output)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise SystemExit("Moignard CSV hash mismatch.")
    frame = pd.read_csv(SOURCE)
    labels = frame.iloc[:, -1].astype(str).to_numpy()
    features = np.ascontiguousarray(frame.iloc[:, :-1].to_numpy(dtype=np.float64))
    if features.shape != (3934, 42) or digest(features) != FEATURE_SHA:
        raise SystemExit("Moignard feature hash mismatch.")
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    initial = classical_jitter(setup, 0)
    results = []
    for curvature in CURVATURES:
        print(f"curvature {curvature}", flush=True)
        fitted = fit_from_initial(
            setup, initial, epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET, curvature=curvature,
        )
        evaluation = evaluate_neighborhoods(
            features, np.sqrt(curvature) * fitted["current"], labels, ks=(15,), geometry="poincare",
        )
        metrics = evaluation.summary["metrics"]["15"]
        results.append(dict(
            curvature=curvature,
            loss_at_0=fitted["loss_trace"][0],
            loss_at_20=fitted["loss_trace"][20],
            macro_purity=metrics["macro_purity"],
            input_neighbor_recall=metrics["input_neighbor_recall"],
            populations=sorted(metrics["populations"], key=lambda item: item["n_cells"]),
        ))
    payload = dict(
        input_id="moignard2015-development-reference",
        feature_sha256=FEATURE_SHA,
        labels_entered_fitting=False,
        curvatures=list(CURVATURES),
        promoted=promotion_allowed(development_input=True, joint_improvements=[True]),
        biological_validation=False,
        results=results,
    )
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({row["curvature"]: row["loss_at_20"] for row in results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
