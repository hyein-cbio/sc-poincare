"""Frozen 2D versus 3D ball probe on exact Moignard features.

Labels and the marker list are used only after fitting. A lower 3D loss is
not a reason to change the 2D default.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from sc_poincare.controlled_probe import classical_ball, fit_from_initial, promotion_allowed
from sc_poincare.evaluation import evaluate_neighborhoods
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress

SOURCE = Path("/tmp/sc-poincare-inputs/Moignard2015.csv")
SOURCE_SHA = "3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1"
FEATURE_SHA = "674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0"
MARKERS = ("Gata1", "Kdr", "Runx1", "Sfpi1", "Cdh5")
STAGE_ORDER = ("PS", "4SFG", "4SG", "NP")
SEPARATE_LABEL = "HF"
DENSE_BUDGET = 512 * 1024**2


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def feature_medoid(features, mask):
    rows = np.flatnonzero(mask)
    distances = cdist(features[rows], features[rows])
    return int(rows[int(np.argmin(distances.sum(axis=1)))])


def main():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise SystemExit("Moignard CSV hash mismatch.")
    frame = pd.read_csv(SOURCE)
    labels = frame.iloc[:, -1].astype(str).to_numpy()
    columns = list(frame.columns[:-1])
    features = np.ascontiguousarray(frame.iloc[:, :-1].to_numpy(dtype=np.float64))
    if features.shape != (3934, 42) or digest(features) != FEATURE_SHA:
        raise SystemExit("Moignard feature hash mismatch.")
    missing = [name for name in MARKERS if name not in columns]
    if missing:
        raise SystemExit(f"Missing predeclared markers: {missing}")
    root = feature_medoid(features, labels == "PS")
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    from sc_poincare._disk import pairwise_disk_length
    results = []
    for dimension in (2, 3):
        print(f"dimension {dimension}", flush=True)
        initial = classical_ball(setup, dimension, 0)
        fitted = fit_from_initial(
            setup, initial, epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET,
        )
        current = fitted["current"]
        evaluation = evaluate_neighborhoods(features, current, labels, ks=(15,), geometry="poincare")
        metrics = evaluation.summary["metrics"]["15"]
        times = pairwise_disk_length(current[root:root + 1], current)[0]
        marker_spearman = {}
        for name in MARKERS:
            values = features[:, columns.index(name)]
            marker_spearman[name] = float(spearmanr(values, times).statistic)
        stage_means = {label: float(times[labels == label].mean()) for label in STAGE_ORDER}
        results.append(dict(
            dimension=dimension,
            loss_at_0=fitted["loss_trace"][0],
            loss_at_20=fitted["loss_trace"][20],
            best_epoch=fitted["best_epoch"],
            macro_purity=metrics["macro_purity"],
            input_neighbor_recall=metrics["input_neighbor_recall"],
            populations=sorted(metrics["populations"], key=lambda item: item["n_cells"]),
            root_index=root,
            stage_mean_pseudotime=stage_means,
            stage_order_matches_means=stage_means["PS"] < stage_means["4SFG"] < stage_means["4SG"] < stage_means["NP"],
            separate_label_mean_pseudotime={SEPARATE_LABEL: float(times[labels == SEPARATE_LABEL].mean())},
            marker_pseudotime_spearman=marker_spearman,
        ))
    payload = dict(
        input_id="moignard2015-development-reference",
        feature_sha256=FEATURE_SHA,
        labels_entered_fitting=False,
        markers_predeclared=list(MARKERS),
        stage_order_predeclared=list(STAGE_ORDER),
        promoted=promotion_allowed(development_input=True, joint_improvements=[True]),
        biological_validation=False,
        spectral_defaults_changed=False,
        results=results,
    )
    Path("docs/moignard-dimension-probe.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({row["dimension"]: row["loss_at_20"] for row in results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
