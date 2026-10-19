"""Frozen Moignard initialization, budget, k, and degree-aware probe.

Labels are loaded only for post-fit diagnostics. They do not enter preparation,
initialization, bandwidth, or the Adam update. Development data cannot promote
a new default.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sc_poincare.controlled_probe import (
    classical_jitter,
    degree_aware_teacher,
    fit_from_initial,
    promotion_allowed,
    random_disk,
)
from sc_poincare.evaluation import evaluate_neighborhoods
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress
from sc_poincare.visibility import evaluate_visibility

SOURCE = Path("/tmp/sc-poincare-inputs/Moignard2015.csv")
SOURCE_SHA = "3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1"
FEATURE_SHA = "674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0"
ROOT_LABEL = "PS"
EPOCHS = 40
BUDGETS = (20, 40)
SEEDS = (0, 1)
K_SENSITIVITY = 30
MARKER_DIAMETERS = (4., 8.)
DENSE_BUDGET = 512 * 1024**2


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def feature_medoid(features, mask):
    rows = np.flatnonzero(mask)
    distances = cdist(features[rows], features[rows])
    return int(rows[int(np.argmin(distances.sum(axis=1)))])


def pseudotime(coordinates, root, geometry):
    if geometry == "poincare":
        from sc_poincare._disk import pairwise_disk_length
        return pairwise_disk_length(coordinates[root:root + 1], coordinates)[0]
    return np.linalg.norm(coordinates - coordinates[root], axis=1)


def run_condition(setup, features, labels, root, initial, *, epochs, geometry, teacher=None):
    fitted = fit_from_initial(
        setup, initial, epochs=epochs, chunk_size=256, geometry=geometry,
        max_dense_bytes=DENSE_BUDGET, teacher=teacher,
    )
    current = fitted["current"]
    evaluation = evaluate_neighborhoods(features, current, labels, ks=(15,), geometry=geometry)
    metrics = evaluation.summary["metrics"]["15"]
    populations = sorted(metrics["populations"], key=lambda item: item["n_cells"])
    times = pseudotime(current, root, geometry)
    record = dict(
        epochs=epochs,
        geometry=geometry,
        loss_at_budget={str(step): fitted["loss_trace"][step] for step in BUDGETS if step <= epochs},
        best_loss=fitted["best_loss"],
        best_epoch=fitted["best_epoch"],
        radius_quantiles=np.quantile(np.linalg.norm(current, axis=1), [0, .5, 1]).tolist(),
        macro_purity=metrics["macro_purity"],
        overall_purity=metrics["overall_purity"],
        input_neighbor_recall=metrics["input_neighbor_recall"],
        populations=populations,
        root_index=root,
        root_pseudotime_quantiles=np.quantile(times, [0, .5, 1]).tolist(),
    )
    if geometry == "poincare":
        visibility = {}
        for diameter in MARKER_DIAMETERS:
            report = evaluate_visibility(evaluation, marker_diameter=diameter)
            visibility[str(diameter)] = dict(
                marker_overlap_fraction=report["marker_overlap_fraction"],
                cross_label_overlap_fraction=report["cross_label_overlap_fraction"],
                macro_width_q90_px=report["macro_width_q90_px"],
                populations=[
                    dict(label=item["label"], n_cells=item["n_cells"],
                         cross_label_overlap_fraction=item["cross_label_overlap_fraction"])
                    for item in report["populations"]
                ],
            )
        record["visibility"] = visibility
    return record


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise SystemExit("Moignard CSV hash mismatch.")
    frame = pd.read_csv(SOURCE)
    labels = frame.iloc[:, -1].astype(str).to_numpy()
    features = np.ascontiguousarray(frame.iloc[:, :-1].to_numpy(dtype=np.float64))
    if features.shape != (3934, 42) or digest(features) != FEATURE_SHA:
        raise SystemExit("Moignard feature hash mismatch.")
    if set(labels) != {"HF", "4SFG", "4SG", "PS", "NP"}:
        raise SystemExit("Unexpected Moignard labels.")
    root = feature_medoid(features, labels == ROOT_LABEL)
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    results = []
    for seed in SEEDS:
        for name, initial in (
            ("classical_jitter", classical_jitter(setup, seed)),
            ("random_disk", random_disk(len(features), config.initial_radius, seed)),
        ):
            print(f"poincare {name} seed {seed}", flush=True)
            record = run_condition(setup, features, labels, root, initial, epochs=EPOCHS, geometry="poincare")
            record.update(initialization=name, seed=seed, n_neighbors=config.n_neighbors, teacher="default")
            results.append(record)
    for name, initial in (
        ("classical_jitter", classical_jitter(setup, 0)),
        ("random_disk", random_disk(len(features), config.initial_radius, 0)),
    ):
        print(f"euclidean {name}", flush=True)
        record = run_condition(setup, features, labels, root, initial, epochs=EPOCHS, geometry="euclidean")
        record.update(initialization=name, seed=0, n_neighbors=config.n_neighbors, teacher="default")
        results.append(record)
    print("k sensitivity", flush=True)
    wide = prepare_graph_stress(
        features, config=GraphStressConfig(n_neighbors=K_SENSITIVITY), max_dense_bytes=DENSE_BUDGET,
    )
    record = run_condition(wide, features, labels, root, classical_jitter(wide, 0), epochs=20, geometry="poincare")
    record.update(initialization="classical_jitter", seed=0, n_neighbors=K_SENSITIVITY, teacher="default")
    results.append(record)
    print("degree-aware", flush=True)
    teacher, teacher_meta = degree_aware_teacher(features, config)
    record = run_condition(
        setup, features, labels, root, classical_jitter(setup, 0), epochs=20, geometry="poincare", teacher=teacher,
    )
    record.update(initialization="classical_jitter", seed=0, n_neighbors=config.n_neighbors,
                  teacher="degree_aware", teacher_meta=teacher_meta)
    results.append(record)
    promoted = promotion_allowed(development_input=True, joint_improvements=[True, True])
    payload = dict(
        input_id="moignard2015-development-reference",
        source_sha256=SOURCE_SHA,
        feature_sha256=FEATURE_SHA,
        labels_entered_fitting=False,
        root_rule="feature-space medoid of PS, applied only after fitting",
        root_index=root,
        epochs=EPOCHS,
        budgets=list(BUDGETS),
        seeds=list(SEEDS),
        promoted=promoted,
        biological_validation=False,
        spectral_defaults_changed=False,
        results=results,
    )
    Path("docs/moignard-controlled-probe.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(dict(promoted=promoted, n_results=len(results), root=root), indent=2), flush=True)


if __name__ == "__main__":
    main()
