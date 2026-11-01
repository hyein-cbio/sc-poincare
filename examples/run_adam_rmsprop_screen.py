"""Seed-0 Adam versus Riemannian RMSProp screen, 1,500 updates.

Both arms share GraphStressConfig(), the Poincaré disk, and one classical
jitter per seed. Labels stay out of preparation and fitting. The scored
coordinates are the lowest training loss, including initialization. Seeds
1–4 run only after seed 0 meets every guardrail in
docs/adam-rmsprop-protocol.md. Development data cannot promote a default.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sc_poincare._disk import pairwise_disk_length
from sc_poincare._research_policy import ensure_active_paths
from sc_poincare.controlled_probe import classical_jitter, fit_from_initial, fit_intrinsic_rmsprop, promotion_allowed
from sc_poincare.evaluation import evaluate_expression_neighborhoods, evaluate_neighborhoods
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress

GUARD = 512 * 1024**2
EPOCHS = 1500
CHUNK = 256
RARE_LABELS = ("17Neu", "18Eos", "19Lymph")
STAGE_RANK = {"PS": 0, "NP": 1, "HF": 2, "4SG": 3, "4SFG": 3}
DATASETS = {
    "paul": {
        "features": Path("data/paul/features.npy"),
        "csv": Path("data/paul/Paul.csv"),
        "source_sha256": "fed836ed28f148d9ee8efd8ec13ebd468f06ef9b92eeb6f9e218235d4ad2f21d",
        "feature_sha256": "0386248d2c02c99f82cda2eb18b8e357b8e0399b10aba7e67575599fb115d90b",
        "shape": (2730, 20),
    },
    "moignard": {
        "features": Path("data/moignard/features.npy"),
        "csv": Path("data/moignard/Moignard2015.csv"),
        "source_sha256": "3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1",
        "feature_sha256": "674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0",
        "shape": (3934, 42),
    },
}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def file_sha256(path):
    digest_ = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest_.update(block)
    return digest_.hexdigest()


def load_dataset(name):
    spec = DATASETS[name]
    if file_sha256(spec["csv"]) != spec["source_sha256"]:
        raise SystemExit(f"{name} CSV hash mismatch.")
    frame = pd.read_csv(spec["csv"])
    labels = frame.iloc[:, -1].astype(str).to_numpy()
    features = np.load(spec["features"], allow_pickle=False)
    features = np.ascontiguousarray(features, dtype=np.float64)
    if features.shape != spec["shape"] or digest(features) != spec["feature_sha256"]:
        raise SystemExit(f"{name} feature hash mismatch.")
    if len(labels) != len(features):
        raise SystemExit(f"{name} label count does not match the feature rows.")
    if name == "moignard":
        raw = np.ascontiguousarray(frame.iloc[:, :-1].to_numpy(dtype=np.float64))
        if not np.array_equal(raw, features) or set(labels) != set(STAGE_RANK):
            raise SystemExit("Moignard labels are not aligned with the stored features.")
    else:
        missing = [label for label in RARE_LABELS if label not in set(labels)]
        if missing:
            raise SystemExit(f"Paul labels missing {missing}.")
    return features, labels


def feature_medoid(features, labels):
    rows = np.flatnonzero(labels == "PS")
    distances = np.linalg.norm(features[rows, None, :] - features[None, rows, :], axis=2)
    return int(rows[int(np.argmin(distances.sum(axis=1)))])


def stage_spearman(features, labels, coordinates):
    root = feature_medoid(features, labels)
    distance = pairwise_disk_length(coordinates[root:root + 1], coordinates)[0]
    ranks = np.array([STAGE_RANK[label] for label in labels], dtype=np.float64)
    return float(spearmanr(distance, ranks).statistic), root


def score_coordinates(features, labels, coordinates, dataset):
    neighborhoods = evaluate_neighborhoods(features, coordinates, labels, ks=(15, 30), geometry="poincare")
    expression = evaluate_expression_neighborhoods(features, coordinates, ks=(15, 30), geometry="poincare")
    metrics = neighborhoods.summary["metrics"]
    record = {
        "overall_purity_k15": metrics["15"]["overall_purity"],
        "macro_purity_k15": metrics["15"]["macro_purity"],
        "mse_k15": expression["normalized_mse"]["15"],
        "mse_k30": expression["normalized_mse"]["30"],
        "populations_k15": metrics["15"]["populations"],
    }
    if dataset == "moignard":
        rho, root = stage_spearman(features, labels, coordinates)
        record["stage_spearman"] = rho
        record["ps_medoid"] = root
    else:
        record["rare_purity_k15"] = {
            label: next(item["purity"] for item in metrics["15"]["populations"] if item["label"] == label)
            for label in RARE_LABELS
        }
    return record


def population(record, label):
    return next(item["purity"] for item in record["populations_k15"] if item["label"] == label)


def deltas(adam, rmsprop, dataset):
    change = {
        "overall_purity_k15": rmsprop["overall_purity_k15"] - adam["overall_purity_k15"],
        "macro_purity_k15": rmsprop["macro_purity_k15"] - adam["macro_purity_k15"],
        "mse_k15": rmsprop["mse_k15"] - adam["mse_k15"],
        "mse_k30": rmsprop["mse_k30"] - adam["mse_k30"],
    }
    if dataset == "paul":
        change["rare_purity_k15"] = {
            label: rmsprop["rare_purity_k15"][label] - adam["rare_purity_k15"][label]
            for label in RARE_LABELS
        }
        change["rare_mean"] = float(np.mean(list(change["rare_purity_k15"].values())))
    else:
        change["stage_spearman"] = rmsprop["stage_spearman"] - adam["stage_spearman"]
    return change


def guardrail_passes(change, dataset):
    checks = {
        "overall_purity_floor": change["overall_purity_k15"] >= -0.002,
        "macro_purity_floor": change["macro_purity_k15"] >= -0.002,
        "mse_k15_ceiling": change["mse_k15"] <= 0.001,
        "mse_k30_ceiling": change["mse_k30"] <= 0.001,
    }
    if dataset == "paul":
        checks["macro_gain"] = change["macro_purity_k15"] >= 0.003
        checks["rare_mean_gain"] = change["rare_mean"] >= 0.005
        checks["rare_label_floor"] = all(value >= -0.01 for value in change["rare_purity_k15"].values())
    else:
        checks["mse_k30_reduction"] = -change["mse_k30"] >= 0.001
        checks["stage_gain"] = change["stage_spearman"] >= 0.005
        checks["stage_floor"] = change["stage_spearman"] >= -0.005
    return checks


def compact(record):
    # Keep per-population diagnostics inside scores, matching the saved screen
    # schema. Dropping the duplicate arm-level copy must not discard them.
    return dict(record)


def fit_arm(setup, initial, optimizer):
    started = time.perf_counter()
    if optimizer == "adam":
        fitted = fit_from_initial(
            setup, initial, epochs=EPOCHS, chunk_size=CHUNK, max_dense_bytes=GUARD,
            optimizer_name="adam", progress_every=100,
        )
    elif optimizer == "rmsprop":
        fitted = fit_intrinsic_rmsprop(
            setup, initial, epochs=EPOCHS, chunk_size=CHUNK, max_dense_bytes=GUARD, progress_every=100,
        )
    else:
        raise ValueError(optimizer)
    elapsed = time.perf_counter() - started
    return fitted, elapsed


def main():
    output = Path("docs/adam-rmsprop-screen.json")
    work = Path("data/adam_rmsprop_screen")
    ensure_active_paths(output, work, *(spec["features"] for spec in DATASETS.values()), *(spec["csv"] for spec in DATASETS.values()))
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}.")
    work.mkdir(parents=True)
    config = GraphStressConfig()
    runs = []
    passed = True
    for dataset in ("paul", "moignard"):
        features, labels = load_dataset(dataset)
        print(f"prepare {dataset}", flush=True)
        setup = prepare_graph_stress(features, config=config, max_dense_bytes=GUARD)
        initial = classical_jitter(setup, 0)
        arms = {}
        for optimizer in ("adam", "rmsprop"):
            print(f"fit {dataset} seed 0 {optimizer}", flush=True)
            fitted, elapsed = fit_arm(setup, initial, optimizer)
            np.savez_compressed(
                work / f"{dataset}_seed0_{optimizer}.npz",
                best=fitted["best"], current=fitted["current"], loss_trace=np.asarray(fitted["loss_trace"]),
            )
            scored = score_coordinates(features, labels, fitted["best"], dataset)
            arms[optimizer] = {
                "elapsed_seconds": elapsed,
                "best_epoch": int(fitted["best_epoch"]),
                "best_loss": float(fitted["best_loss"]),
                "final_loss": float(fitted["loss_trace"][-1]),
                "initial_loss": float(fitted["loss_trace"][0]),
                "loss_trace_sha256": digest(np.asarray(fitted["loss_trace"], dtype=np.float64)),
                "best_coordinates_sha256": digest(fitted["best"]),
                "scores": compact(scored),
                "populations_k15": scored["populations_k15"],
            }
            print(f"done {dataset} {optimizer} best_epoch {fitted['best_epoch']} best_loss {fitted['best_loss']:.6f} seconds {elapsed:.1f}", flush=True)
        if abs(arms["adam"]["initial_loss"] - arms["rmsprop"]["initial_loss"]) > 1e-8:
            raise SystemExit(f"{dataset} arms did not share an initial loss.")
        change = deltas(arms["adam"]["scores"], arms["rmsprop"]["scores"], dataset)
        checks = guardrail_passes(change, dataset)
        runs.append({
            "dataset": dataset,
            "seed": 0,
            "n_cells": int(len(features)),
            "feature_sha256": DATASETS[dataset]["feature_sha256"],
            "adam": {key: value for key, value in arms["adam"].items() if key != "populations_k15"},
            "rmsprop": {key: value for key, value in arms["rmsprop"].items() if key != "populations_k15"},
            "rmsprop_minus_adam": change,
            "guardrails": checks,
            "guardrails_pass": all(checks.values()),
        })
        passed = passed and runs[-1]["guardrails_pass"]
        print(json.dumps({"dataset": dataset, "pass": runs[-1]["guardrails_pass"], "checks": checks}, indent=2), flush=True)
    payload = {
        "protocol": "docs/adam-rmsprop-protocol.md",
        "epochs": EPOCHS,
        "seed_selection": 0,
        "confirmation_seeds": [1, 2, 3, 4],
        "confirmation_seeds_executed": False,
        "optimizer_candidate": "riemannian_rmsprop",
        "baseline": "tangent_adam",
        "geometry": "poincare",
        "labels_entered_fitting": False,
        "checkpoint_rule": "lowest_training_loss_including_initialization",
        "max_dense_bytes": GUARD,
        "fits_executed": 4,
        "promoted": promotion_allowed(development_input=True, joint_improvements=[passed]),
        "biological_validation": False,
        "default_changed": False,
        "seed0_guardrails_pass": passed,
        "runs": runs,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"seed0_guardrails_pass": passed, "fits_executed": payload["fits_executed"]}, indent=2), flush=True)
    if not passed:
        return
    for seed in (1, 2, 3, 4):
        for dataset in ("paul", "moignard"):
            features, labels = load_dataset(dataset)
            setup = prepare_graph_stress(features, config=config, max_dense_bytes=GUARD)
            initial = classical_jitter(setup, seed)
            for optimizer in ("adam", "rmsprop"):
                print(f"fit {dataset} seed {seed} {optimizer}", flush=True)
                fitted, elapsed = fit_arm(setup, initial, optimizer)
                np.savez_compressed(
                    work / f"{dataset}_seed{seed}_{optimizer}.npz",
                    best=fitted["best"], current=fitted["current"], loss_trace=np.asarray(fitted["loss_trace"]),
                )
                payload["fits_executed"] += 1
                payload.setdefault("confirmation", []).append({
                    "dataset": dataset,
                    "seed": seed,
                    "optimizer": optimizer,
                    "elapsed_seconds": elapsed,
                    "best_epoch": int(fitted["best_epoch"]),
                    "best_loss": float(fitted["best_loss"]),
                    "final_loss": float(fitted["loss_trace"][-1]),
                    "initial_loss": float(fitted["loss_trace"][0]),
                    "best_coordinates_sha256": digest(fitted["best"]),
                    "scores": compact(score_coordinates(features, labels, fitted["best"], dataset)),
                })
                output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    payload["confirmation_seeds_executed"] = True
    payload["promoted"] = False
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
