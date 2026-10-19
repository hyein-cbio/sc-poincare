"""Frozen Moignard optimizer and multiscale probes. Neither is a new default.

The feature hash is checked before any target is built. Labels are not read.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sc_poincare.controlled_probe import (
    classical_jitter,
    fit_from_initial,
    fixed_multiscale_accessibility,
    promotion_allowed,
)
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress
from sc_poincare._fitting import fit_disk_embedding
from sc_poincare._proximity import forest_accessibility, local_affinity

SOURCE = Path("/tmp/sc-poincare-inputs/Moignard2015.csv")
SOURCE_SHA = "3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1"
FEATURE_SHA = "674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0"
DENSE_BUDGET = 512 * 1024**2


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise SystemExit("Moignard CSV hash mismatch.")
    features = np.ascontiguousarray(pd.read_csv(SOURCE).iloc[:, :-1].to_numpy(dtype=np.float64))
    if features.shape != (3934, 42) or digest(features) != FEATURE_SHA:
        raise SystemExit("Moignard feature hash mismatch.")
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    initial = classical_jitter(setup, 0)
    print("adam", flush=True)
    adam = fit_from_initial(setup, initial, epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET, optimizer_name="adam")
    print("sgd", flush=True)
    sgd = fit_from_initial(setup, initial, epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET, optimizer_name="sgd")
    print("affinity", flush=True)
    weights, _lengths = local_affinity(features, 15, "median")
    started = time.perf_counter()
    base = forest_accessibility(weights, max_dense_bytes=DENSE_BUDGET, forest_tau=1.)
    base_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mixed = fixed_multiscale_accessibility(weights, max_dense_bytes=DENSE_BUDGET)
    mixed_seconds = time.perf_counter() - started
    print("rfa fits", flush=True)
    base_fit = fit_disk_embedding(base, connectivities=weights, n_epochs=5, random_state=0, max_dense_bytes=DENSE_BUDGET)
    mixed_fit = fit_disk_embedding(mixed, connectivities=weights, n_epochs=5, random_state=0, max_dense_bytes=DENSE_BUDGET)
    promoted = promotion_allowed(development_input=True, joint_improvements=[True])
    payload = dict(
        input_id="moignard2015-development-reference",
        feature_sha256=FEATURE_SHA,
        labels_entered_fitting=False,
        promoted=promoted,
        biological_validation=False,
        optimizer=dict(
            learning_rate=config.learning_rate,
            epochs=20,
            seed=0,
            adam_loss_at_20=adam["loss_trace"][20],
            sgd_loss_at_20=sgd["loss_trace"][20],
            adam_loss_at_0=adam["loss_trace"][0],
            sgd_loss_at_0=sgd["loss_trace"][0],
        ),
        multiscale=dict(
            rule="equal mixture of forest_tau 0.1 and 10, then row-normalized",
            tau_one_seconds=base_seconds,
            mixture_seconds=mixed_seconds,
            max_abs_difference_from_tau_one=float(np.max(np.abs(mixed - base))),
            mixture_row_sum_error=float(np.max(np.abs(mixed.sum(1) - 1.))),
            tau_one_loss_trace=[float(value) for value in base_fit.loss_trace],
            mixture_loss_trace=[float(value) for value in mixed_fit.loss_trace],
        ),
    )
    Path("docs/moignard-mechanism-probe.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(dict(promoted=promoted, adam=payload["optimizer"]["adam_loss_at_20"],
                          sgd=payload["optimizer"]["sgd_loss_at_20"],
                          target_gap=payload["multiscale"]["max_abs_difference_from_tau_one"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
