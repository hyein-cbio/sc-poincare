"""Predeclared landmark-gradient tolerance on the exact Moignard matrix.

Written before the schedule runs. Tolerance is relative L2 1e-3 against the
exact MU endpoint gradient at the classical initializer. The recorded
32-landmark full-denominator value 0.875 is already above that tolerance, so
it is a negative control, not a pass. Counts are 32, 128, 512, 2048, and every
cell. Reductions are ``full_denominator`` and ``included_mean``. The 512 MiB
guard stays on. No epoch is taken and no default changes. A miss is not refit
with a new tolerance.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress
from sc_poincare.landmark import _farthest_landmarks, _transformed_mutual, column_gradient_relative_l2
from sc_poincare._research_policy import ensure_active_paths

FEATURE_SHA256 = "674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0"
SOURCE_SHA256 = "3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1"
GUARD = 512 * 1024**2
TOLERANCE = 1e-3
COUNTS = (32, 128, 512, 2048)
REDUCTIONS = ("full_denominator", "included_mean")
PUBLISHED_32 = 0.8750312252544279


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    features_path = root / "data" / "moignard_landmark_reference" / "features.npy"
    csv_path = root / "data" / "moignard" / "Moignard2015.csv"
    output = root / "docs" / "landmark-gradient-tolerance.json"
    ensure_active_paths(features_path, csv_path, output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}.")
    if _sha256(csv_path) != SOURCE_SHA256:
        raise SystemExit("Moignard CSV does not match the recorded source hash.")
    features = np.load(features_path, allow_pickle=False)
    array_sha = hashlib.sha256(np.ascontiguousarray(features, dtype=np.float64).tobytes()).hexdigest()
    if array_sha != FEATURE_SHA256:
        raise SystemExit("Moignard features do not match the recorded hash.")
    config = GraphStressConfig()
    started = time.perf_counter()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=GUARD)
    indices, distances = _farthest_landmarks(
        _transformed_mutual(np.ascontiguousarray(features, dtype=np.float64), config.n_neighbors),
        len(features),
        0,
    )
    rows = []
    for count in (*COUNTS, len(features)):
        for reduction in REDUCTIONS:
            value = column_gradient_relative_l2(
                setup,
                distances[:count],
                indices[:count],
                reduction=reduction,
            )
            rows.append({
                "n_landmarks": int(count),
                "reduction": reduction,
                "column_gradient_relative_l2": value,
                "meets_tolerance": bool(value <= TOLERANCE),
            })
            print(rows[-1], flush=True)
    elapsed = time.perf_counter() - started
    control = next(row for row in rows if row["n_landmarks"] == 32 and row["reduction"] == "full_denominator")
    if abs(control["column_gradient_relative_l2"] - PUBLISHED_32) > 1e-8:
        raise SystemExit(
            "32-landmark full-denominator gradient did not reproduce the published reference: "
            f"{control['column_gradient_relative_l2']}"
        )
    passing = [row for row in rows if row["meets_tolerance"]]
    payload = {
        "protocol": {
            "input_id": "moignard2015-development-reference",
            "feature_sha256": FEATURE_SHA256,
            "source_sha256": SOURCE_SHA256,
            "n_cells": int(len(features)),
            "tolerance_relative_l2": TOLERANCE,
            "counts": [*COUNTS, len(features)],
            "reductions": list(REDUCTIONS),
            "max_dense_bytes": GUARD,
            "max_epochs": 0,
            "landmark_seed": 0,
            "initializer": "exact_classical_scaling",
            "labels_excluded": True,
            "declared_before_schedule": True,
        },
        "published_32_full_denominator": PUBLISHED_32,
        "published_32_abs_difference": abs(control["column_gradient_relative_l2"] - PUBLISHED_32),
        "rows": rows,
        "any_partial_landmark_meets_tolerance": any(
            row["meets_tolerance"] and row["n_landmarks"] < len(features) for row in rows
        ),
        "all_cell_rows_meet_tolerance": all(
            row["meets_tolerance"] for row in rows if row["n_landmarks"] == len(features)
        ),
        "n_passing_rows": len(passing),
        "elapsed_seconds": elapsed,
        "fits_executed": 0,
        "biological_validation": False,
        "default_changed": False,
        "exact_guard_disabled": False,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "elapsed_seconds": elapsed,
        "any_partial": payload["any_partial_landmark_meets_tolerance"],
        "all_cell": payload["all_cell_rows_meet_tolerance"],
        "n_passing_rows": len(passing),
    }, indent=2))


if __name__ == "__main__":
    main()
