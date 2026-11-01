"""Zero-epoch landmark reference on the recovered Paul 2,730×20 matrix.

The stored feature bytes are the recorded SHA256. PCA scores recomputed from
``pca.npz`` are recorded only as a transform residual and are not substituted.
``max_epochs`` is 0. The 512 MiB dense guard stays on. This does not fit.
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

from sc_poincare.experimental import GraphStressConfig
from sc_poincare.landmark import (
    compare_landmark_reference,
    freeze_comparison,
    reevaluate_study_tree,
    write_study_tree,
)
from sc_poincare._research_policy import ensure_active_paths

FEATURE_SHA256 = "0386248d2c02c99f82cda2eb18b8e357b8e0399b10aba7e67575599fb115d90b"
FEATURE_FILE_SHA256 = "0d1d53e52f404a74bc8211b2441174adace7f63cbd57965d91c440552f20fba5"
SOURCE_SHA256 = "fed836ed28f148d9ee8efd8ec13ebd468f06ef9b92eeb6f9e218235d4ad2f21d"
GUARD = 512 * 1024**2
SENSITIVITY = (
    "triangle_excess_max",
    "completion_mean_relative_error",
    "init_procrustes_rmse",
    "completion_gradient_relative_l2",
    "column_objective_gradient_relative_l2",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    features_path = root / "data" / "paul" / "features.npy"
    csv_path = root / "data" / "paul" / "Paul.csv"
    pca_path = root / "data" / "paul" / "pca.npz"
    tree = root / "data" / "paul_landmark_reference"
    report_path = root / "docs" / "paul-landmark-reference.json"
    restore_path = root / "docs" / "paul-landmark-restore.json"
    ensure_active_paths(features_path, csv_path, pca_path, tree, report_path, restore_path)
    for path in (report_path, restore_path):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}.")
    if _sha256(csv_path) != SOURCE_SHA256:
        raise SystemExit("Paul.csv does not match the recorded source hash.")
    if _sha256(features_path) != FEATURE_FILE_SHA256:
        raise SystemExit("features.npy file hash does not match the recorded feature file.")
    features = np.load(features_path, allow_pickle=False)
    if _array_sha256(features) != FEATURE_SHA256:
        raise SystemExit("Paul feature bytes do not match the recorded array hash.")
    with np.load(pca_path) as pca:
        components = np.asarray(pca["components"], dtype=np.float64)
        mean = np.asarray(pca["mean"], dtype=np.float64)
    import pandas as pd
    matrix = pd.read_csv(csv_path).iloc[:, :-1].to_numpy(dtype=np.float64)
    recomputed = np.ascontiguousarray((matrix - mean) @ components.T, dtype=np.float64)
    transform_max_abs = float(np.max(np.abs(recomputed - features)))
    recomputed_sha = _array_sha256(recomputed)
    # Matching reconstruction is valid too. Always retain the independently
    # pinned features.npy bytes rather than substituting the reconstruction.
    config = GraphStressConfig()
    protocol = freeze_comparison(
        features,
        config,
        input_id="paul-development-reference",
        role="reference_check",
        n_landmarks=32,
        max_dense_bytes=GUARD,
        max_epochs=0,
        seeds=(0,),
        primary_metric="landmark_column_max_abs",
        sensitivity_metrics=SENSITIVITY,
    )
    if protocol.feature_sha256 != FEATURE_SHA256 or protocol.max_epochs != 0:
        raise SystemExit("Frozen Paul protocol does not match the recorded matrix or the zero-epoch rule.")
    started = time.perf_counter()
    reference = compare_landmark_reference(features, protocol, config)
    elapsed = time.perf_counter() - started
    reference.report["elapsed_seconds"] = elapsed
    reference.report["source_sha256"] = SOURCE_SHA256
    reference.report["feature_array_sha256_checked"] = FEATURE_SHA256
    reference.report["fits_executed"] = 0
    write_study_tree(tree, features, reference)
    restored = reevaluate_study_tree(tree)
    if restored["primary_abs_difference"] != 0.:
        raise SystemExit(f"Paul restore changed the primary value by {restored['primary_abs_difference']}.")
    import numpy
    import pandas
    import scipy
    import sklearn
    payload = dict(reference.report)
    payload["transform_pin"] = {
        "pca_file_sha256": _sha256(pca_path),
        "recomputed_from": "(X - mean) @ components.T using data/paul/pca.npz",
        "recomputed_array_sha256": recomputed_sha,
        "recomputed_scores_used": False,
        "max_abs_residual_versus_recorded_features": transform_max_abs,
    }
    payload["runtime"] = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__,
    }
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    restore_path.write_text(json.dumps(dict(
        restored,
        source_sha256=SOURCE_SHA256,
        tree="data/paul_landmark_reference",
        external_backup=False,
        off_machine_destination=None,
    ), indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "primary_value": payload["primary_value"],
        "elapsed_seconds": elapsed,
        "column_gradient": payload["metrics"]["column_objective_gradient_relative_l2"],
        "transform_max_abs": transform_max_abs,
        "restore_difference": restored["primary_abs_difference"],
    }, indent=2))


if __name__ == "__main__":
    main()
