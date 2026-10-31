"""Write-once records for explicit research reruns; never used by model defaults."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from sc_poincare._research_policy import ensure_active_paths

SOURCE_SHA = "3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1"
FEATURE_SHA = "674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0"
SOURCE_URL = ("https://raw.githubusercontent.com/facebookresearch/PoincareMaps/"
              "7ffa518cabe6e6424a875d480c4298bda3bd9678/datasets/Moignard2015.csv")
ROOT = Path(__file__).resolve().parents[2]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def parse_probe_args(kind, argv=None):
    parser = argparse.ArgumentParser(description=f"Write-once Moignard {kind} rerun, 20 epochs, seed 0.")
    parser.add_argument("--source", type=Path, default=Path("data/inputs/Moignard2015.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path(f"docs/moignard-{kind}-probe-rerun-2026-10-30.json"))
    if kind == "lorentz":
        parser.add_argument("--gradient-ablation", action="store_true",
                            help="Also fit with only the pre-PR3 gradient conversion restored.")
    args = parser.parse_args(argv)
    ensure_active_paths(args.source, args.output, args.output.with_suffix(".npz"))
    if args.output.suffix != ".json":
        raise ValueError("Output must end in .json.")
    for path in (args.output, args.output.with_suffix(".npz")):
        # lexists behavior also refuses dangling symlinks. Exclusive creation
        # below repeats this protection against a concurrent writer.
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite research output: {path}")
    for name in ("lorentz", "curvature"):
        archived = ROOT / f"docs/moignard-{name}-probe.json"
        if args.output.resolve() == archived.resolve():
            raise ValueError("Historical probe paths are reserved.")
    return args


def load_moignard(source):
    import pandas as pd
    ensure_active_paths(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Moignard CSV hash mismatch.")
    frame = pd.read_csv(source)
    features = np.ascontiguousarray(frame.iloc[:, :-1].to_numpy(dtype=np.float64))
    if features.shape != (3934, 42) or digest(features) != FEATURE_SHA:
        raise ValueError("Moignard feature hash mismatch.")
    if frame.iloc[:, -1].isna().any():
        raise ValueError("Moignard labels must not be missing.")
    return features, frame.iloc[:, -1].astype(str).to_numpy()


def metrics_for(features, disk, labels):
    from sc_poincare.evaluation import evaluate_neighborhoods
    metrics = evaluate_neighborhoods(features, disk, labels, ks=(15,), geometry="poincare").summary["metrics"]["15"]
    return {"macro_purity": metrics["macro_purity"],
            "input_neighbor_recall": metrics["input_neighbor_recall"],
            "populations": sorted(metrics["populations"], key=lambda item: item["n_cells"])}


def stability(fitted, disk):
    trace = np.asarray(fitted["loss_trace"])
    finite = bool(np.isfinite(trace).all() and np.isfinite(fitted["current"]).all() and np.isfinite(disk).all())
    radius_sq = np.sum(disk * disk, axis=1)
    if not finite or np.any(radius_sq >= 1):
        raise FloatingPointError("Nonfinite result or point outside the open evaluation disk.")
    result = dict(all_losses_and_final_coordinates_finite=finite,
                  max_unit_disk_radius=float(np.sqrt(radius_sq.max())),
                  min_unit_disk_margin=float((1 - radius_sq).min()),
                  loss_increase_steps=int(np.count_nonzero(np.diff(trace) > 0)),
                  loss_trace=trace.tolist())
    if "minkowski_diag" in fitted:
        result["minkowski_max_abs_error"] = float(np.max(np.abs(fitted["minkowski_diag"] + 1)))
        result["min_time_coordinate"] = float(fitted["current"][:, 0].min())
    return result


def provenance(config, initial, *, elapsed_seconds):
    import torch
    from threadpoolctl import threadpool_info
    source_paths = [ROOT / "src/sc_poincare" / name for name in
                    ("controlled_probe.py", "experimental.py", "_fitting.py", "_disk.py", "evaluation.py", "_probe_record.py")]
    source_paths += [ROOT / "examples" / f"run_moignard_{name}_probe.py" for name in ("lorentz", "curvature")]
    return dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        csv_sha256=SOURCE_SHA, source_url=SOURCE_URL, feature_shape=[3934, 42], feature_dtype="float64",
        feature_hash_encoding="C-contiguous float64 raw bytes, without dtype/shape header",
        initial_sha256=digest(initial), config=asdict(config), seed=0, epochs=20, chunk_size=256,
        evaluation_k=15, max_dense_bytes=512 * 1024**2, device="cpu", dtype="float64",
        python=sys.version, platform=platform.platform(),
        versions={name: version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "threadpoolctl")},
        torch_threads=torch.get_num_threads(), threadpools=threadpool_info(), elapsed_seconds=elapsed_seconds,
    )


def write_record(output, payload, arrays):
    ensure_active_paths(output, output.with_suffix(".npz"))
    # Serialize before creating either file. The JSON is the completion record;
    # an interrupted array write leaves a reserved path, never a valid report.
    json.dumps(payload, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    array_path = output.with_suffix(".npz")
    with array_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    payload["arrays"] = dict(file=array_path.name, sha256=hashlib.sha256(array_path.read_bytes()).hexdigest())
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
