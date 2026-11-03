"""Reread factorial checkpoints and objectives without refitting any condition."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sc_poincare import _lorentz_ablation as ab, controlled_probe as cp, experimental
from sc_poincare._probe_record import ROOT, digest, load_moignard, metrics_for, stability


def verify(record, features, labels, setup):
    report = json.loads(record.read_text())
    if report["schema"] != "lorentz-distance-gradient-factorial-v1":
        raise ValueError("Unknown factorial schema")
    if report["biological_validation"] or report["promoted"] or report["labels_entered_fitting"]:
        raise ValueError("Invalid development-data claims")
    sidecar = record.with_name(report["arrays"]["file"])
    if hashlib.sha256(sidecar.read_bytes()).hexdigest() != report["arrays"]["sha256"]:
        raise ValueError("NPZ SHA256 mismatch")
    for relative, expected in report["provenance"]["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source SHA256 mismatch: {relative}")
    if report["provenance"]["config"] != asdict(setup.config):
        raise ValueError("Configuration mismatch")
    if (report["historical_source_commit"] != ab.HISTORICAL_COMMIT
            or report["historical_function_sha256"] != ab.HISTORICAL_FUNCTION_SHA256):
        raise ValueError("Historical source mismatch")
    initial = cp.classical_jitter(setup, 0)
    hashes = dict(initial=digest(initial), target=digest(setup.target), teacher=digest(setup.teacher))
    if hashes != report["shared_sha256"]:
        raise ValueError("Shared input hashes mismatch")
    target = torch.tensor(np.array(setup.target), dtype=torch.float64)
    teacher = torch.tensor(np.array(setup.teacher), dtype=torch.float64)
    checks = []
    expected = {d + "__" + g for d in ab.DISTANCES for g in ab.GRADIENTS}
    if len(report["results"]) != len(expected) or {r["condition"] for r in report["results"]} != expected:
        raise ValueError("Incomplete or duplicate factorial conditions")
    with np.load(sidecar, allow_pickle=False) as saved:
        if not np.array_equal(saved["initial"], initial):
            raise ValueError("Initialization mismatch")
        for row in report["results"]:
            key = row["condition"]
            if row["biological_validation"] or row["promoted"]:
                raise ValueError(f"Invalid development-data claims: {key}")
            if key != row["distance_mode"] + "__" + row["gradient_mode"] or row["shared_sha256"] != hashes:
                raise ValueError("Condition or shared input mismatch")
            trace, current = saved[key + "__loss_trace"], saved[key + "__current"]
            completed = row["epochs_completed"]
            if (trace.shape != (completed + 1,) or not np.isfinite(trace).all()
                    or trace[0] != row["loss_at_0"] or trace[-1] != row["loss_at_last_checkpoint"]):
                raise ValueError("Loss trace endpoints/shape mismatch")
            if row["requested_epochs"] != 20 or row["backward_mode"] != "chunked":
                raise ValueError("Factorial budget mismatch")
            if row["status"] == "completed":
                if completed != 20 or row["failure"] is not None or row["loss_at_20"] != trace[-1]:
                    raise ValueError("Completed budget mismatch")
            elif row["status"] == "failed_numerical":
                if row["failure"] is None or row["loss_at_20"] is not None or completed >= 20:
                    raise ValueError("Failure incorrectly labeled as a completed fit")
            else:
                raise ValueError("Unknown condition status")
            initial_loss = ab.objective(torch.tensor(cp.lorentz_from_disk(initial)), target, teacher,
                                        setup.config, 256, distance_mode=row["distance_mode"])
            if not np.isclose(initial_loss, trace[0], rtol=0, atol=1e-12):
                raise ValueError("Initial objective mismatch")
            errors = {}
            for mode in ab.DISTANCES:
                saved_loss = row["final_objectives"][mode]
                if not np.isfinite(saved_loss):
                    raise ValueError(f"Nonfinite saved objective: {key}/{mode}")
                loss = ab.objective(torch.tensor(current), target, teacher, setup.config, 256, distance_mode=mode)
                errors[mode] = abs(loss - saved_loss)
                if errors[mode] > 1e-12:
                    raise ValueError("Cross-objective mismatch")
                if mode == row["distance_mode"] and not np.isclose(loss, trace[-1], rtol=0, atol=1e-12):
                    raise ValueError("Final fitted objective mismatch")
            disk = cp.disk_from_lorentz(current)
            actual = metrics_for(features, disk, labels)
            if actual != {name: row[name] for name in actual}:
                raise ValueError("Population metrics mismatch")
            minkowski = -current[:, 0] ** 2 + (current[:, 1:] ** 2).sum(1)
            if stability(dict(current=current, loss_trace=trace, minkowski_diag=minkowski), disk) != row["stability"]:
                raise ValueError("Stability/trace mismatch")
            # A failed first backward must be reproducible on its retained point.
            if row["failure"] and row["failure"]["stage"] == "backward":
                point = torch.tensor(current, requires_grad=True)
                ab.objective(point, target, teacher, setup.config, 256,
                             distance_mode=row["distance_mode"], backward=True)
                if int((~torch.isfinite(point.grad)).sum()) != row["failure"]["nonfinite_gradient_entries"]:
                    raise ValueError("Failure gradient mismatch")
            checks.append(dict(condition=key, status=row["status"], cross_objective_absolute_errors=errors,
                               population_metrics_exact=True, stability_and_trace_exact=True))
        historical = report["historical_preflight"]
        if historical is not None:
            current, trace = saved["historical_preflight_current"], saved["historical_preflight_loss_trace"]
            finite = bool(np.isfinite(current).all() and np.isfinite(trace).all())
            if (finite != historical["finite"] or trace.shape != (2,)
                    or trace[0] != historical["loss_at_0"]
                    or int((~np.isfinite(current)).sum()) != historical["nonfinite_coordinate_entries"]
                    or int((~np.isfinite(trace)).sum()) != historical["nonfinite_loss_entries"]):
                raise ValueError("Historical preflight mismatch")
            if historical["full_20_epochs_attempted"]:
                current, trace = saved["historical_replay_current"], saved["historical_replay_loss_trace"]
                finite = bool(np.isfinite(current).all() and np.isfinite(trace).all())
                if trace.shape != (21,) or finite != historical["full_20_epochs_finite"]:
                    raise ValueError("Historical replay mismatch")
                if finite:
                    row = historical["full_20_epochs_summary"]
                    disk = cp.disk_from_lorentz(current)
                    minkowski = -current[:, 0] ** 2 + (current[:, 1:] ** 2).sum(1)
                    if (row["stability"] != stability(dict(current=current, loss_trace=trace, minkowski_diag=minkowski), disk)
                            or row["loss_at_0"] != trace[0] or row["loss_at_20"] != trace[-1]):
                        raise ValueError("Historical replay trace/stability mismatch")
                    actual = metrics_for(features, disk, labels)
                    if actual != {name: row[name] for name in actual}:
                        raise ValueError("Historical replay metrics mismatch")
                    loss = ab.objective(torch.tensor(current), target, teacher, setup.config, 256, distance_mode="acosh_exact")
                    if not np.isclose(loss, trace[-1], rtol=0, atol=1e-12):
                        raise ValueError("Historical replay objective mismatch")

    return dict(record=record.name, checks=checks, historical_preflight_checked=historical is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    features, labels = load_moignard(args.source)
    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig(),
                                              max_dense_bytes=512 * 1024**2)
    print(json.dumps(verify(args.record, features, labels, setup), indent=2))


if __name__ == "__main__":
    main()
