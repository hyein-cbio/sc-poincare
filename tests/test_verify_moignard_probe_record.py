"""Saved-record integrity checks, using tiny coordinates without optimization."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sc_poincare import controlled_probe, experimental
from sc_poincare._probe_record import ROOT, digest, metrics_for, stability, write_record


CONDITIONS = ("lorentz", "gradient_ablation", "c1_0", "c0_25")
FAULTS = ("json_trace", "npz_trace", "trace_shape", "sidecar_hash", "source_hash",
          "initial_hash", "initial_values", "population", "stability", "final_loss",
          "initial_loss", "objective")


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_moignard_probe_record", ROOT / "examples/verify_moignard_probe_record.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(folder, condition):
    folder.mkdir(parents=True, exist_ok=True)
    features = np.random.default_rng(19).normal(size=(19, 4))
    labels = np.array(["A"] * 9 + ["B"] * 10)
    setup = experimental.prepare_graph_stress(
        features, config=experimental.GraphStressConfig(n_neighbors=4))
    initial = controlled_probe.classical_jitter(setup, 0)
    target = torch.tensor(np.array(setup.target), dtype=torch.float64)
    teacher = torch.tensor(np.array(setup.teacher), dtype=torch.float64)
    arrays = dict(initial=initial)
    rows = []
    is_lorentz = condition in CONDITIONS[:2]
    for index in range(2):
        disk = initial * (1 + index * .1)
        if is_lorentz:
            current = controlled_probe.lorentz_from_disk(disk)
            disk = controlled_probe.disk_from_lorentz(current)
            loss = controlled_probe._probe_objective(
                torch.tensor(current), target, teacher, setup.config, 256, lorentz=True)
            key = "current" if index == 0 else "legacy_gradient_current"
            trace_key = "loss_trace" if index == 0 else "legacy_gradient_loss_trace"
        else:
            curvature = (1., .25)[index]
            current = disk / np.sqrt(curvature)
            disk = np.sqrt(curvature) * current
            tangent = controlled_probe._tangent_from_disk_curvature(current, curvature)
            loss, _, _ = controlled_probe._objective_curvature(
                torch.tensor(tangent), target, teacher, setup.config, 256, curvature, backward=False)
            suffix = str(curvature).replace(".", "_")
            key, trace_key = "current_c" + suffix, "loss_trace_c" + suffix
        # Deliberately synthetic history; no fitting or biological experiment.
        trace = np.array([loss + .02, loss + .01, loss])
        arrays.update({key: current, trace_key: trace})
        fitted = dict(current=current, loss_trace=trace)
        if is_lorentz:
            fitted["minkowski_diag"] = -current[:, 0] ** 2 + (current[:, 1:] ** 2).sum(1)
        row = dict(loss_at_0=trace[0], loss_at_20=loss,
                   **metrics_for(features, disk, labels), stability=stability(fitted, disk))
        if not is_lorentz:
            row.update(curvature=curvature, legacy_unit_curvature_evaluation=metrics_for(features, current, labels))
        rows.append(row)
    report = copy.deepcopy(rows[0]) if is_lorentz else dict(results=rows)
    if is_lorentz:
        report["legacy_gradient_only_ablation"] = rows[1]
    source = "src/sc_poincare/_probe_record.py"
    report["provenance"] = dict(initial_sha256=digest(initial), source_sha256={
        source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()})
    record = folder / "record.json"
    write_record(record, report, arrays)
    return record, features, labels, setup


def damage(case, condition, fault):
    record = case[0]
    report = json.loads(record.read_text())
    if condition == "lorentz":
        row, trace_key = report, "loss_trace"
    elif condition == "gradient_ablation":
        row, trace_key = report["legacy_gradient_only_ablation"], "legacy_gradient_loss_trace"
    else:
        row = report["results"][0 if condition == "c1_0" else 1]
        trace_key = "loss_trace_" + condition
    if fault == "json_trace":
        # Keep endpoints and all summary diagnostics unchanged.
        row["stability"]["loss_trace"][1] += .001
    elif fault in ("npz_trace", "trace_shape", "initial_values"):
        sidecar = record.with_suffix(".npz")
        with np.load(sidecar, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if fault == "npz_trace":
            arrays[trace_key][1] += .001
        elif fault == "trace_shape":
            arrays[trace_key] = arrays[trace_key][:, None]
        else:
            arrays["initial"][0, 0] += .001
            report["provenance"]["initial_sha256"] = digest(arrays["initial"])
        np.savez_compressed(sidecar, **arrays)
        report["arrays"]["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    elif fault == "sidecar_hash":
        report["arrays"]["sha256"] = "0" * 64
    elif fault == "source_hash":
        report["provenance"]["source_sha256"] = {"src/sc_poincare/_probe_record.py": "0" * 64}
    elif fault == "initial_hash":
        report["provenance"]["initial_sha256"] = "0" * 64
    elif fault == "population":
        row["populations"][0]["n_cells"] += 1
    elif fault == "stability":
        row["stability"]["loss_increase_steps"] += 1
    elif fault == "final_loss":
        row["loss_at_20"] += .1
    elif fault == "initial_loss":
        row["loss_at_0"] += .1
    elif fault == "objective":
        # Keep both saved traces and endpoint summary consistent, but wrong
        # relative to the independently recomputed coordinate objective.
        sidecar = record.with_suffix(".npz")
        with np.load(sidecar, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays[trace_key][-1] += .1
        row["stability"]["loss_trace"][-1] += .1
        row["loss_at_20"] += .1
        np.savez_compressed(sidecar, **arrays)
        report["arrays"]["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    elif fault == "legacy_metrics":
        row["legacy_unit_curvature_evaluation"]["macro_purity"] += .1
    else:
        raise ValueError(f"Unknown test fault: {fault}")
    record.write_text(json.dumps(report))


@pytest.mark.parametrize("condition", CONDITIONS)
def test_valid_record(condition, tmp_path):
    result = load_verifier().verify(*make_case(tmp_path, condition))
    assert len(result["checks"]) == 2
    assert all(check["loss_trace_exact"] and check["stability_exact"] and check["all_population_metrics_exact"]
               for check in result["checks"])


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("fault", FAULTS)
def test_tampered_record_fails(condition, fault, tmp_path):
    case = make_case(tmp_path, condition)
    damage(case, condition, fault)
    with pytest.raises(ValueError, match="mismatch"):
        load_verifier().verify(*case)


@pytest.mark.parametrize("condition", CONDITIONS[2:])
def test_legacy_curvature_metrics_checked(condition, tmp_path):
    case = make_case(tmp_path, condition)
    damage(case, condition, "legacy_metrics")
    with pytest.raises(ValueError, match="mismatch"):
        load_verifier().verify(*case)


@pytest.mark.parametrize("mode", ["flag", "environment"])
def test_checks_survive_optimized_python(mode, tmp_path):
    # Run the actual verifier in a fresh interpreter, not pytest's rewritten asserts.
    code = """
import pathlib, runpy, sys
helpers = runpy.run_path(sys.argv[1])
verify = helpers['load_verifier']().verify
for condition in helpers['CONDITIONS']:
    for fault in (None, *helpers['FAULTS'], *(['legacy_metrics'] if condition.startswith('c') else [])):
        folder = pathlib.Path(sys.argv[2]) / condition / str(fault)
        case = helpers['make_case'](folder, condition)
        if fault is not None:
            helpers['damage'](case, condition, fault)
        try:
            result = verify(*case)
        except ValueError as error:
            if fault is None or 'mismatch' not in str(error):
                raise
        else:
            if fault is not None:
                raise RuntimeError(f'Accepted tampering: {condition} {fault}')
            if len(result['checks']) != 2:
                raise RuntimeError('Incomplete success report')
print('All optimized integrity checks passed')
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONOPTIMIZE", None)
    if mode == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    command = [sys.executable, *(["-O"] if mode == "flag" else []), "-c", code,
               str(Path(__file__).resolve()), str(tmp_path)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All optimized integrity checks passed" in result.stdout
