"""Frozen 20-epoch Lorentz rerun. Historical records are never overwritten."""
from __future__ import annotations

from time import perf_counter
from unittest.mock import patch


from sc_poincare import controlled_probe
from sc_poincare.controlled_probe import classical_jitter, disk_from_lorentz, fit_lorentz, promotion_allowed
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress
from sc_poincare._probe_record import (
    FEATURE_SHA, load_moignard, metrics_for, parse_probe_args,
    provenance, stability, write_record,
)

DENSE_BUDGET = 512 * 1024**2


def legacy_gradient_only(point, gradient):
    """Pre-13ce854 projection, for a labeled ablation with current distances.

    This is deliberately incorrect: the time component lacks the metric sign
    flip. The scoped patch never changes package source or the default fitter.
    """
    return gradient + controlled_probe._minkowski(point, gradient)[:, None] * point


def summarize(fitted, features, labels):
    disk = disk_from_lorentz(fitted["current"])
    return dict(loss_at_0=fitted["loss_trace"][0], loss_at_20=fitted["loss_trace"][20],
                **metrics_for(features, disk, labels), stability=stability(fitted, disk))


def main(argv=None):
    args = parse_probe_args("lorentz", argv)
    if args.distance_ablation:
        run_factorial(args)
        return
    start = perf_counter()
    features, labels = load_moignard(args.source)
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    initial = classical_jitter(setup, 0)
    print("lorentz corrected, 20 epochs", flush=True)
    fitted = fit_lorentz(setup, initial, epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET)
    payload = dict(
        input_id="moignard2015-development-reference", feature_sha256=FEATURE_SHA,
        labels_entered_fitting=False, chart="lorentz_hyperboloid", epochs=20,
        learning_rate=config.learning_rate, **summarize(fitted, features, labels),
        promoted=promotion_allowed(development_input=True, joint_improvements=[True]), biological_validation=False,
    )
    arrays = dict(initial=initial, current=fitted["current"], loss_trace=fitted["loss_trace"])
    if args.gradient_ablation:
        print("lorentz legacy-gradient-only ablation, current distance objective, 20 epochs", flush=True)
        with patch.object(controlled_probe, "_lorentz_tangent_gradient", legacy_gradient_only):
            ablated = fit_lorentz(setup, initial, epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET)
        payload["legacy_gradient_only_ablation"] = dict(
            interpretation="Only gradient conversion restored; current stable distance and chunking retained. Not a historical replay.",
            **summarize(ablated, features, labels),
        )
        arrays.update(legacy_gradient_current=ablated["current"], legacy_gradient_loss_trace=ablated["loss_trace"])
    payload["provenance"] = provenance(config, initial, elapsed_seconds=perf_counter() - start)
    write_record(args.output, payload, arrays)
    print(f"{args.output}: loss {payload['loss_at_0']:.12f} -> {payload['loss_at_20']:.12f}", flush=True)


def run_factorial(args):
    """Shared preparation and write-once output for the six isolated arms."""
    import hashlib
    import numpy as np
    import torch
    from sc_poincare import _lorentz_ablation as ab
    from sc_poincare._probe_record import ROOT, digest

    start = perf_counter()
    features, labels = load_moignard(args.source)
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    initial = classical_jitter(setup, 0)
    target = torch.tensor(np.array(setup.target), dtype=torch.float64)
    teacher = torch.tensor(np.array(setup.teacher), dtype=torch.float64)
    hashes = dict(initial=digest(initial), target=digest(setup.target), teacher=digest(setup.teacher))
    arrays = dict(initial=initial)
    rows = []
    for distance in ab.DISTANCES:
        for gradient in ab.GRADIENTS:
            key = distance + "__" + gradient
            print(key + ": 20 epochs", flush=True)
            fitted = ab.fit_control(setup, initial, distance_mode=distance, gradient_mode=gradient,
                                    epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET)
            # A failed backward retains only its last finite checkpoint; it is
            # never labeled as a 20-update result.
            current, trace = fitted["current"], fitted["loss_trace"]
            cross = {mode: ab.objective(torch.tensor(current), target, teacher, config, 256,
                                       distance_mode=mode) for mode in ab.DISTANCES}
            row = {name: fitted[name] for name in (
                "status", "failure", "distance_mode", "gradient_mode", "backward_mode",
                "requested_epochs", "epochs_completed", "trajectory_diagnostics")}
            row.update(condition=key, shared_sha256=hashes.copy(),
                       loss_at_0=trace[0], loss_at_last_checkpoint=trace[-1],
                       loss_at_20=trace[-1] if fitted["status"] == "completed" else None,
                       final_objectives=cross, **metrics_for(features, disk_from_lorentz(current), labels),
                       stability=stability(fitted, disk_from_lorentz(current)),
                       biological_validation=False, promoted=False)
            rows.append(row)
            arrays[key + "__current"] = current
            arrays[key + "__loss_trace"] = np.asarray(trace)
            print(f"{key}: {row['status']}, {row['epochs_completed']} updates, {trace[-1]:.12f}", flush=True)
    historical = None
    if args.historical_replay:
        import importlib.util
        import ast
        path = ROOT / "examples/fixtures/pre_pr3_lorentz.py"
        text = path.read_text()
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "fit_lorentz")
        function = "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
        if hashlib.sha256(function.encode()).hexdigest() != ab.HISTORICAL_FUNCTION_SHA256:
            raise ValueError("Historical function hash mismatch")
        spec = importlib.util.spec_from_file_location("pre_pr3_lorentz", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("Verbatim historical fitter: one-update full-backward preflight", flush=True)
        fitted = module.fit_lorentz(setup, initial, epochs=1, chunk_size=256, max_dense_bytes=DENSE_BUDGET)
        finite = bool(np.isfinite(fitted["current"]).all() and np.isfinite(fitted["loss_trace"]).all())
        historical = dict(requested_epochs=1, finite=finite,
                          nonfinite_coordinate_entries=int((~np.isfinite(fitted["current"])).sum()),
                          nonfinite_loss_entries=int((~np.isfinite(fitted["loss_trace"])).sum()),
                          loss_at_0=fitted["loss_trace"][0],
                          note="Verbatim historical function/full backward, current unchanged helper dependencies; not the historical runtime.")
        arrays["historical_preflight_current"] = fitted["current"]
        arrays["historical_preflight_loss_trace"] = np.asarray(fitted["loss_trace"])
        historical["full_20_epochs_attempted"] = finite
        if finite:
            print("Verbatim historical fitter: 20 epochs, full backward", flush=True)
            replay = module.fit_lorentz(setup, initial, epochs=20, chunk_size=256, max_dense_bytes=DENSE_BUDGET)
            replay_finite = bool(np.isfinite(replay["current"]).all() and np.isfinite(replay["loss_trace"]).all())
            historical["full_20_epochs_finite"] = replay_finite
            arrays["historical_replay_current"] = replay["current"]
            arrays["historical_replay_loss_trace"] = np.asarray(replay["loss_trace"])
            if replay_finite:
                historical["full_20_epochs_summary"] = summarize(replay, features, labels)

    prov = provenance(config, initial, elapsed_seconds=perf_counter() - start)
    for relative in ("src/sc_poincare/_lorentz_ablation.py", "examples/fixtures/pre_pr3_lorentz.py",
                     "examples/verify_moignard_lorentz_factorial.py"):
        prov["source_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    payload = dict(schema="lorentz-distance-gradient-factorial-v1", input_id="moignard2015-development-reference",
                   feature_sha256=FEATURE_SHA, labels_entered_fitting=False,
                   epochs=20, shared_sha256=hashes, results=rows, historical_preflight=historical,
                   historical_source_commit=ab.HISTORICAL_COMMIT,
                   historical_function_sha256=ab.HISTORICAL_FUNCTION_SHA256,
                   interpretation="Exact old distance operations with current chunked backward; masked acosh is a stabilized reconstruction, never a historical replay.",
                   biological_validation=False, promoted=False, provenance=prov)
    write_record(args.output, payload, arrays)
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
