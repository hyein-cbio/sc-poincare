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


if __name__ == "__main__":
    main()
