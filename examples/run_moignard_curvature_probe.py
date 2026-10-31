"""Frozen curvature rerun with matching geometry and evaluation-only control.

Own losses are not a common score across curvatures. Labels enter only evaluation.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from sc_poincare.controlled_probe import classical_jitter, fit_from_initial, promotion_allowed
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress
from sc_poincare._probe_record import (
    FEATURE_SHA, load_moignard, metrics_for, parse_probe_args,
    provenance, stability, write_record,
)

CURVATURES = (1., 0.25)
DENSE_BUDGET = 512 * 1024**2


def main(argv=None):
    args = parse_probe_args("curvature", argv)
    start = perf_counter()
    features, labels = load_moignard(args.source)
    config = GraphStressConfig()
    setup = prepare_graph_stress(features, config=config, max_dense_bytes=DENSE_BUDGET)
    initial = classical_jitter(setup, 0)
    results, arrays = [], dict(initial=initial)
    for curvature in CURVATURES:
        print(f"curvature {curvature}, 20 epochs", flush=True)
        fitted = fit_from_initial(setup, initial, epochs=20, chunk_size=256,
                                  max_dense_bytes=DENSE_BUDGET, curvature=curvature)
        # d_c(x,y) = d_1(sqrt(c)*x,sqrt(c)*y)/sqrt(c).
        # The positive constant leaves neighbor ranks unchanged.
        disk = np.sqrt(curvature) * fitted["current"]
        legacy_metrics = (metrics_for(features, fitted["current"], labels)
                          if np.all(np.sum(fitted["current"] ** 2, axis=1) < 1) else None)
        results.append(dict(
            curvature=curvature, loss_at_0=fitted["loss_trace"][0], loss_at_20=fitted["loss_trace"][20],
            **metrics_for(features, disk, labels), stability=stability(fitted, disk),
            legacy_unit_curvature_evaluation=legacy_metrics,
            legacy_evaluation_defined=legacy_metrics is not None,
            evaluation_control="Same final coordinates; only sqrt(curvature) scaling differs. No refit.",
        ))
        key = str(curvature).replace(".", "_")
        arrays[f"current_c{key}"] = fitted["current"]
        arrays[f"loss_trace_c{key}"] = fitted["loss_trace"]
    payload = dict(
        input_id="moignard2015-development-reference", feature_sha256=FEATURE_SHA,
        labels_entered_fitting=False, curvatures=list(CURVATURES), epochs=20,
        promoted=promotion_allowed(development_input=True, joint_improvements=[True]),
        biological_validation=False, results=results,
        provenance=provenance(config, initial, elapsed_seconds=perf_counter() - start),
    )
    write_record(args.output, payload, arrays)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
