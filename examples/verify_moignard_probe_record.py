"""Check saved traces and recompute objectives/metrics from an NPZ, without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sc_poincare import controlled_probe, experimental
from sc_poincare._probe_record import ROOT, digest, load_moignard, metrics_for, stability


def verify(record, features, labels, setup):
    report = json.loads(record.read_text())
    sidecar = record.with_name(report['arrays']['file'])
    if hashlib.sha256(sidecar.read_bytes()).hexdigest() != report['arrays']['sha256']:
        raise ValueError(f"NPZ SHA256 mismatch: {sidecar}")
    for relative, expected in report['provenance']['source_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source SHA256 mismatch: {relative}")
    target = torch.tensor(np.array(setup.target), dtype=torch.float64)
    teacher = torch.tensor(np.array(setup.teacher), dtype=torch.float64)
    checks = []
    with np.load(sidecar, allow_pickle=False) as arrays:
        if digest(arrays['initial']) != report['provenance']['initial_sha256']:
            raise ValueError("Initial-coordinate SHA256 mismatch")
        if not np.array_equal(arrays['initial'], controlled_probe.classical_jitter(setup, 0)):
            raise ValueError("Initial-coordinate mismatch with prepared setup")
        if 'results' in report:
            rows = [(row, 'current_c' + str(row['curvature']).replace('.', '_'),
                     'loss_trace_c' + str(row['curvature']).replace('.', '_'), row['curvature'])
                    for row in report['results']]
        else:
            rows = [(report, 'current', 'loss_trace', None)]
            if 'legacy_gradient_only_ablation' in report:
                rows.append((report['legacy_gradient_only_ablation'], 'legacy_gradient_current',
                             'legacy_gradient_loss_trace', None))
        for row, key, trace_key, curvature in rows:
            current = arrays[key]
            trace = arrays[trace_key]
            if (trace.ndim != 1 or not trace.size or not np.isfinite(trace).all()
                    or not np.array_equal(trace, row['stability']['loss_trace'])):
                raise ValueError(f"Loss trace mismatch: {trace_key}")
            if trace[0] != row['loss_at_0'] or trace[-1] != row['loss_at_20']:
                raise ValueError(f"Loss trace endpoint mismatch: {trace_key}")
            if curvature is None:
                disk = controlled_probe.disk_from_lorentz(current)
                loss = controlled_probe._probe_objective(torch.tensor(current), target, teacher,
                                                        setup.config, 256, lorentz=True)
                minkowski = -current[:, 0] ** 2 + (current[:, 1:] ** 2).sum(1)
                fitted = dict(current=current, loss_trace=trace, minkowski_diag=minkowski)
            else:
                disk = np.sqrt(curvature) * current
                tangent = (experimental._tangent_from_disk(current) if curvature == 1.
                           else controlled_probe._tangent_from_disk_curvature(current, curvature))
                u = torch.tensor(tangent, dtype=torch.float64)
                if curvature == 1.:
                    loss, _, _ = experimental._objective(u, target, teacher, setup.config, 'poincare', 256, backward=False)
                else:
                    loss, _, _ = controlled_probe._objective_curvature(u, target, teacher, setup.config,
                                                                      256, curvature, backward=False)
                fitted = dict(current=current, loss_trace=trace)
                if row['legacy_unit_curvature_evaluation'] is not None:
                    if metrics_for(features, current, labels) != row['legacy_unit_curvature_evaluation']:
                        raise ValueError(f"Legacy population metrics mismatch: {key}")
            if not np.isclose(loss, row['loss_at_20'], rtol=0, atol=1e-12):
                raise ValueError(f"Final loss mismatch: {key}")
            actual = metrics_for(features, disk, labels)
            if actual != {name: row[name] for name in actual}:
                raise ValueError(f"Population metrics mismatch: {key}")
            if stability(fitted, disk) != row['stability']:
                raise ValueError(f"Stability mismatch: {key}")
            checks.append(dict(condition=key, final_loss_absolute_error=abs(loss - row['loss_at_20']),
                               loss_trace_exact=True, all_population_metrics_exact=True, stability_exact=True))
    return dict(record=record.name, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('records', type=Path, nargs='+')
    args = parser.parse_args()
    features, labels = load_moignard(args.source)
    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig(), max_dense_bytes=512 * 1024**2)
    print(json.dumps([verify(record, features, labels, setup) for record in args.records], indent=2))


if __name__ == '__main__':
    main()
