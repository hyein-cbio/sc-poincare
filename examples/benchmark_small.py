"""Benchmark user-provided small CSV datasets without visualization transforms.

This does not download or execute any reference implementation. See
``docs/benchmarking.md`` before interpreting the results as a comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sc_poincare._disk import pairwise_disk_length, place_point_at_origin
from sc_poincare._fitting import evaluate_alignment, fit_disk_embedding
from sc_poincare._proximity import estimate_proximity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Numeric features plus one label column; header required")
    parser.add_argument("--label-column", help="Defaults to the last column")
    parser.add_argument("--root-label", default="root", help="Recenter only if exactly one matching cell exists")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--sigma", type=lambda value: "median" if value == "median" else float(value), default=2.,
                        help="Numeric bandwidth or explicit median positive graph-edge distance")
    parser.add_argument("--gamma", type=float, default=2.)
    parser.add_argument("--objective", choices=["symmetric_kl", "conditional_symmetric_kl"],
                        default="symmetric_kl", help="Conditional mode is experimental and changes the objective")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--initialization", choices=["spectral", "random"], default="spectral")
    parser.add_argument("--batch-size", type=int, default=256, help="Exact row chunk size")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--terminal-labels", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True, help="JSON report, preferably under ignored data/")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.set_num_threads(args.threads)
    frame = pd.read_csv(args.csv)
    label_column = args.label_column or frame.columns[-1]
    labels = frame[label_column].to_numpy(dtype=str)
    features = frame.drop(columns=label_column).to_numpy(dtype=np.float64)
    if features.shape[0] < 3:
        parser.error("At least three observations are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with threadpool_limits(limits=args.threads):
        started = time.perf_counter()
        proximity = estimate_proximity(features, args.n_neighbors, args.sigma)
        preparation_seconds = time.perf_counter() - started
        for seed in args.seeds:
            started = time.perf_counter()
            report = fit_disk_embedding(
                proximity.accessibility,
                connectivities=proximity.weights if args.initialization == "spectral" else None,
                temperature=args.gamma, n_epochs=args.epochs, patience=args.epochs,
                random_state=seed, batch_size=args.batch_size, objective=args.objective,
            )
            fitting_seconds = time.perf_counter() - started
            loss = evaluate_alignment(report.coordinates, proximity.accessibility, args.gamma,
                                      objective=args.objective)
            coordinates = report.coordinates
            root = np.flatnonzero(labels == args.root_label)
            if len(root) == 1:
                coordinates = place_point_at_origin(coordinates, int(root[0]))
            distance = pairwise_disk_length(coordinates)
            np.fill_diagonal(distance, np.inf)
            k = min(15, len(coordinates) - 1)
            ranked = np.argsort(distance, axis=1)
            nearest = ranked[:, :k]
            graph_recall = []
            degree_matched_recall = []
            for i, row in enumerate(nearest):
                neighbors = proximity.weights.indices[proximity.weights.indptr[i]:proximity.weights.indptr[i + 1]]
                # Fixed-15 recall has an upper bound <1 for degree>15. Retain
                # that diagnostic and also report degree-matched retrieval.
                graph_recall.append(len(np.intersect1d(row, neighbors)) / len(neighbors))
                degree_matched_recall.append(len(np.intersect1d(ranked[i, :len(neighbors)], neighbors)) / len(neighbors))
            terminal_metrics = {}
            for label in args.terminal_labels:
                indices = np.flatnonzero(labels == label)
                if len(indices) < 2:
                    raise ValueError(f"Terminal label {label!r} needs at least two cells")
                block = distance[np.ix_(indices, indices)]
                terminal_metrics[label] = {
                    "within_hyperbolic_median": float(np.median(block[np.triu_indices(len(indices), 1)])),
                    "same_label_neighbor_fraction": float((labels[nearest[indices]] == label).mean()),
                }
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != "darwin":
                rss *= 1024
            record = {
                "seed": seed, "epochs_run": report.n_epochs,
                "stop_reason": report.stop_reason,
                "initialization_actual": report.initialization,
                "initialization_reason": report.initialization_reason,
                "preparation_seconds": preparation_seconds, "fitting_seconds": fitting_seconds,
                "global_loss_at_returned_coordinates": loss, "best_reported_loss": report.best_loss,
                "graph_neighbor_recall_at_15": float(np.mean(graph_recall)),
                "graph_neighbor_recall_at_node_degree": float(np.mean(degree_matched_recall)),
                "terminals": terminal_metrics,
                "process_peak_rss_bytes": rss,
            }
            records.append(record)
            np.savez(args.output.with_name(f"{args.output.stem}_seed{seed}.npz"),
                     coordinates=coordinates, raw_coordinates=report.coordinates,
                     labels=labels, loss_trace=report.loss_trace)
            print(json.dumps(record), flush=True)
    result = {
        "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "n_cells": len(features), "n_features": features.shape[1],
        "configuration": {"objective": args.objective, "probability_computation": "log_softmax",
                          "zero_target_policy": "machine_tiny_logs_only", "n_neighbors": args.n_neighbors, "sigma": proximity.kernel_width, "sigma_requested": args.sigma, "gamma": args.gamma,
                          "epochs": args.epochs, "initialization": args.initialization,
                          "batch_size": args.batch_size, "threads": args.threads,
                          "max_grad_norm": 2.5, "gradient_clipping": "riemannian_norm",
                          "affinity_kernel": "gaussian_all_edges"},
        "proximity_diagnostics": proximity.diagnostics,
        "records": records,
        "rss_note": "Whole-process high-water mark including imports, evaluation and previous seeds; not per-fit peak.",
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
