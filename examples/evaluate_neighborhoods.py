"""Evaluate or compare existing embeddings; never fit, tune or download data.

Input NPZ: features (fitted numeric representation without labels), coordinates
(raw N-by-2 disk points), labels, and optional unique string cell_ids. If a
raw_coordinates array exists it takes precedence over coordinates. A --baseline
NPZ requires matching IDs/order, features and labels on both sides. Legacy
before_coordinates supports an unchecked visual panel only, not metric deltas.

Example::
    python examples/evaluate_neighborhoods.py candidate.npz \
        --baseline baseline.npz --output data/evaluation
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np

from sc_poincare import evaluation as evaluation_module
from sc_poincare.evaluation import (
    compare_neighborhoods, evaluate_neighborhoods, plot_neighborhood_diagnostics,
)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_evaluation(path, args):
    fingerprint = _file_hash(path)
    with np.load(path, allow_pickle=False) as data:
        points = data['raw_coordinates'] if 'raw_coordinates' in data else data['coordinates']
        result = evaluate_neighborhoods(data['features'], points, data['labels'],
            ks=args.ks, block_size=args.block_size,
            cell_ids=data['cell_ids'] if 'cell_ids' in data else None,
            global_n_anchors=args.global_anchors, global_random_state=args.global_seed)
        before = data['before_coordinates'].copy() if 'before_coordinates' in data else None
    if _file_hash(path) != fingerprint:
        raise ValueError('Input file changed during evaluation.')
    return result, before, fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--baseline', type=Path, help='Matched baseline NPZ; both files require cell_ids')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--ks', type=int, nargs='+', default=[15, 30])
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--global-anchors', type=int, default=128)
    parser.add_argument('--global-seed', type=int, default=714)
    parser.add_argument('--title', default='Post-fit neighborhood diagnostics')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output directory must not exist; existing results are preserved.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result, before, fingerprint = _read_evaluation(args.input, args)
    baseline, comparison, baseline_hash = None, None, None
    if args.baseline is not None:
        if before is not None:
            raise ValueError('Use --baseline or embedded before_coordinates, not both.')
        baseline, _, baseline_hash = _read_evaluation(args.baseline, args)
        comparison = compare_neighborhoods(baseline, result)
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'scipy', 'scikit-learn', 'matplotlib', 'sc-poincare'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = 'not installed as a distribution'
    manifest = dict(input_sha256=fingerprint, baseline_sha256=baseline_hash,
        versions=versions, ks=result.summary['ks'], block_size=args.block_size,
        global_anchors=args.global_anchors, global_seed=args.global_seed,
        cell_order_verified=comparison is not None,
        unchecked_before_panel=before is not None,
        evaluator_sha256=_file_hash(Path(evaluation_module.__file__)),
        cli_sha256=_file_hash(Path(__file__)))
    # Validate/render before creating output files, including optional before panels.
    figures = []
    try:
        for k in result.summary['ks']:
            figures.append((k, plot_neighborhood_diagnostics(result, k=k,
                            before_coordinates=before, baseline=baseline, title=args.title)))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'summary.json').write_text(json.dumps(result.summary, indent=2, allow_nan=False) + '\n')
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if comparison is not None:
            (args.output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
        np.savez_compressed(args.output / 'cell_scores.npz',
            **{f'purity_k{k}': result.purity[k] for k in result.summary['ks']},
            **{f'input_recall_k{k}': result.input_recall[k] for k in result.summary['ks']})
        for k, figure in figures:
            figure.savefig(args.output / f'diagnostics_k{k}.png', dpi=150)
    finally:
        for _, figure in figures:
            plt.close(figure)
    return result.summary


if __name__ == '__main__':
    main()
