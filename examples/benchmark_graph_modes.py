"""Bounded paired mutual/union experiment with shared sigma and initialization.

Inputs are NPZs with features (explicitly preprocessed, label-free), labels and
unique string cell_ids in matching row order. No dataset selection/preprocessing
is implicit. Example (12 fits for two inputs)::

    python examples/benchmark_graph_modes.py input1.npz input2.npz \
        --output data/graph_diagnostics_integration --wall-seconds 600

Requires Torch and benchmark extras, and Unix interval timers for a whole-run
wall limit. Does not choose a winner, promote defaults, or compare KL accuracy
across different targets. See docs/algorithm-experiment-01.md.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import signal
import time

import numpy as np
from threadpoolctl import threadpool_limits

from sc_poincare._fitting import _seed_from_connectivities, fit_disk_embedding
from sc_poincare._proximity import estimate_proximity
from sc_poincare._resources import check_dense_budget
from sc_poincare.evaluation import compare_neighborhoods, evaluate_neighborhoods, plot_neighborhood_diagnostics


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrites')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--neighbors', type=int, default=15)
    parser.add_argument('--ks', type=int, nargs='+', default=[15, 30])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--wall-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output directory must not exist.')
    if any(value < 1 for value in (args.epochs, args.neighbors, args.batch_size, args.threads)):
        parser.error('epochs, neighbors, batch-size and threads must be positive.')
    if not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error('wall-seconds must be finite and positive.')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('seeds must be distinct nonnegative integers.')
    if len({path.stem for path in args.inputs}) != len(args.inputs):
        parser.error('Input filenames must have distinct stems.')
    if not hasattr(signal, 'setitimer') or signal.getitimer(signal.ITIMER_REAL)[0]:
        parser.error('This bounded runner requires an unused Unix interval timer in the main thread.')
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sc_poincare import _disk, _fitting, _proximity, evaluation

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(status='running', planned_fits=len(args.inputs) * len(args.seeds) * 2,
        configuration=dict(epochs=args.epochs, seeds=args.seeds, neighbors=args.neighbors,
            ks=args.ks, threads=args.threads, batch_size=args.batch_size, wall_seconds=args.wall_seconds,
            gamma=2., learning_rate=.01, max_grad_norm=2.5, burn_in_epochs=5, patience=args.epochs,
            sigma='mutual-positive-edge-median-once', initialization='shared-mutual-spectral',
            objective='symmetric_kl', max_dense_bytes=512 * 1024**2,
            global_n_anchors=128, global_random_state=714,
            representative_seed=args.seeds[0]),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                      scipy=version('scipy'), sklearn=version('scikit-learn')),
        source_sha256={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (_disk, _fitting, _proximity, evaluation)},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={}, runs=[], comparisons=[], active=None)
    path = args.output / 'summary.json'
    atomic_json(path, status)
    old_handler = signal.getsignal(signal.SIGALRM)
    old_threads = torch.get_num_threads()
    def timeout(signum, frame):
        raise TimeoutError(f'Whole experiment exceeded {args.wall_seconds:g} seconds.')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    try:
        torch.set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            for input_path in args.inputs:
                name = input_path.stem
                status['active'] = dict(dataset=name, stage='preparation')
                atomic_json(path, status)
                file_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
                with np.load(input_path, allow_pickle=False) as data:
                    features, labels, ids = data['features'], data['labels'], data['cell_ids']
                check_dense_budget(len(features), 512 * 1024**2)
                # Validate evaluation inputs before any fitting. No labels enter
                # graph construction, initialization or optimization.
                evaluate_neighborhoods(features, np.zeros((len(features), 2)), labels,
                                       cell_ids=ids, ks=args.ks)
                if hashlib.sha256(input_path.read_bytes()).hexdigest() != file_hash:
                    raise ValueError('Input changed while loading.')
                graphs = {'mutual': estimate_proximity(features, args.neighbors, 'median')}
                sigma = graphs['mutual'].kernel_width
                graphs['union'] = estimate_proximity(features, args.neighbors, sigma, graph_mode='union')
                folder = args.output / name
                folder.mkdir()
                np.savez_compressed(folder / 'input.npz', features=features, labels=labels, cell_ids=ids)
                status['inputs'][name] = dict(input_sha256=file_hash, feature_sha256=digest(features),
                    n_cells=len(features), n_features=features.shape[1], sigma=sigma,
                    graphs={mode: dict(diagnostics=prox.diagnostics, n_edges=prox.weights.nnz // 2,
                        total_affinity=float(prox.weights.sum()), target_sha256=digest(prox.accessibility))
                        for mode, prox in graphs.items()})
                for seed in args.seeds:
                    init_status = {}
                    initial = _seed_from_connectivities(graphs['mutual'].weights, 2,
                                np.random.default_rng(seed), status=init_status)
                    initial_hash = digest(initial)
                    reports, evaluations, run_records = {}, {}, {}
                    for mode in ('mutual', 'union'):
                        status['active'] = dict(dataset=name, seed=seed, graph_mode=mode, epoch=0)
                        atomic_json(path, status)
                        observed = dict(epoch=0, best_epoch=0, best_loss=None)
                        def progress(p):
                            observed['epoch'] = p.epoch
                            if observed['best_loss'] is None or p.loss < observed['best_loss']:
                                observed.update(best_loss=p.loss, best_epoch=p.epoch)
                            observed['current_loss'] = p.loss
                            status['active']['epoch'] = p.epoch
                            if p.epoch == 0:
                                np.testing.assert_array_equal(p.coordinates, initial)
                        fit_start = time.monotonic()
                        report = fit_disk_embedding(graphs[mode].accessibility, initial_coordinates=initial,
                            random_state=seed, n_epochs=args.epochs, patience=args.epochs, burn_in_epochs=5,
                            temperature=2., learning_rate=.01, max_grad_norm=2.5, batch_size=args.batch_size,
                            objective='symmetric_kl', progress_callback=progress)
                        assert digest(initial) == initial_hash
                        record = dict(dataset=name, seed=seed, graph_mode=mode, status='completed',
                            sigma=sigma, initial_sha256=initial_hash, original_initializer=init_status,
                            fit_initializer=report.initialization, n_epochs=report.n_epochs,
                            best_epoch=observed['best_epoch'], best_loss=report.best_loss,
                            current_loss=observed['current_loss'], stop_reason=report.stop_reason,
                            fitting_seconds=time.monotonic() - fit_start)
                        reports[mode], run_records[mode] = report, record
                        status['runs'].append(record)
                        atomic_json(path, status)
                        evaluations[mode] = evaluate_neighborhoods(features, report.coordinates, labels,
                                                                  cell_ids=ids, ks=args.ks)
                    np.savez_compressed(folder / f'seed{seed}.npz', initial=initial,
                        mutual=reports['mutual'].coordinates, union=reports['union'].coordinates,
                        mutual_loss=reports['mutual'].loss_trace, union_loss=reports['union'].loss_trace)
                    comparison = compare_neighborhoods(evaluations['mutual'], evaluations['union'])
                    entry = dict(dataset=name, seed=seed, comparison=comparison,
                                 mutual=evaluations['mutual'].summary, union=evaluations['union'].summary)
                    status['comparisons'].append(entry)
                    atomic_json(path, status)
                    if seed == args.seeds[0]:
                        for k in sorted(set(args.ks)):
                            figure = plot_neighborhood_diagnostics(evaluations['union'], baseline=evaluations['mutual'],
                                k=k, title=f'{name}: mutual (before) vs union (after), seed {seed}')
                            try:
                                figure.savefig(folder / f'comparison_k{k}.png', dpi=150)
                            finally:
                                plt.close(figure)
                    print(json.dumps(dict(dataset=name, seed=seed,
                        primary_delta=comparison['metrics'][str(sorted(set(args.ks))[0])],
                        global_delta=comparison['global_feature_rank_delta'])), flush=True)
        signal.setitimer(signal.ITIMER_REAL, 0)
        status.update(status='completed', active=None)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        status.update(status='timeout' if isinstance(exc, TimeoutError) else 'failed',
                      error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        torch.set_num_threads(old_threads)
        status['elapsed_seconds'] = time.monotonic() - started
        atomic_json(path, status)
    return status


if __name__ == '__main__':
    main()
