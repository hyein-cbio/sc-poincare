"""One isolated MU/RFA comparison job; called by the bounded supervisor.

Input NPZ contains features, labels, unique string cell_ids, and optionally
expression in the same row order. Labels are evaluation-only. RFA uses the
same mutual classical-scaling initializer as MU (not its normal spectral start).
No reference coordinates or dataset-name model branches are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from sc_poincare import experimental as mu, _fitting, _proximity, evaluation, visibility

from sc_poincare._research_policy import ensure_active_paths

METHODS = ('rfa', 'mu_fixed', 'mu_local', 'euclidean_fixed', 'euclidean_local')


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--resume', type=Path, help='Trusted bare MU Adam state; epochs is the total target')
    args = parser.parse_args(argv)
    ensure_active_paths(args.input, args.resume, args.output)
    if args.resume is not None and args.method == 'rfa':
        parser.error('RFA coordinate warm starts are new runs, not exact Adam resume.')
    if args.output.exists(): parser.error('Output directory must not exist.')
    if min(args.epochs, args.threads, args.chunk_size, args.checkpoint_every) < 1 or args.seed < 0:
        parser.error('Counts must be positive and seed nonnegative.')
    import torch
    args.output.mkdir(parents=True, exist_ok=False)
    metadata_path = args.output / 'run.json'
    started = time.monotonic()
    metadata = dict(status='running', method=args.method, epochs_requested=args.epochs, seed=args.seed,
        threads=args.threads, chunk_size=args.chunk_size, input_sha256=file_hash(args.input),
        worker_sha256=file_hash(__file__), source_sha256={Path(m.__file__).name:file_hash(m.__file__)
            for m in (mu, _fitting, _proximity, evaluation, visibility)},
        versions=dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__)),
        phase='load', epoch=0)
    atomic_json(metadata_path, metadata)
    old_threads = torch.get_num_threads()
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise TimeoutError('Terminated by experiment supervisor; last atomic checkpoint is retained.')
    signal.signal(signal.SIGTERM, terminated)
    try:
        torch.set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            with np.load(args.input, allow_pickle=False) as data:
                features, labels, ids = data['features'], data['labels'], data['cell_ids']
                expression = data['expression'] if 'expression' in data else None
            _proximity.check_dense_budget(len(features), 512 * 1024**2)
            geometry = 'euclidean' if args.method.startswith('euclidean') else 'poincare'
            # Validate labels/IDs before fitting, without using them for the model.
            evaluation.evaluate_neighborhoods(features, np.zeros((len(features), 2)), labels,
                cell_ids=ids, geometry=geometry)
            if expression is not None:
                expression = evaluation._matrix(expression, 'expression')
                if len(expression) != len(features): raise ValueError('Expression row count differs.')
            metadata.update(phase='preparation', n_cells=len(features), n_features=features.shape[1], geometry=geometry)
            atomic_json(metadata_path, metadata)
            preparation_start = time.monotonic()
            bandwidth = 'local_radius' if args.method.endswith('local') else 'global'
            config = mu.GraphStressConfig(local_bandwidth=bandwidth)
            prox = _proximity.estimate_proximity(features, 15, 'median') if args.method == 'rfa' else None
            setup = mu.prepare_graph_stress(features, config=config)
            initial = setup.base_initial + np.random.default_rng(args.seed).normal(scale=config.jitter, size=setup.base_initial.shape)
            metadata.update(parameters=asdict(config), setup=setup.metadata,
                initial_disk_sha256=mu._digest(initial), initial_tangent_sha256=mu._digest(mu._tangent_from_disk(initial)),
                preparation_seconds=time.monotonic()-preparation_start, phase='fitting')
            atomic_json(metadata_path, metadata)
            fit_start = time.monotonic()
            if args.method == 'rfa':
                metadata['parameters'] = dict(objective='symmetric_kl', sigma=prox.kernel_width,
                    gamma=2., learning_rate=.01, max_grad_norm=2.5, burn_in_epochs=5, patience=args.epochs,
                    initialization='shared_mu_classical_not_default_spectral')
                metadata['proximity_diagnostics'] = prox.diagnostics
                metadata['rfa_target_sha256'] = mu._digest(prox.accessibility)
                del setup; gc.collect()
                observed = dict(best_epoch=0, best_loss=None)
                def progress(p):
                    if observed['best_loss'] is None or p.loss < observed['best_loss']:
                        observed.update(best_epoch=p.epoch, best_loss=p.loss)
                    if p.epoch % args.checkpoint_every == 0 or p.epoch == args.epochs:
                        metadata.update(epoch=p.epoch, current_loss=p.loss, best_loss=p.best_loss)
                        temporary=args.output/'checkpoint.npz.tmp'
                        with temporary.open('wb') as handle:
                            np.savez_compressed(handle, current=p.coordinates, best=p.best_coordinates,
                                                epoch=p.epoch, coordinate_only_warm_start=True)
                        temporary.replace(args.output/'checkpoint.npz')
                        atomic_json(metadata_path, metadata)
                report = _fitting.fit_disk_embedding(prox.accessibility, initial_coordinates=initial,
                    n_epochs=args.epochs, patience=args.epochs, burn_in_epochs=5, temperature=2.,
                    learning_rate=.01, max_grad_norm=2.5, batch_size=args.chunk_size,
                    random_state=args.seed, progress_callback=progress)
                best, trace = report.coordinates, np.asarray(report.loss_trace)
                with np.load(args.output/'checkpoint.npz', allow_pickle=False) as state:
                    current=state['current'].copy()
                metadata.update(best_epoch=observed['best_epoch'], best_loss=report.best_loss,
                                n_epochs=report.n_epochs, stop_reason=report.stop_reason)
                del prox
            else:
                resume_state = None
                if args.resume is not None:
                    resume_state = torch.load(args.resume, map_location='cpu', weights_only=True)
                    metadata.update(resume_sha256=file_hash(args.resume), resume_from_epoch=resume_state.get('epoch'))
                def progress(p):
                    metadata.update(epoch=p.epoch, current_loss=p.loss, best_loss=p.best_loss,
                                    best_epoch=p.best_epoch)
                    temporary = args.output/'checkpoint.pt.tmp'
                    torch.save(p.state, temporary); temporary.replace(args.output/'checkpoint.pt')
                    atomic_json(metadata_path, metadata)
                report = mu.fit_graph_stress(setup, epochs=args.epochs, random_state=args.seed,
                    geometry=geometry, chunk_size=args.chunk_size, callback=progress,
                    checkpoint_every=args.checkpoint_every, resume=resume_state)
                np.testing.assert_array_equal(initial, report.initial)
                best, current, trace = report.coordinates, report.current, report.loss_trace
                metadata.update(best_epoch=report.best_epoch, best_loss=report.best_loss,
                    current_loss=report.current_loss, n_epochs=report.n_epochs, stop_reason='epoch_limit')
                del setup
            metadata.update(fitting_seconds=time.monotonic()-fit_start, phase='evaluation')
            atomic_json(metadata_path, metadata)
            temporary=args.output/'result.npz.tmp'
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, coordinates=best, current=current, initial_disk=initial, loss_trace=trace)
            temporary.replace(args.output/'result.npz')
            assessed = evaluation.evaluate_neighborhoods(features, best, labels, cell_ids=ids, geometry=geometry)
            metadata['evaluation'] = assessed.summary
            metadata['visibility'] = visibility.evaluate_visibility(assessed) if geometry=='poincare' else None
            metadata['expression'] = (evaluation.evaluate_expression_neighborhoods(expression,best,geometry=geometry)
                                      if expression is not None else None)
            metadata.update(status='completed', phase='done')
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt,TimeoutError)) else 'failed',
                        error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        torch.set_num_threads(old_threads)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata.update(elapsed_seconds=time.monotonic()-started,
                        process_peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024))
        atomic_json(metadata_path,metadata)
    print(json.dumps(dict(status=metadata['status'],method=args.method,seed=args.seed,
                         best_loss=metadata['best_loss'],seconds=metadata['elapsed_seconds'])),flush=True)
    return metadata


if __name__ == '__main__':
    main()
