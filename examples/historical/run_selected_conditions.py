"""HISTORICAL ONLY: selected dataset-specific conditions, not current defaults.

Does not download data or read reference coordinates. See docs/reference-parity.md.
Requires .[torch,benchmark]. Labels are saved for plotting, never passed to fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import torch

from sc_poincare._research_policy import ensure_active_paths
from sc_poincare._fitting import fit_disk_embedding
from sc_poincare._proximity import estimate_proximity


def _historical_fitter():
    path = Path(__file__).with_name('experimental_geodesic_fit.py')
    spec = importlib.util.spec_from_file_location('_sc_poincare_historical_geodesic', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.fit_geodesic_regularized


# BEGIN HISTORICAL DATASET-SELECTED RECIPES (not general-purpose defaults)
RECIPES = {
    "ToggleSwitch": dict(k=15, sigma=1., pca=0, gamma=.5, lr=.02, stages=((800, 10.),)),
    "MyeloidProgenitors": dict(k=30, sigma=2., pca=0, gamma=.5, lr=.02, stages=((800, 10.),)),
    "krumsiek11_blobs": dict(k=30, sigma=1., pca=11, gamma=.25, lr=.02, stages=((800, 3.),)),
    "Olsson": dict(k=15, sigma="median", pca=20, gamma=.25, lr=.01, epochs=500),
    "Paul": dict(k=15, sigma="median", pca=20, gamma=.5, lr=.02, stages=((800, 10.), (150, 20.))),
    "Moignard2015": dict(k=20, sigma="median", pca=0, gamma=.5, lr=.02, stages=((700, 20.), (250, 15.))),
}
# END HISTORICAL DATASET-SELECTED RECIPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge-historical', action='store_true',
                        help='Explicitly opt into old selected recipes, not the current MU/RFA defaults')
    parser.add_argument("dataset", choices=list(RECIPES))
    parser.add_argument("csv", type=Path)
    parser.add_argument("--label-column", help="Defaults to the last column")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.acknowledge_historical:
        parser.error('Historical recipe execution requires --acknowledge-historical; use run_scanpy_demo.py for current workflows.')
    ensure_active_paths(args.csv, args.output, args.output.with_suffix('.json'))
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error("Output already exists; choose a new name to preserve prior evidence")
    frame = pd.read_csv(args.csv)
    label_column = args.label_column or frame.columns[-1]
    labels = frame[label_column].to_numpy(dtype=str)
    features = frame.drop(columns=label_column).to_numpy(dtype=np.float64)
    config = RECIPES[args.dataset]
    torch.set_num_threads(1)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if config['pca']:
            features = PCA(min(config['pca'], features.shape[0], features.shape[1]), svd_solver='full').fit_transform(features)
        if 'stages' in config:
            fit = _historical_fitter()(features, n_neighbors=config['k'], sigma=config['sigma'],
                gamma=config['gamma'], learning_rate=config['lr'], stages=config['stages'], random_state=args.seed)
            coordinates = fit.coordinates
            arrays = {f'loss_stage_{i}': stage['loss_trace'] for i, stage in enumerate(fit.stages)}
            metadata = {'parameters': fit.parameters,
                        'stages': [{k: v for k, v in stage.items() if k != 'loss_trace'} for stage in fit.stages]}
        else:
            proximity = estimate_proximity(features, config['k'], config['sigma'])
            fit = fit_disk_embedding(proximity.accessibility, connectivities=proximity.weights,
                temperature=config['gamma'], learning_rate=config['lr'], n_epochs=config['epochs'],
                patience=config['epochs'], random_state=args.seed, batch_size=256)
            coordinates = fit.coordinates
            arrays = {'loss_stage_0': np.asarray(fit.loss_trace)}
            metadata = {'parameters': dict(objective='RFA_KL', optimizer='rsgd',
                        sigma=proximity.kernel_width, **{k:v for k,v in config.items() if k != 'sigma'}),
                        'best_loss': fit.best_loss, 'stop_reason': fit.stop_reason,
                        'initialization': fit.initialization, 'initialization_reason': fit.initialization_reason}
    metadata.update(dataset=args.dataset, seed=args.seed, seconds=time.perf_counter()-started,
                    n_cells=len(features), n_features=features.shape[1],
                    input_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                    note='Raw learned coordinates. No root move, reference alignment, jitter, repel or fan.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, coordinates=coordinates, labels=labels, **arrays)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
