"""Explicit-input experimental CLI; outputs a new directory, never overwrites.

Example (CSV header, label in last column, explicit full-SVD PCA20)::

    python examples/run_graph_stress.py --input cells.csv --csv-label-column -1 \
        --pca 20 --local-bandwidth local_radius --output new-run

The checkpoint contains both optimizer state and matching current coordinates.
Resume into another new directory with the same input/options and --resume
old-run/checkpoint.pt --epochs TOTAL. No labels or dataset names select a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys


def load_features(path, *, csv_label_column=None, pca=None):
    """Remove the explicitly selected CSV label BEFORE numeric conversion/PCA."""
    import numpy as np
    path = Path(path)
    if path.suffix.lower() == ".npy":
        if csv_label_column is not None:
            raise ValueError("--csv-label-column applies only to CSV input.")
        features = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".csv":
        import pandas as pd
        frame = pd.read_csv(path)
        if csv_label_column is not None:
            if not -frame.shape[1] <= csv_label_column < frame.shape[1]:
                raise ValueError("CSV label column is out of range.")
            keep = [i for i in range(frame.shape[1]) if i != csv_label_column % frame.shape[1]]
            frame = frame.iloc[:, keep]
        features = frame.to_numpy(dtype=float)
    else:
        raise ValueError("Input must be a .csv (with header) or numeric .npy file.")
    if np.iscomplexobj(features) or features.dtype.kind not in "biuf":
        raise ValueError("Input must contain real numeric features.")
    if features.ndim != 2 or not features.shape[1] or not np.isfinite(features).all():
        raise ValueError("Input must be a finite numeric matrix with features.")
    if pca is not None:
        if pca < 1 or pca > min(features.shape):
            raise ValueError("PCA dimensions must be between 1 and min(input.shape).")
        from sklearn.decomposition import PCA
        features = PCA(pca, svd_solver="full").fit_transform(features)
    return np.ascontiguousarray(features, dtype=np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    parser.add_argument("--csv-label-column", type=int)
    parser.add_argument("--pca", type=int)
    parser.add_argument("--local-bandwidth", choices=("global", "local_radius"), default="global")
    parser.add_argument("--geometry", choices=("poincare", "euclidean"), default="poincare")
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=1500, help="Total target epochs, including on resume")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-dense-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--resume", type=Path, help="Previously generated checkpoint.pt (weights_only load)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory must not exist; all existing artifacts are preserved.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress, fit_graph_stress
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        features = load_features(args.input, csv_label_column=args.csv_label_column, pca=args.pca)
        setup = prepare_graph_stress(features, config=GraphStressConfig(
            n_neighbors=args.neighbors, local_bandwidth=args.local_bandwidth),
            max_dense_bytes=args.max_dense_bytes)
        resume = None
        if args.resume is not None:
            saved = torch.load(args.resume, map_location="cpu", weights_only=True)
            resume = saved["state"]
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                        metadata=setup.metadata, python=sys.version, numpy=np.__version__,
                        torch=str(torch.__version__), cli_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        np.save(args.output / "features.npy", features, allow_pickle=False)
        def checkpoint(progress):
            temporary = args.output / "checkpoint.pt.tmp"
            torch.save(dict(state=progress.state, current=torch.tensor(progress.current),
                            coordinates=torch.tensor(progress.best)), temporary)
            os.replace(temporary, args.output / "checkpoint.pt")
        result = fit_graph_stress(setup, epochs=args.epochs, random_state=args.seed,
                                  geometry=args.geometry, chunk_size=args.chunk_size,
                                  max_dense_bytes=args.max_dense_bytes, resume=resume,
                                  callback=checkpoint, checkpoint_every=args.checkpoint_every)
        np.savez(args.output / "result.npz", coordinates=result.coordinates, current=result.current,
                 initial=result.initial, initial_tangent=result.initial_tangent, loss_trace=result.loss_trace)
        manifest.update(n_epochs=result.n_epochs, best_epoch=result.best_epoch,
                        best_loss=result.best_loss, current_loss=result.current_loss)
        temporary = args.output / "run.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, args.output / "run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
