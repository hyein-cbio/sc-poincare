#!/usr/bin/env python
"""Standalone Scanpy demo for sc_poincare Poincaré maps.

Uses Scanpy's built-in datasets only (no extra downloads beyond what
``scanpy.datasets`` already caches). Default is ``krumsiek11``: a small
simulated myeloid differentiation tree, which is the geometry this method
is meant for.

    PYTHONPATH=src python examples/run_scanpy_demo.py

    PYTHONPATH=src python examples/run_scanpy_demo.py --dataset paul15
    PYTHONPATH=src python examples/run_scanpy_demo.py --dataset pbmc68k_reduced
    PYTHONPATH=src python examples/run_scanpy_demo.py --dataset krumsiek11 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

import sc_poincare as dr


def load_dataset(name: str):
    if name == "krumsiek11":
        adata = sc.datasets.krumsiek11()
        adata.obs_names_make_unique()
        color = "cell_type"
        # Simulated 11-gene myeloid cascade: no PCA needed.
        return adata, color, dict(use_rep="X", n_neighbors=15, sigma=2.0, gamma=2.0)

    if name == "toggleswitch":
        adata = sc.datasets.toggleswitch()
        color = None
        return adata, color, dict(use_rep="X", n_neighbors=15, sigma=1.0, gamma=1.0)

    if name == "pbmc68k_reduced":
        adata = sc.datasets.pbmc68k_reduced()
        color = "louvain"
        # Already processed; reuse the stored PCA.
        return adata, color, dict(use_rep="X_pca", n_pcs=20, n_neighbors=15, sigma=2.0, gamma=2.0)

    if name == "paul15":
        adata = sc.datasets.paul15()
        # Standard Scanpy hematopoietic recipe: normalize, HVG, log, scale.
        sc.pp.recipe_zheng17(adata)
        sc.tl.pca(adata, n_comps=50)
        # Cluster names are like "7MEP" / "3Ery"; the suffix is the lineage.
        suffix = adata.obs["paul15_clusters"].astype(str).str.replace(r"^\d+", "", regex=True)
        adata.obs["lineage"] = suffix.astype("category")
        return adata, "lineage", dict(use_rep="X_pca", n_pcs=20, n_neighbors=15, sigma=2.0, gamma=2.0)

    raise ValueError(
        f"Unknown dataset {name!r}. Choose krumsiek11, toggleswitch, pbmc68k_reduced, or paul15."
    )


def pick_root(adata, color: str | None) -> int:
    """Prefer an early / progenitor-like cell when labels exist."""
    if color == "cell_type":
        labels = adata.obs[color].astype(str).to_numpy()
        hits = np.flatnonzero(labels == "progenitor")
        if hits.size:
            return int(hits[0])
    if color == "lineage" and "lineage" in adata.obs:
        labels = adata.obs["lineage"].astype(str).to_numpy()
        for name in ("MEP", "GMP", "progenitor"):
            hits = np.flatnonzero(labels == name)
            if hits.size:
                return int(hits[0])
    # Scanpy's stored iroot is a fallback only: paul15 ships with a
    # megakaryocyte at that index, which is a leaf, not a progenitor.
    if "iroot" in adata.uns:
        return int(adata.uns["iroot"])
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="krumsiek11",
        choices=("krumsiek11", "toggleswitch", "pbmc68k_reduced", "paul15"),
    )
    parser.add_argument('--method', choices=('mu','rfa'), default='mu')
    parser.add_argument('--local-bandwidth', choices=('global','local_radius'))
    parser.add_argument("--n-epochs", type=int, help='Defaults to MU 1500 or legacy-demo RFA 200')
    parser.add_argument("--learning-rate", type=float, help='Omit to use the selected method default')
    parser.add_argument("--n-lineages", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=Path("examples/output"))
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = parser.parse_args(argv)
    if args.n_epochs is not None and args.n_epochs < 1:
        parser.error('Demo n-epochs must be positive.')
    if args.method == 'rfa' and args.local_bandwidth is not None:
        parser.error('local-bandwidth is MU-only.')

    adata, color, fit_kw = load_dataset(args.dataset)
    print(f"loaded {args.dataset}: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"obs columns: {list(adata.obs.columns)}")

    options = dict(method=args.method, random_state=0, verbose=True)
    if args.method == 'rfa':
        options.update(fit_kw)
        epochs = 200 if args.n_epochs is None else args.n_epochs
        options.update(n_epochs=epochs, burn_in_epochs=30, patience=epochs,
                       learning_rate=.01 if args.learning_rate is None else args.learning_rate)
    else:
        options.update({k:v for k,v in fit_kw.items() if k in ('use_rep','n_pcs','n_neighbors')})
        if args.n_epochs is not None:options['n_epochs']=args.n_epochs
        if args.learning_rate is not None:options['learning_rate']=args.learning_rate
        if args.local_bandwidth is not None:options['local_bandwidth']=args.local_bandwidth
    dr.tl.poincare_map(adata, **options)

    coords = adata.obsm["X_poincare"]
    radii = np.linalg.norm(coords, axis=1)
    print(f"embedding shape={coords.shape}  max radius={radii.max():.4f}  finite={np.isfinite(coords).all()}")
    print(f"loss epochs={len(adata.uns['poincare']['loss'])}  last={adata.uns['poincare']['loss'][-1]:.3f}")

    if color == "louvain":
        root = int(np.argmin(radii))
    else:
        root = pick_root(adata, color)
    print(f"root index={root}  obs_name={adata.obs_names[root]!r}")

    dr.tl.poincare_recenter(adata, root)
    dr.tl.poincare_pseudotime(adata, root=root)
    dr.tl.poincare_lineages(adata, n_lineages=args.n_lineages, root=root)

    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.outdir / args.dataset

    point_size = 8.0 if adata.n_obs > 1500 else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    dr.pl.poincare_map(adata, color=color, ax=axes[0], show=False, title=f"{args.dataset} / labels", size=point_size)
    dr.pl.poincare_map(adata, color="poincare_pseudotime", ax=axes[1], show=False, title="pseudotime", size=point_size)
    dr.pl.poincare_map(adata, color="poincare_lineage", ax=axes[2], show=False, title="angular lineages", size=point_size)
    fig.tight_layout()
    panel_path = prefix.with_name(prefix.name + "_poincare.png")
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    print(f"wrote {panel_path}")

    # Scanpy's generic embedding lookup: basis "poincare" → obsm["X_poincare"].
    sc.pl.embedding(
        adata,
        basis="poincare",
        color=["poincare_pseudotime", "poincare_lineage"] + ([color] if color else []),
        show=False,
        save=f"_{args.dataset}_scanpy.png",
    )
    # Scanpy writes into scanpy.settings.figdir (usually ./figures).
    print(f"scanpy also wrote figures/*_{args.dataset}_scanpy.png")

    umap_kw = {k: fit_kw[k] for k in ("use_rep", "n_pcs", "n_neighbors") if k in fit_kw}
    sc.pp.neighbors(adata, **umap_kw)
    sc.tl.umap(adata, random_state=0)
    fig_u, ax_u = plt.subplots(figsize=(5.6, 5.2))
    sc.pl.umap(adata, color=color, ax=ax_u, show=False, title=f"{args.dataset} / UMAP", frameon=True)
    fig_u.tight_layout()
    umap_path = prefix.with_name(prefix.name + "_umap.png")
    fig_u.savefig(umap_path, dpi=150, bbox_inches="tight")
    print(f"wrote {umap_path}")

    pt = adata.obs["poincare_pseudotime"].to_numpy()
    print(f"pseudotime range [{pt.min():.3f}, {pt.max():.3f}]")
    print(f"lineages: {adata.obs['poincare_lineage'].value_counts().to_dict()}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
