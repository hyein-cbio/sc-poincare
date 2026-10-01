"""Disk-aware scatter plots that sit next to Scanpy's embedding views."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_poincare._anndata import DEFAULT_BASIS


def _import_pyplot():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    return plt, Circle


def _as_color_values(adata, color: str | np.ndarray | None):
    if color is None:
        return None, None, False
    if isinstance(color, str):
        if color in adata.obs:
            values = adata.obs[color]
            categorical = getattr(values, "dtype", None) is not None and (
                getattr(values.dtype, "name", "") == "category" or values.dtype == object
            )
            if hasattr(values, "cat"):
                return values.to_numpy(), color, True
            numeric = np.asarray(values)
            if numeric.dtype.kind in {"U", "S", "O", "b"}:
                return numeric.astype(str), color, True
            return numeric, color, False
        if color in adata.var_names:
            index = adata.var_names.get_loc(color)
            matrix = adata.X[:, index]
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            return np.asarray(matrix).reshape(-1), color, False
        raise KeyError(f"{color!r} is neither an obs annotation nor a var name.")
    values = np.asarray(color)
    if values.shape[0] != adata.n_obs:
        raise ValueError("color array must have one entry per cell.")
    return values, None, values.dtype.kind in {"U", "S", "O"}


def draw_disk_embedding(
    adata,
    *,
    basis: str = DEFAULT_BASIS,
    color: str | np.ndarray | None = None,
    ax=None,
    size: float | None = None,
    cmap: str = "viridis",
    palette: dict[str, Any] | None = None,
    title: str | None = None,
    show: bool = True,
    save: str | None = None,
    frameon: bool = True,
):
    """Scatter cells inside the unit disk, with the Poincaré boundary drawn.

    Coordinates are read from ``adata.obsm[basis]`` (default ``X_poincare``),
    so ``scanpy.pl.embedding(adata, basis='poincare')`` also works if you only
    need Scanpy's usual scatter styling.
    """
    plt, Circle = _import_pyplot()
    if basis not in adata.obsm:
        alt = basis if basis.startswith("X_") else f"X_{basis}"
        if alt in adata.obsm:
            basis = alt
        else:
            raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    points = np.asarray(adata.obsm[basis], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Poincaré plot expects at least two embedding coordinates.")

    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))

    rim = Circle((0.0, 0.0), 1.0, fill=False, linewidth=1.1, edgecolor="0.35", zorder=0)
    ax.add_patch(rim)
    ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
    ax.axvline(0.0, color="0.85", linewidth=0.6, zorder=0)

    n_obs = points.shape[0]
    point_size = 18.0 if size is None else float(size)
    if n_obs > 2000 and size is None:
        point_size = max(4.0, 18000.0 / n_obs)

    values, label, categorical = _as_color_values(adata, color)
    scatter = None
    if values is None:
        scatter = ax.scatter(points[:, 0], points[:, 1], s=point_size, c="#3b6ea5", linewidths=0, zorder=2)
    elif categorical:
        categories = np.unique(values.astype(str))
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        for i, cat in enumerate(categories):
            mask = values.astype(str) == cat
            face = None if palette is None else palette.get(cat)
            if face is None:
                face = default_cycle[i % len(default_cycle)]
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=point_size,
                color=face,
                label=str(cat),
                linewidths=0,
                zorder=2,
            )
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, title=label)
    else:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=point_size,
            c=np.asarray(values, dtype=np.float64),
            cmap=cmap,
            linewidths=0,
            zorder=2,
        )
        cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        if label is not None:
            cbar.set_label(label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("PD1")
    ax.set_ylabel("PD2")
    if title is None:
        title = "Poincaré map"
    ax.set_title(title)
    if not frameon:
        ax.set_axis_off()
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if save:
        ax.figure.savefig(save, bbox_inches="tight", dpi=150)
    if show and created_fig:
        plt.show()
    return ax
