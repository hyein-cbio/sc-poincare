"""AnnData-facing wrappers that follow Scanpy storage conventions."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

from sc_poincare._disk import (
    circular_gap,
    fan_dendrogram_angles,
    jitter_angles,
    length_from_origin,
    repel_points,
    place_point_at_origin,
    polar_angles,
)
from sc_poincare._fitting import fit_disk_embedding
from sc_poincare._proximity import estimate_proximity

DEFAULT_BASIS = "X_poincare"
DEFAULT_UNS = "poincare"
DEFAULT_PSEUDOTIME = "poincare_pseudotime"
DEFAULT_LINEAGE = "poincare_lineage"


def _as_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _choose_features(adata, use_rep: str | None, n_pcs: int | None) -> tuple[np.ndarray, str]:
    if use_rep == "X":
        return _as_dense(adata.X), "X"
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"Representation {use_rep!r} is not present in adata.obsm.")
        features = np.asarray(adata.obsm[use_rep], dtype=np.float64)
        if n_pcs is not None:
            features = features[:, :n_pcs]
        return features, use_rep

    n_vars = int(adata.n_vars)
    if n_vars <= 100:
        return _as_dense(adata.X), "X"
    if "X_pca" in adata.obsm:
        pcs = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        width = 50 if n_pcs is None else n_pcs
        return pcs[:, :width], "X_pca"
    raise ValueError(
        "Input has more than 100 features and no PCA was found. "
        "Run scanpy.pp.pca first, or pass use_rep explicitly."
    )


def _resolve_cell_index(adata, key: int | str | None) -> int | None:
    if key is None:
        return None
    if isinstance(key, (int, np.integer)):
        index = int(key)
        if index < 0:
            index += adata.n_obs
        if not 0 <= index < adata.n_obs:
            raise IndexError(f"Cell index {key} is out of range for {adata.n_obs} observations.")
        return index
    if isinstance(key, str):
        if key not in adata.obs_names:
            raise KeyError(f"Cell name {key!r} is not in adata.obs_names.")
        loc = adata.obs_names.get_loc(key)
        if isinstance(loc, (slice, np.ndarray)):
            raise ValueError(f"Cell name {key!r} is not unique in adata.obs_names.")
        return int(loc)
    raise TypeError("root must be an integer position, an obs_names label, or None.")


def _obsm_and_uns_keys(key_added: str | None) -> tuple[str, str]:
    if key_added is None:
        return DEFAULT_BASIS, DEFAULT_UNS
    if key_added.startswith("X_"):
        return key_added, key_added[2:]
    return f"X_{key_added}", key_added


def layout_cells(
    adata,
    *,
    n_neighbors: int = 15,
    sigma: float = 2.0,
    gamma: float = 2.0,
    n_components: int = 2,
    use_rep: str | None = None,
    n_pcs: int | None = None,
    metric: str = "euclidean",
    n_epochs: int = 250,
    learning_rate: float = 0.01,
    batch_size: int | None = None,
    burn_in_epochs: int = 30,
    min_delta: float = 1e-4,
    patience: int = 40,
    root: int | str | None = None,
    random_state: int | np.random.Generator | None = 0,
    key_added: str | None = None,
    copy: bool = False,
    verbose: bool = False,
):
    """Embed cells on the Poincaré disk and store the result on ``adata``.

    Parameters
    ----------
    adata
        Annotated cell-by-feature matrix.
    n_neighbors
        Size of the mutual neighborhood used for the local graph (paper: ``k``).
    sigma
        Gaussian kernel width on neighborhood edge lengths (paper: ``σ``).
    gamma
        Softmax temperature on hyperbolic distances (paper: ``γ``).
    n_components
        Embedding dimension. Two is the usual visualization setting.
    use_rep
        Feature source. ``'X'`` or any ``adata.obsm`` key. If omitted, ``X`` is
        used when there are at most 100 features, otherwise ``X_pca``.
    n_pcs
        Optional column cap when the representation is a PCA (or similar) matrix.
    metric
        Local distance used to build the neighborhood (``euclidean``, ``cosine``, ...).
    n_epochs, learning_rate, batch_size, burn_in_epochs, min_delta, patience
        Riemannian fitting controls. Training stops early when the loss stalls.
    root
        Optional cell (integer position or ``obs_names`` label) placed at the origin
        after fitting, via a distance-preserving disk translation.
    random_state
        Seed for the initial cloud and any mini-batch shuffling.
    key_added
        Storage prefix. Default writes ``adata.obsm['X_poincare']`` and
        ``adata.uns['poincare']``.
    copy
        If True, operate on a copy and return it.
    verbose
        Print occasional loss values.

    Returns
    -------
    AnnData or None
        The annotated object when ``copy=True``, otherwise ``None``.

    Notes
    -----
    Writes
      * ``adata.obsm['X_poincare']`` — disk coordinates,
      * ``adata.obsp['poincare_connectivities']`` — local Gaussian affinities,
      * ``adata.uns['poincare']`` — hyperparameters and the loss trace.
    """
    if copy:
        adata = adata.copy()

    features, resolved_rep = _choose_features(adata, use_rep, n_pcs)
    if features.shape[0] != adata.n_obs:
        raise ValueError("Feature matrix must have one row per observation.")

    proximity = estimate_proximity(
        features,
        n_neighbors=n_neighbors,
        kernel_width=sigma,
        metric=metric,
    )
    report = fit_disk_embedding(
        proximity.accessibility,
        n_components=n_components,
        temperature=gamma,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        burn_in_epochs=burn_in_epochs,
        min_delta=min_delta,
        patience=patience,
        random_state=random_state,
        verbose=verbose,
    )
    coordinates = report.coordinates
    root_index = _resolve_cell_index(adata, root)
    if root_index is not None:
        coordinates = place_point_at_origin(coordinates, root_index)

    obsm_key, uns_key = _obsm_and_uns_keys(key_added)
    adata.obsm[obsm_key] = coordinates
    connectivities_key = f"{uns_key}_connectivities"
    adata.obsp[connectivities_key] = proximity.weights
    adata.uns[uns_key] = {
        "params": {
            "n_neighbors": int(n_neighbors),
            "sigma": float(sigma),
            "gamma": float(gamma),
            "n_components": int(n_components),
            "use_rep": resolved_rep,
            "n_pcs": None if n_pcs is None else int(n_pcs),
            "metric": metric,
            "n_epochs_requested": int(n_epochs),
            "n_epochs_run": int(report.n_epochs),
            "learning_rate": float(learning_rate),
            "batch_size": batch_size,
            "burn_in_epochs": int(burn_in_epochs),
            "root": None if root is None else (int(root) if isinstance(root, (int, np.integer)) else str(root)),
            "random_state": None if isinstance(random_state, np.random.Generator) else random_state,
            "backend": report.backend,
        },
        "connectivities_key": connectivities_key,
        "loss": np.asarray(report.loss_trace, dtype=np.float64),
    }
    return adata if copy else None


def move_root_to_origin(
    adata,
    root: int | str,
    *,
    basis: str = DEFAULT_BASIS,
    key_added: str | None = None,
    copy: bool = False,
):
    """Apply the Poincaré isometry that sends ``root`` to the disk origin."""
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    index = _resolve_cell_index(adata, root)
    if index is None:
        raise ValueError("root is required.")
    shifted = place_point_at_origin(np.asarray(adata.obsm[basis], dtype=np.float64), index)
    dest = basis if key_added is None else key_added
    if dest.startswith("X_") or dest in adata.obsm or key_added is None:
        adata.obsm[dest] = shifted
    else:
        adata.obsm[f"X_{dest}"] = shifted
    return adata if copy else None


def score_pseudotime(
    adata,
    root: int | str | None = None,
    *,
    basis: str = DEFAULT_BASIS,
    key_added: str = DEFAULT_PSEUDOTIME,
    copy: bool = False,
):
    """Store hyperbolic distance from a root cell as Poincaré pseudotime.

    If ``root`` is given, the embedding is first recentered so that cell sits at
    the origin (in-place on ``basis``). If omitted, distance from the current
    origin is used — which is already the right reading after a recenter.
    """
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    coordinates = np.asarray(adata.obsm[basis], dtype=np.float64)
    index = _resolve_cell_index(adata, root)
    if index is not None:
        coordinates = place_point_at_origin(coordinates, index)
        adata.obsm[basis] = coordinates
    adata.obs[key_added] = length_from_origin(coordinates)
    return adata if copy else None


def infer_lineages(
    adata,
    n_lineages: int,
    *,
    root: int | str | None = None,
    basis: str = DEFAULT_BASIS,
    key_added: str = DEFAULT_LINEAGE,
    copy: bool = False,
):
    """Slice the recentered disk into angular lineages.

    After (optional) root recentering, cells are clustered by circular distance
    of their polar angle — the naive lineage rule described in the paper.
    """
    if copy:
        adata = adata.copy()
    if n_lineages < 1:
        raise ValueError("n_lineages must be at least 1.")
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    coordinates = np.asarray(adata.obsm[basis], dtype=np.float64)
    index = _resolve_cell_index(adata, root)
    if index is not None:
        coordinates = place_point_at_origin(coordinates, index)
        adata.obsm[basis] = coordinates
    if n_lineages == 1:
        labels = np.zeros(adata.n_obs, dtype=np.int64)
    else:
        theta = polar_angles(coordinates)
        angular = circular_gap(theta)
        model = AgglomerativeClustering(
            n_clusters=n_lineages,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(angular).astype(np.int64)
    adata.obs[key_added] = labels.astype(str)
    adata.obs[key_added] = adata.obs[key_added].astype("category")
    return adata if copy else None


def jitter_embedding(
    adata,
    *,
    jitter: float = 0.0,
    basis: str = DEFAULT_BASIS,
    key_added: str = "X_poincare_jitter",
    random_state: int | np.random.Generator | None = 0,
    copy: bool = False,
):
    """Write a display layout with uniform angular jitter.

    ``jitter=0`` copies ``adata.obsm[basis]`` exactly. Positive values are
    the half-width, in radians, of a uniform angle shift. Radius is unchanged.
    """
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    adata.obsm[key_added] = jitter_angles(
        np.asarray(adata.obsm[basis]),
        jitter=jitter,
        random_state=random_state,
    )
    return adata if copy else None


def repel_embedding(
    adata,
    *,
    repel: float = 0.0,
    basis: str = DEFAULT_BASIS,
    key_added: str = "X_poincare_repel",
    copy: bool = False,
):
    """Write a display layout that separates overlaps along each ray.

    ``repel=0`` copies ``adata.obsm[basis]`` exactly. Positive values change
    radius only. Polar angle is left unchanged.
    """
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    adata.obsm[key_added] = repel_points(np.asarray(adata.obsm[basis]), repel=repel)
    return adata if copy else None


def fan_branches(
    adata,
    groupby: str,
    *,
    spread: float = 0.0,
    basis: str = DEFAULT_BASIS,
    key_added: str = "X_poincare_fan",
    copy: bool = False,
):
    """Write a dendrogram-angle layout of an existing Poincaré embedding.

    ``spread=0`` copies ``adata.obsm[basis]`` exactly. Larger values rotate
    each group about the origin so dendrogram leaves occupy a wider fan.
    Radii, and therefore hyperbolic distance from the origin, stay put.
    """
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} is not a column of adata.obs.")
    coordinates = np.asarray(adata.obsm[basis])
    groups = adata.obs[groupby].astype(str).to_numpy()
    adata.obsm[key_added] = fan_dendrogram_angles(coordinates, groups, spread=spread)
    return adata if copy else None
