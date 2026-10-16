"""AnnData-facing wrappers that follow Scanpy storage conventions."""

from __future__ import annotations

import hashlib
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

from sc_poincare._disk import (
    circular_gap,
    fan_dendrogram_angles,
    jitter_angles,
    repel_points,
    length_from_origin,
    place_point_at_origin,
    polar_angles,
)
from sc_poincare._fitting import fit_disk_embedding
from sc_poincare._objectives import DEFAULT_OBJECTIVE, CONDITIONAL_OBJECTIVE, _validate_objective
from sc_poincare._proximity import estimate_proximity, _validate_graph_mode
from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES, check_dense_budget
from sc_poincare._mu_anndata import METHOD_DEFAULT, layout_mu
from sc_poincare._fit_progress import (
    PoincareProgress, observation_provenance, coordinates_sha256,
    record_root_display, record_annotation,
)

DEFAULT_BASIS = "X_poincare"
DEFAULT_UNS = "poincare"
DEFAULT_PSEUDOTIME = "poincare_pseudotime"
DEFAULT_LINEAGE = "poincare_lineage"


def _as_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _choose_features(adata, use_rep: str | None, n_pcs: int | None, *, reject_complex=False) -> tuple[np.ndarray, str]:
    def checked(matrix):
        if reject_complex and np.iscomplexobj(matrix):
            raise ValueError('MU features must be real; complex coordinates are not silently cast.')
        return matrix
    if use_rep == "X":
        return _as_dense(checked(adata.X)), "X"
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"Representation {use_rep!r} is not present in adata.obsm.")
        features = np.asarray(checked(adata.obsm[use_rep]), dtype=np.float64)
        if n_pcs is not None:
            features = features[:, :n_pcs]
        return features, use_rep

    n_vars = int(adata.n_vars)
    if n_vars <= 100:
        return _as_dense(checked(adata.X)), "X"
    if "X_pca" in adata.obsm:
        pcs = np.asarray(checked(adata.obsm["X_pca"]), dtype=np.float64)
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
    adata, *, method: str = 'mu', n_neighbors=METHOD_DEFAULT,
    sigma=METHOD_DEFAULT, gamma=METHOD_DEFAULT, n_components: int = 2,
    use_rep: str | None = None, n_pcs: int | None = None,
    metric: str = 'euclidean', n_epochs=METHOD_DEFAULT, learning_rate=METHOD_DEFAULT,
    batch_size: int | None = None, burn_in_epochs=METHOD_DEFAULT,
    min_delta=METHOD_DEFAULT, patience=METHOD_DEFAULT,
    root: int | str | None = None, random_state=METHOD_DEFAULT,
    key_added: str | None = None, copy: bool = False, verbose: bool = False,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    max_grad_norm=METHOD_DEFAULT, objective=METHOD_DEFAULT, graph_mode=METHOD_DEFAULT,
    mu_config=None, local_bandwidth=METHOD_DEFAULT, progress_callback=None,
    resume=None, checkpoint_every=METHOD_DEFAULT, initial_coordinates=None,
):
    """Write a Scanpy-compatible disk embedding; MU is the provisional default.

    ``method='mu'`` uses mutual-global stress + union-local cross-entropy and
    tangent Adam. Defaults: global local-teacher bandwidth, 1500 updates, gamma
    .1*2/3 and learning rate .02*2/3. ``mu_config`` accepts GraphStressConfig;
    explicitly supplied shared knobs override that configuration. Neighbor
    counts are capped at n_obs-1 and both requested/effective values are saved.

    ``method='rfa'`` restores the previous corrected RFA fitter and ALL its old
    numerical defaults (sigma=2, gamma=2, 250 epochs, lr=.01, burn-in=30,
    min_delta=1e-4, patience=40, max_grad_norm=2.5, symmetric_kl, mutual graph).
    Legacy RFA-only knobs require that explicit method; they are never silently
    ignored or reinterpreted as MU parameters. The default switch intentionally
    changes numerical results/runtime and requires the Torch extra for MU.

    Shared storage/copy/root/representation conventions are unchanged. MU is
    Euclidean-feature, 2-D Poincare only; RFA retains other metrics/dimensions.
    progress_callback receives independent PoincareProgress training snapshots.
    MU checkpoint payloads can be persisted externally and supplied as resume,
    with explicit TOTAL n_epochs and identical input cells/order/features,
    config/source/Torch environment. No Torch state is put into H5AD uns.
    RFA initial_coordinates is a coordinate-only NEW run, not optimizer resume.
    checkpoint_every controls MU callback/state intervals; RFA callbacks are
    evaluated each epoch. Root translation is display-only, after fitting.
    """
    if not isinstance(method, str) or method not in ('mu', 'rfa'):
        raise ValueError("method must be 'mu' or 'rfa'.")
    if method == 'rfa':
        if mu_config is not None or local_bandwidth is not METHOD_DEFAULT or resume is not None or checkpoint_every is not METHOD_DEFAULT:
            raise ValueError('mu_config, local_bandwidth, resume and checkpoint_every are MU-only; '
                             'RFA supports initial_coordinates as a new warm-started run.')
        def old(value, default):
            return default if value is METHOD_DEFAULT else value
        return _layout_rfa(adata, n_neighbors=old(n_neighbors,15), sigma=old(sigma,2.),
            gamma=old(gamma,2.), n_components=n_components, use_rep=use_rep, n_pcs=n_pcs,
            metric=metric, n_epochs=old(n_epochs,250), learning_rate=old(learning_rate,.01),
            batch_size=batch_size, burn_in_epochs=old(burn_in_epochs,30),
            min_delta=old(min_delta,1e-4), patience=old(patience,40), root=root,
            random_state=old(random_state,0), key_added=key_added, copy=copy, verbose=verbose,
            max_dense_bytes=max_dense_bytes, max_grad_norm=old(max_grad_norm,2.5),
            objective=old(objective,DEFAULT_OBJECTIVE), graph_mode=old(graph_mode,'mutual'),
            progress_callback=progress_callback, initial_coordinates=initial_coordinates)
    legacy = dict(sigma=sigma, burn_in_epochs=burn_in_epochs, min_delta=min_delta,
                  patience=patience, max_grad_norm=max_grad_norm, objective=objective,
                  graph_mode=graph_mode)
    supplied = [name for name,value in legacy.items() if value is not METHOD_DEFAULT]
    if supplied or initial_coordinates is not None:
        names = ', '.join(supplied + (['initial_coordinates'] if initial_coordinates is not None else []))
        raise ValueError(f"{names} are RFA-only settings; add method='rfa' to preserve the old behavior. "
                         'MU uses mu_config/local_bandwidth and full Adam resume instead.')
    if not isinstance(metric, str) or metric != 'euclidean':
        raise ValueError("MU uses Euclidean input distances; use method='rfa' for other metrics.")
    return layout_mu(adata, n_neighbors=n_neighbors, gamma=gamma, n_components=n_components,
        use_rep=use_rep, n_pcs=n_pcs, n_epochs=n_epochs, learning_rate=learning_rate,
        batch_size=batch_size, root=root, random_state=random_state, key_added=key_added,
        copy=copy, verbose=verbose, max_dense_bytes=max_dense_bytes, mu_config=mu_config,
        local_bandwidth=local_bandwidth, progress_callback=progress_callback, resume=resume,
        checkpoint_every=100 if checkpoint_every is METHOD_DEFAULT else checkpoint_every)


def _layout_rfa(
    adata,
    *,
    n_neighbors: int = 15,
    sigma: float | str = 2.0,
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
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    max_grad_norm: float | None = 2.5,
    objective: str = DEFAULT_OBJECTIVE,
    graph_mode: str = "mutual",
    progress_callback=None,
    initial_coordinates=None,
):
    """Embed cells on the Poincaré disk and store the result on ``adata``.

    Parameters
    ----------
    adata
        Annotated cell-by-feature matrix.
    n_neighbors
        Directed neighbor count used for the local graph (paper: ``k``).
    graph_mode
        ``'mutual'`` (default) keeps reciprocal neighbors, as in the paper.
        Opt-in experimental ``'union'`` keeps either direction. Both use the
        same component-MST stitching, actual Gaussian weights and RFA objective.
        Union changes the target and degree distribution; it is not an automatic
        quality improvement. Sigma='median' resolves on the selected graph.
    sigma
        Gaussian kernel width on neighborhood edge lengths (paper: ``σ``).
        Explicitly opt into ``'median'`` to use the median positive stitched
        graph edge length. This chooses one global bandwidth without labels
        or feature rescaling; it is not a reproduction of the paper's settings.
    gamma
        Softmax temperature on hyperbolic distances (paper: ``γ``).
    objective
        ``'symmetric_kl'`` (default) includes self probabilities. Experimental
        ``'conditional_symmetric_kl'`` excludes self from BOTH target and
        prediction and retains every other cell. This changes the objective,
        may lose isolation information, and is not the paper's self-inclusive KL.
    n_components
        Embedding dimension. Two is the usual visualization setting.
    use_rep
        Feature source. ``'X'`` or any ``adata.obsm`` key. If omitted, ``X`` is
        used when there are at most 100 features, otherwise ``X_pca``.
    n_pcs
        Optional column cap when the representation is a PCA (or similar) matrix.
    metric
        Local distance used to build the neighborhood (``euclidean``, ``cosine``, ...).
    n_epochs, learning_rate, burn_in_epochs, min_delta, patience
        Riemannian fitting controls. After burn-in, patience counts epochs since
        the last cumulative improvement of at least min_delta in the global loss.
    batch_size
        Row chunk size (default at most 256 with PyTorch). All comparison cells
        remain in every row; gradients accumulate before one update per epoch.
    max_grad_norm
        Positive cap on each gradient's Riemannian norm (default 2.5); None
        disables clipping. Applied after global gradient accumulation.
    max_dense_bytes
        Dense working-array budget, default 512 MiB, estimated as four float64
        N×N arrays. Checked before feature densification. This is not a total RSS
        limit. Increase explicitly or use None to disable; RFA is still O(N²).
    root
        Optional cell (integer position or ``obs_names`` label) placed at the origin
        after fitting, via a distance-preserving disk translation.
    random_state
        Seed for spectral initialization and its small noise (or random fallback).
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
    _validate_objective(objective)
    _validate_graph_mode(graph_mode)
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError('progress_callback must be callable or None.')
    check_dense_budget(adata.n_obs, max_dense_bytes)
    if copy:
        adata = adata.copy()

    features, resolved_rep = _choose_features(adata, use_rep, n_pcs)
    if features.shape[0] != adata.n_obs:
        raise ValueError("Feature matrix must have one row per observation.")

    root_index = _resolve_cell_index(adata, root)
    provenance = dict(observation_provenance(adata), resolved_rep=resolved_rep,
                      n_features=int(features.shape[1]),
                      feature_sha256=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest())
    proximity = estimate_proximity(
        features,
        n_neighbors=n_neighbors,
        kernel_width=sigma,
        metric=metric,
        max_dense_bytes=max_dense_bytes,
        graph_mode=graph_mode,
    )
    observed = {'best_loss': np.inf, 'best_epoch': 0}
    def progress(p):
        if p.epoch == 0:
            observed['initial_loss'] = p.loss
        if p.loss < observed['best_loss']:
            observed.update(best_loss=p.loss, best_epoch=p.epoch)
        observed['current_loss'] = p.loss
        if progress_callback is not None:
            progress_callback(PoincareProgress('rfa', p.epoch, p.loss, p.best_loss,
                observed['best_epoch'], p.coordinates, p.best_coordinates, None,
                p.initialization, p.initialization_reason))
    report = fit_disk_embedding(
        proximity.accessibility,
        connectivities=proximity.weights,
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
        max_dense_bytes=max_dense_bytes,
        max_grad_norm=max_grad_norm,
        objective=objective,
        initial_coordinates=initial_coordinates,
        progress_callback=progress,
    )
    current_identity = observation_provenance(adata)
    current_features, _ = _choose_features(adata, resolved_rep, int(features.shape[1]))
    if (current_identity['obs_names_sha256'] != provenance['obs_names_sha256']
            or hashlib.sha256(np.ascontiguousarray(current_features).tobytes()).hexdigest() != provenance['feature_sha256']):
        raise ValueError('Cell identities or fitted features changed during fitting; no result slots were written.')
    coordinates = report.coordinates
    raw_best_sha256 = coordinates_sha256(coordinates)
    if root_index is not None:
        coordinates = place_point_at_origin(coordinates, root_index)

    obsm_key, uns_key = _obsm_and_uns_keys(key_added)
    previous = adata.uns.get(uns_key, {})
    owned_teacher = f'{uns_key}_local_teacher'
    if (isinstance(previous, dict) and isinstance(previous.get('method'), str) and previous.get('method') == 'mu'
            and previous.get('local_teacher_key') == owned_teacher and owned_teacher in adata.obsp):
        del adata.obsp[owned_teacher]
    adata.obsm[obsm_key] = coordinates
    connectivities_key = f"{uns_key}_connectivities"
    adata.obsp[connectivities_key] = proximity.weights
    adata.uns[uns_key] = {
        "schema_version": 2,
        "method": "rfa",
        "params": {
            "method": "rfa",
            "n_neighbors": int(n_neighbors),
            "graph_mode": proximity.graph_mode,
            "graph_experimental": proximity.graph_mode == "union",
            "sigma": float(proximity.kernel_width),
            "sigma_requested": sigma if isinstance(sigma, str) else float(sigma),
            "gamma": float(gamma),
            "n_components": int(n_components),
            "use_rep": resolved_rep,
            "n_pcs": None if n_pcs is None else int(n_pcs),
            "metric": metric,
            "n_epochs_requested": int(n_epochs),
            "n_epochs_run": int(report.n_epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(report.batch_size),
            "optimization": "exact_global_row_chunks",
            "objective": report.objective,
            "objective_experimental": report.objective == CONDITIONAL_OBJECTIVE,
            "probability_computation": report.probability_computation,
            "zero_target_policy": report.zero_target_policy,
            "gradient_clipping": "riemannian_norm",
            "max_grad_norm": None if max_grad_norm is None else float(max_grad_norm),
            "affinity_kernel": "gaussian_all_edges",
            "max_dense_bytes": None if max_dense_bytes is None else int(max_dense_bytes),
            "burn_in_epochs": int(burn_in_epochs),
            "root": None if root is None else (int(root) if isinstance(root, (int, np.integer)) else str(root)),
            "random_state": None if isinstance(random_state, np.random.Generator) else random_state,
            "backend": report.backend,
            "initialization": report.initialization,
            "initialization_reason": report.initialization_reason,
        },
        "connectivities_key": connectivities_key,
        "connectivities_semantics": "gaussian_affinities",
        "proximity_diagnostics": proximity.diagnostics,
        "loss": np.asarray(report.loss_trace, dtype=np.float64),
        "best_loss": float(report.best_loss),
        "best_epoch": int(observed['best_epoch']),
        "initial_loss": float(observed['initial_loss']),
        "current_loss": float(observed['current_loss']),
        "current_epoch": int(report.n_epochs),
        "loss_includes_epoch_zero": False,
        "provenance": provenance,
        "coordinate_space": "raw_training" if root_index is None else "root_isometry",
        "raw_best_sha256": raw_best_sha256,
        "display_coordinates_sha256": coordinates_sha256(coordinates),
        "stop_reason": report.stop_reason,
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
    if not (dest.startswith("X_") or dest in adata.obsm or key_added is None):
        dest = f"X_{dest}"
    adata.obsm[dest] = shifted
    record_root_display(adata, basis, dest, shifted, root)
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
        record_root_display(adata, basis, basis, coordinates, root)
    adata.obs[key_added] = length_from_origin(coordinates)
    record_annotation(adata, basis, key_added, 'hyperbolic_root_distance')
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
        record_root_display(adata, basis, basis, coordinates, root)
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
    record_annotation(adata, basis, key_added, 'angular_lineages', n_lineages=int(n_lineages))
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
