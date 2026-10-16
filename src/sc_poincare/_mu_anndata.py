"""MU AnnData adapter. Numerical graph-stress implementation stays separate."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from sc_poincare._disk import place_point_at_origin
from sc_poincare._fit_progress import (
    PoincareProgress, observation_provenance, wrap_mu_checkpoint, unwrap_mu_checkpoint,
)
from sc_poincare._resources import check_dense_budget


class _MethodDefault:
    def __repr__(self):
        return 'method_default'

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_method_default, ())


def _method_default():
    return METHOD_DEFAULT


METHOD_DEFAULT = _MethodDefault()


def layout_mu(adata, *, n_neighbors, gamma, n_components, use_rep, n_pcs,
              n_epochs, learning_rate, batch_size, root, random_state, key_added,
              copy, verbose, max_dense_bytes, mu_config, local_bandwidth,
              progress_callback, resume, checkpoint_every):
    from sc_poincare import experimental as mu
    from sc_poincare._anndata import _choose_features, _resolve_cell_index, _obsm_and_uns_keys
    from sc_poincare.diagnostics import graph_topology

    if progress_callback is not None and not callable(progress_callback):
        raise ValueError('progress_callback must be callable or None.')
    if mu._integer('n_components', n_components) != 2:
        raise ValueError("MU currently supports n_components=2; use method='rfa' for the legacy dimensional API.")
    chunk_size = 256 if batch_size is None else mu._integer('batch_size', batch_size)
    checkpoint_every = mu._integer('checkpoint_every', checkpoint_every)
    check_dense_budget(adata.n_obs, max_dense_bytes)
    if adata.n_obs < 3:
        raise ValueError('Need at least 3 observations for MU.')
    root_index = _resolve_cell_index(adata, root)
    core_state, saved = (None, None) if resume is None else unwrap_mu_checkpoint(resume, adata)
    if core_state is not None:
        try:
            intact = core_state.get('checkpoint_sha256') == mu._checkpoint_hash(core_state)
        except (TypeError, ValueError, KeyError):
            intact = False
        if not intact:
            raise ValueError('MU optimizer checkpoint integrity mismatch.')
    if core_state is not None and n_epochs is METHOD_DEFAULT:
        raise ValueError('Set n_epochs explicitly to the total desired epoch on MU resume.')
    epochs = mu._integer('n_epochs', 1500 if n_epochs is METHOD_DEFAULT else n_epochs, 0)
    if core_state is not None and epochs <= mu._integer('checkpoint epoch', core_state['epoch'], 0):
        raise ValueError('n_epochs is the total target and must exceed the resumed epoch.')
    if mu_config is not None and not isinstance(mu_config, mu.GraphStressConfig):
        raise TypeError('mu_config must be a GraphStressConfig or None.')
    if mu_config is not None:
        config = mu_config
    elif core_state is not None:
        try:
            config = mu.GraphStressConfig(**core_state['parameters'])
        except (KeyError, TypeError, ValueError):
            raise ValueError('Invalid checkpoint MU parameters.') from None
    else:
        config = mu.GraphStressConfig()
    requested_k = config.n_neighbors if n_neighbors is METHOD_DEFAULT else mu._integer('n_neighbors', n_neighbors)
    if saved is not None and mu_config is None and n_neighbors is METHOD_DEFAULT:
        requested_k = saved['n_neighbors_requested']
    requested_k = mu._integer('n_neighbors', requested_k)
    changes = dict(n_neighbors=min(requested_k, adata.n_obs - 1))
    for name, value in [('gamma', gamma), ('learning_rate', learning_rate), ('local_bandwidth', local_bandwidth)]:
        if value is not METHOD_DEFAULT:
            changes[name] = value
    config = replace(config, **changes)
    if random_state is METHOD_DEFAULT:
        seed = 0 if core_state is None else core_state['fit_configuration']['random_state']
    elif core_state is not None and (random_state is None or isinstance(random_state, np.random.Generator)):
        raise ValueError('MU resume requires the saved integer seed, not a new RNG draw.')
    elif random_state is None or isinstance(random_state, np.random.Generator):
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng()
        seed = int(rng.integers(0, 2**32 - 1))
    else:
        seed = mu._integer('random_state', random_state, 0)
    try:
        torch = mu._torch()
    except ImportError:
        raise ImportError("MU requires sc-poincare[torch]. Use method='rfa' for the previous "
                          'NumPy fallback on at most 128 cells.') from None
    if core_state is not None:
        if asdict(config) != core_state['parameters']:
            raise ValueError('MU resume parameters differ; start a new run to retune.')
        expected = dict(geometry='poincare', random_state=seed, chunk_size=chunk_size)
        if core_state['fit_configuration'] != expected:
            raise ValueError('MU resume seed/geometry/chunk size differ.')
        if core_state['torch_version'] != str(torch.__version__):
            raise ValueError('MU resume Torch version differs.')
        if core_state['metadata']['source_hashes'] != mu._source_hashes():
            raise ValueError('MU resume numerical source hashes differ.')
    if copy:
        adata = adata.copy()
    if saved is not None and use_rep is None:
        use_rep = saved['resolved_rep']
        if n_pcs is None and use_rep != 'X':
            n_pcs = saved['n_features']
    features, resolved_rep = _choose_features(adata, use_rep, n_pcs, reject_complex=True)
    if features.shape[0] != adata.n_obs:
        raise ValueError('Feature matrix must have one row per observation.')
    if saved is not None and mu._digest(np.asarray(features, dtype=np.float64)) != saved['feature_sha256']:
        raise ValueError('Checkpoint fitted feature values/order differ; use exactly the original representation.')
    provenance = dict(observation_provenance(adata), resolved_rep=resolved_rep,
                      n_features=int(features.shape[1]), n_neighbors_requested=int(requested_k),
                      feature_sha256=mu._digest(np.asarray(features, dtype=np.float64)))
    setup = mu.prepare_graph_stress(features, config=config, max_dense_bytes=max_dense_bytes)
    initialization = 'classical_scaling' if core_state is None else 'adam_resume'
    def progress(p):
        if verbose:
            print(f'poincare_map MU epoch {p.epoch}/{epochs} loss={p.loss:.6f}')
        if progress_callback is not None:
            progress_callback(PoincareProgress('mu', p.epoch, p.loss, p.best_loss, p.best_epoch,
                p.current, p.best, wrap_mu_checkpoint(p.state, provenance), initialization))
    report = mu.fit_graph_stress(setup, epochs=epochs, random_state=seed, chunk_size=chunk_size,
        max_dense_bytes=max_dense_bytes, resume=core_state,
        callback=progress if progress_callback is not None or verbose else None,
        checkpoint_every=checkpoint_every)
    current_identity = observation_provenance(adata)
    if current_identity['obs_names_sha256'] != provenance['obs_names_sha256']:
        raise ValueError('Cell identities/order changed during fitting; no result slots were written.')
    current_features, _ = _choose_features(adata, resolved_rep, int(features.shape[1]), reject_complex=True)
    if mu._digest(np.asarray(current_features, dtype=np.float64)) != provenance['feature_sha256']:
        raise ValueError('Fitted feature representation changed during fitting; no result slots were written.')
    raw_best = report.coordinates
    coordinates = raw_best if root_index is None else place_point_at_origin(raw_best, root_index)
    checkpoint = wrap_mu_checkpoint(report.state, provenance)
    # Preserve symmetric sparse connectivity semantics in the existing slot;
    # retain the actual directed local teacher in a separately named slot.
    teacher = sparse.csr_matrix(setup.teacher)
    connectivities = (.5 * (teacher + teacher.T)).tocsr()
    obsm_key, uns_key = _obsm_and_uns_keys(key_added)
    graph_key, teacher_key = f'{uns_key}_connectivities', f'{uns_key}_local_teacher'
    params = dict(method='mu', objective='relative_graph_stress_plus_local_cross_entropy',
        objective_experimental=True, graph_mode='mutual_global_union_local',
        n_neighbors=int(config.n_neighbors), n_neighbors_requested=int(requested_k),
        n_components=2, use_rep=resolved_rep, n_pcs=n_pcs,
        metric='euclidean', local_bandwidth=config.local_bandwidth, gamma=config.gamma,
        learning_rate=config.learning_rate, mu_config=asdict(config),
        n_epochs_requested=epochs, n_epochs_run=report.n_epochs, batch_size=chunk_size,
        optimization='tangent_adam_exact_global_row_chunks', probability_computation='log_softmax_nonself',
        affinity_kernel='symmetrized_union_teacher', random_state=seed,
        backend='torch', initialization=initialization, initialization_reason='',
        root=None if root is None else (int(root) if isinstance(root,(int,np.integer)) else str(root)),
        max_dense_bytes=None if max_dense_bytes is None else int(max_dense_bytes),
        resume_from_epoch=None if core_state is None else int(core_state['epoch']),
        checkpoint_every=checkpoint_every)
    topology = graph_topology(features, n_neighbors=config.n_neighbors)
    diagnostics = deepcopy(setup.metadata)
    diagnostics['topology_reconstruction'] = topology
    diagnostics['weighted_local_teacher_components'] = int(connected_components(teacher, directed=False)[0])
    metadata = dict(schema_version=2, method='mu', params=params,
        connectivities_key=graph_key, local_teacher_key=teacher_key,
        connectivities_semantics='symmetric_mean_of_row_normalized_union_teacher',
        proximity_diagnostics=diagnostics,
        provenance=provenance, last_checkpoint_sha256=checkpoint['integrity_sha256'],
        loss=report.loss_trace[1:].copy(), initial_loss=float(report.loss_trace[0]),
        loss_includes_epoch_zero=False, best_loss=report.best_loss, best_epoch=report.best_epoch,
        current_loss=report.current_loss, current_epoch=report.n_epochs, stop_reason='epoch_limit',
        coordinate_space='raw_training' if root_index is None else 'root_isometry',
        raw_best_sha256=mu._digest(raw_best), display_coordinates_sha256=mu._digest(coordinates))
    # No result slots are modified until fitting, callbacks and root geometry
    # have all completed successfully. Never put Torch state into H5AD uns.
    adata.obsm[obsm_key] = coordinates
    adata.obsp[graph_key] = connectivities
    adata.obsp[teacher_key] = teacher
    adata.uns[uns_key] = metadata
    return adata if copy else None
