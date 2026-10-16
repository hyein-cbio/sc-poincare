"""Read-only neighborhood topology inspection; no RFA or embedding is fitted."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from sc_poincare.evaluation import _matrix, _positive_integer


def result_status(adata, *, basis='X_poincare'):
    """Read-only freshness check for stored coordinates and owned annotations.

    Old H5AD files without coordinate/annotation provenance remain readable:
    missing evidence is reported as unknown, never silently marked current.
    Re-fitting does not delete pre-existing observation columns or recalculate
    pseudotime/lineages; rerun those helpers after changing the embedding.
    """
    from sc_poincare._fit_progress import coordinates_sha256, annotation_sha256
    if basis not in adata.obsm:
        raise KeyError(f'Embedding {basis!r} is missing.')
    key = basis[2:] if basis.startswith('X_') else basis
    metadata = adata.uns.get(key, {})
    if not isinstance(metadata, dict):
        metadata = {}
    params = metadata.get('params', {})
    if not isinstance(params, dict):
        params = {}
    method = metadata.get('method', params.get('method'))
    method = method if isinstance(method, str) else None
    objective = params.get('objective')
    if method is None and isinstance(objective, str) and objective in ('symmetric_kl', 'conditional_symmetric_kl'):
        method = 'rfa'
    digest = coordinates_sha256(adata.obsm[basis])
    stored = metadata.get('display_coordinates_sha256')
    stored = stored if isinstance(stored, str) else None
    fields = metadata.get('derived_fields', {})
    if not isinstance(fields, dict):
        fields = {}
    annotations = {}
    for name, info in fields.items():
        if name not in adata.obs:
            annotations[name] = 'missing'
        elif not isinstance(info, dict) or info.get('basis') != basis:
            annotations[name] = 'unknown'
        elif not isinstance(info.get('source_coordinates_sha256'), str):
            annotations[name] = 'unknown'
        elif info['source_coordinates_sha256'] != digest:
            annotations[name] = 'stale'
        elif info.get('annotation_sha256') is None:
            annotations[name] = 'unknown'
        else:
            annotations[name] = 'current' if info['annotation_sha256'] == annotation_sha256(adata.obs[name]) else 'stale'
    if basis == 'X_poincare':
        for name in ('poincare_pseudotime', 'poincare_lineage'):
            if name in adata.obs and name not in annotations:
                annotations[name] = 'unknown'
    return dict(method=method, coordinate_space=metadata.get('coordinate_space', 'unknown'),
                coordinates_match_record=None if stored is None else stored == digest,
                annotations=annotations)


def graph_topology(features, *, n_neighbors=15, metric='euclidean'):
    """Reconstruct raw mutual/union graphs before stitching or kernel weighting.

    Uses the same self-by-index exclusion and k+1 query as fitting. Explicit
    zero-distance neighbors remain edges. Cutoff ties are diagnosed separately
    with one extra neighbor: a reconstruction is not a certificate of the exact
    fitted tie choices on every numerical backend. No labels, kernel tuning,
    dense global targets or display transformations are involved.
    """
    features = _matrix(features, 'features')
    if len(features) < 3:
        raise ValueError('Need at least 3 cells to inspect graph topology.')
    requested = _positive_integer(n_neighbors, 'n_neighbors')
    n = len(features); k = min(requested, n - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=1).fit(features)
    distances, indices = model.kneighbors(features)
    if not np.isfinite(distances).all():
        raise ValueError('Neighbor distances must be finite.')
    rows, columns = [], []
    for i in range(n):
        selected = indices[i, indices[i] != i][:k]
        rows.extend([i] * k); columns.extend(selected.tolist())
    directed = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, columns)), shape=(n, n))
    graphs = dict(mutual=directed.multiply(directed.T), union=directed.maximum(directed.T))
    result = dict(scope='unweighted_pre_stitch_reconstruction', n_obs=n,
                  n_neighbors_requested=requested, n_neighbors_effective=k, cutoff_tie_rows=0)
    if k < n - 1:
        extra_distances, extra_indices = model.kneighbors(features, n_neighbors=k + 2)
        ambiguous = 0
        for i in range(n):
            other = extra_distances[i, extra_indices[i] != i]
            ambiguous += bool(np.isclose(other[k - 1], other[k], rtol=1e-12, atol=0))
        result['cutoff_tie_rows'] = ambiguous
    for mode, graph in graphs.items():
        graph = graph.tocsr()
        count, labels = connected_components(graph, directed=False)
        result[mode] = dict(n_components_before_stitching=int(count),
            n_isolated_before_stitching=int(np.count_nonzero(np.diff(graph.indptr) == 0)),
            n_edges_before_stitching=int(graph.nnz // 2),
            required_bridges_for_connection=int(count - 1),
            largest_component_fraction=float(np.bincount(labels).max() / n))
    return result
