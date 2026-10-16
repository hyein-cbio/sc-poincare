"""Topology choice without changing Gaussian weights, RFA or the default path."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components

from sc_poincare import _proximity
from sc_poincare._fitting import fit_disk_embedding


@pytest.mark.parametrize('mode', ['mutual', 'union'])
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_raw_topology_matches_independent_dense_knn(mode, metric, monkeypatch):
    features = np.random.default_rng(71).normal(size=(17, 4))
    k, sigma = 3, 2.
    distances = cdist(features, features, metric=metric)
    np.fill_diagonal(distances, np.inf)
    directed = np.zeros((len(features), len(features)), dtype=bool)
    directed[np.arange(len(features))[:, None], np.argsort(distances, axis=1)[:, :k]] = True
    expected = directed & directed.T if mode == 'mutual' else directed | directed.T
    original = _proximity._stitch_components
    seen = []
    def capture(graph, *args, **kwargs):
        seen.append(graph.copy())
        return original(graph, *args, **kwargs)
    monkeypatch.setattr(_proximity, '_stitch_components', capture)
    prox = _proximity.estimate_proximity(features, k, sigma, metric=metric, graph_mode=mode)
    topology = seen[0].copy()
    topology.data[:] = 1
    np.testing.assert_array_equal(topology.toarray().astype(bool), expected)
    assert prox.graph_mode == mode
    assert prox.diagnostics['n_components_before_stitching'] == connected_components(seen[0], directed=False)[0]
    assert prox.diagnostics['n_isolated_before_stitching'] == int(np.sum(expected.sum(axis=1) == 0))
    assert prox.diagnostics['n_bridges_added'] == prox.diagnostics['n_components_before_stitching'] - 1
    # Independent inverse checks that union changes only the input graph,
    # not the RFA formula or its normalization.
    weights = prox.weights.toarray()
    expected_rfa = np.linalg.inv(np.eye(len(features)) + np.diag(weights.sum(axis=1)) - weights)
    np.testing.assert_allclose(prox.accessibility, expected_rfa, atol=2e-15)
    rows, cols = prox.weights.nonzero()
    actual_distance = cdist(features, features, metric=metric)[rows, cols]
    np.testing.assert_allclose(weights[rows, cols], np.exp(-.5 * (actual_distance / sigma)**2), atol=2e-15)


def test_union_keeps_reverse_only_neighbors_and_needs_no_bridge():
    features = np.array([[0.], [.1], [.4], [1.2], [2.5]])
    diagnostics = {}
    weights, lengths = _proximity.local_affinity(features, 1, 2., graph_mode='union', diagnostics=diagnostics)
    # 1 does not choose 2; the reverse-only edge must still be present.
    assert lengths[1, 2] == pytest.approx(.3)
    assert weights[1, 2] == pytest.approx(np.exp(-.5 * (.3 / 2.)**2))
    assert diagnostics == dict(n_components_before_stitching=1, n_isolated_before_stitching=0, n_bridges_added=0)


def test_disconnected_union_uses_same_mst_and_underflow_policy():
    x = np.array([[0.], [.1], [2.], [2.1]])
    prox = _proximity.estimate_proximity(x, 1, 2., graph_mode='union')
    assert prox.diagnostics['n_components_before_stitching'] == 2
    assert prox.diagnostics['n_bridges_added'] == 1
    assert prox.weights[1, 2] == pytest.approx(np.exp(-.5 * (1.9 / 2.)**2))
    with pytest.raises(ValueError, match='underflow'):
        _proximity.estimate_proximity(x * 100, 1, 1., graph_mode='union')


def test_union_preserves_duplicate_edges_and_caps_k():
    weights, lengths = _proximity.local_affinity(np.zeros((5, 2)), 20, 1., graph_mode='union')
    assert lengths.nnz == weights.nnz == 20
    np.testing.assert_array_equal(weights.toarray(), np.ones((5, 5)) - np.eye(5))
    assert np.all(lengths.data == 0)


@pytest.mark.parametrize('mode', ['both', '', 'Union', None, True, [], np.array(['union'])])
def test_invalid_mode_fails_before_neighbor_search(mode, monkeypatch):
    monkeypatch.setattr(_proximity, 'NearestNeighbors', lambda **kw: pytest.fail('Search must not run'))
    with pytest.raises(ValueError, match='graph_mode'):
        _proximity.estimate_proximity(np.ones((5, 2)), 2, 1., graph_mode=mode)


def test_default_is_exactly_explicit_mutual_through_fitting():
    features = np.random.default_rng(12).normal(size=(12, 4))
    before = _proximity.estimate_proximity(features, 3, 1.)
    explicit = _proximity.estimate_proximity(features, 3, 1., graph_mode='mutual')
    np.testing.assert_array_equal(before.accessibility, explicit.accessibility)
    for attr in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(before.weights, attr), getattr(explicit.weights, attr))
    a = fit_disk_embedding(before.accessibility, connectivities=before.weights, n_epochs=2, random_state=1)
    b = fit_disk_embedding(explicit.accessibility, connectivities=explicit.weights, n_epochs=2, random_state=1)
    np.testing.assert_array_equal(a.coordinates, b.coordinates)
    np.testing.assert_array_equal(a.loss_trace, b.loss_trace)


@pytest.mark.parametrize('mode', ['mutual', 'union'])
def test_anndata_records_mode_and_roundtrips(tmp_path, mode):
    ad = pytest.importorskip('anndata')
    from sc_poincare.tl import poincare_map
    features = np.random.default_rng(91).normal(size=(9, 3))
    data = ad.AnnData(features)
    result = poincare_map(data, n_neighbors=2, n_epochs=1, graph_mode=mode, copy=True)
    assert 'X_poincare' not in data.obsm
    params = result.uns['poincare']['params']
    assert params['graph_mode'] == mode
    assert params['graph_experimental'] == (mode == 'union')
    path = tmp_path / f'{mode}.h5ad'
    result.write_h5ad(path)
    assert ad.read_h5ad(path).uns['poincare']['params']['graph_mode'] == mode


def test_invalid_public_mode_fails_before_graph_work(monkeypatch):
    ad = pytest.importorskip('anndata')
    from sc_poincare import _anndata
    monkeypatch.setattr(_anndata, 'estimate_proximity', lambda *a, **kw: pytest.fail('Graph must not run'))
    with pytest.raises(ValueError, match='graph_mode'):
        _anndata.layout_cells(ad.AnnData(np.ones((5, 2))), graph_mode='invalid')
