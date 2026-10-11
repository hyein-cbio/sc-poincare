"""Small independent references for exact graph, RFA and global fitting."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances

from sc_poincare import _fitting, _proximity
from sc_poincare._resources import check_dense_budget


def _target(n=9):
    rng = np.random.default_rng(4)
    matrix = rng.uniform(0.01, 1.0, (n, n))
    matrix = (matrix + matrix.T) / 2
    return matrix / matrix.sum(axis=1, keepdims=True)


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_chunked_bridges_match_dense_kruskal(metric):
    rng = np.random.default_rng(12)
    features = rng.normal(size=(21, 5))
    # Three internally connected components. The test reference sorts all pairs.
    graph = sparse.block_diag([sparse.csr_matrix(np.ones((7, 7)))] * 3, format='csr')
    graph.setdiag(0)
    graph.eliminate_zeros()
    stitched = _proximity._stitch_components(graph, features, metric=metric, block_size=3)
    distances = pairwise_distances(features, metric=metric)
    labels = np.repeat(np.arange(3), 7)
    expected = set()
    pairs = sorted((float(distances[i, j]), i, j) for i in range(21) for j in range(i + 1, 21))
    for distance, i, j in pairs:
        if labels[i] != labels[j]:
            old, new = labels[j], labels[i]
            labels[labels == old] = new
            expected.add((i, j))
    actual = set()
    for i, j in zip(*stitched.nonzero()):
        if i < j and i // 7 != j // 7:
            actual.add((i, j))
    assert actual == expected
    for i, j in actual:
        assert stitched[i, j] == pytest.approx(distances[i, j], abs=1e-12)
    assert connected_components(stitched, directed=False)[0] == 1


def test_zero_length_bridge_is_retained():
    features = np.array([[0.], [1.], [0.], [-1.]])
    graph = sparse.coo_matrix(([1., 1., 1., 1.], ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(4, 4)).tocsr()
    stitched = _proximity._stitch_components(graph, features, block_size=1)
    assert 2 in stitched.indices[stitched.indptr[0]:stitched.indptr[1]]
    assert stitched[0, 2] == 0
    assert connected_components(stitched, directed=False)[0] == 1


def test_connected_graph_needs_no_all_pair_distance_matrix(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No pairwise distances are needed for a connected kNN graph')
    monkeypatch.setattr(_proximity, 'pairwise_distances', forbidden)
    features = np.random.default_rng(0).normal(size=(12, 3))
    affinity, lengths = _proximity.local_affinity(features, 11, 2.)
    assert sparse.issparse(lengths)
    assert affinity.nnz == 12 * 11


def test_duplicate_cells_keep_zero_length_neighbors():
    n = 7
    affinity, _ = _proximity.local_affinity(np.zeros((n, 2)), n - 1, 2.)
    expected = np.ones((n, n)) - np.eye(n)
    np.testing.assert_allclose(affinity.toarray(), expected)
    assert np.all(affinity.diagonal() == 0)


@pytest.mark.parametrize('block_size', [1, 4, 30])
def test_blocked_rfa_matches_direct_inverse(block_size):
    rng = np.random.default_rng(3)
    adjacency = rng.uniform(0.01, 1., (15, 15))
    adjacency = (adjacency + adjacency.T) / 2
    np.fill_diagonal(adjacency, 0.)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    expected = np.linalg.inv(np.eye(15) + laplacian)
    actual = _proximity.forest_accessibility(sparse.csr_matrix(adjacency), block_size=block_size)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_rfa_factorizes_once_and_solves_only_small_rhs(monkeypatch):
    original = _proximity.splu
    shapes = []
    factorizations = []
    def factor(matrix):
        factorizations.append(matrix.shape)
        solver = original(matrix)
        class Solver:
            def solve(self, rhs):
                shapes.append(rhs.shape)
                return solver.solve(rhs)
        return Solver()
    monkeypatch.setattr(_proximity, 'splu', factor)
    _proximity.forest_accessibility(sparse.csr_matrix(np.ones((11, 11))), block_size=3)
    assert factorizations == [(11, 11)]
    assert shapes == [(11, 3), (11, 3), (11, 3), (11, 2)]


def test_dense_guard_runs_before_graph_or_factorization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Guard should run before allocating or solving')
    monkeypatch.setattr(_proximity, 'local_affinity', forbidden)
    monkeypatch.setattr(_proximity, 'splu', forbidden)
    with pytest.raises(MemoryError, match=r'O\(N²\)'):
        _proximity.estimate_proximity(np.zeros((10000, 1)), 15, 2.)
    with pytest.raises(MemoryError):
        _proximity.forest_accessibility(sparse.eye(10000, format='csr'))


def test_anndata_guard_precedes_feature_densification(monkeypatch):
    ad = pytest.importorskip('anndata')
    from sc_poincare import _anndata
    def forbidden(*args, **kwargs):
        raise AssertionError('Features must not be densified before the guard')
    monkeypatch.setattr(_anndata, '_choose_features', forbidden)
    data = ad.AnnData(sparse.csr_matrix((10000, 1)))
    with pytest.raises(MemoryError):
        _anndata.layout_cells(data)


@pytest.mark.parametrize('limit', [0, -1, True, 3.5])
def test_invalid_memory_budget_is_rejected(limit):
    with pytest.raises(ValueError):
        check_dense_budget(5, limit)


def test_explicit_memory_budget_can_be_increased_or_disabled():
    check_dense_budget(10000, 4 * 8 * 10000**2)
    check_dense_budget(10000, None)


@pytest.mark.parametrize('chunk_size', [1, 4, 20])
def test_global_row_chunks_match_full_loss_and_all_coordinate_gradients(chunk_size):
    torch = pytest.importorskip('torch')
    coords = torch.tensor(np.random.default_rng(10).normal(scale=.1, size=(9, 2)), dtype=torch.float64, requires_grad=True)
    target = torch.tensor(_target(), dtype=torch.float64)
    full = _fitting._torch_sym_kl(target, torch.softmax(-_fitting._torch_disk_length(coords) / 2., dim=1))
    full.backward()
    expected_gradient = coords.grad.clone()
    coords.grad = None
    chunked_loss = 0.
    for start in range(0, 9, chunk_size):
        stop = min(start + chunk_size, 9)
        dist = _fitting._torch_disk_length(coords[start:stop], coords, self_columns=torch.arange(start, stop))
        loss = _fitting._torch_sym_kl(target[start:stop], torch.softmax(-dist / 2., dim=1))
        chunked_loss += float(loss.detach())
        loss.backward()
    assert chunked_loss == pytest.approx(float(full.detach()), rel=1e-12)
    np.testing.assert_allclose(coords.grad.numpy(), expected_gradient.numpy(), rtol=1e-10, atol=1e-10)


def test_fit_chunk_sizes_preserve_trajectory_and_postupdate_checkpoints():
    torch = pytest.importorskip('torch')
    target = _target()
    reports = [_fitting.fit_disk_embedding(target, batch_size=chunk, n_epochs=5, patience=5, random_state=5)
               for chunk in [1, 4, 9]]
    for report in reports:
        np.testing.assert_allclose(report.coordinates, reports[-1].coordinates, atol=1e-10)
        np.testing.assert_allclose(report.loss_trace, reports[-1].loss_trace, atol=1e-10)
        assert report.n_epochs == 5
        coordinates = torch.tensor(report.coordinates, dtype=torch.float64)
        loss = _fitting._torch_sym_kl(torch.tensor(target), torch.softmax(-_fitting._torch_disk_length(coordinates) / 2., dim=1))
        assert float(loss) == pytest.approx(min(report.loss_trace), rel=1e-10)
        assert _fitting.evaluate_alignment(report.coordinates, target, 2., block_size=2) == pytest.approx(float(loss), rel=1e-10)


def test_numeric_gradient_uses_global_target_not_a_conditional_submatrix():
    torch = pytest.importorskip('torch')
    points = np.random.default_rng(2).normal(scale=.1, size=(9, 2))
    target = _target()
    coordinates = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    loss = _fitting._torch_sym_kl(torch.tensor(target), torch.softmax(-_fitting._torch_disk_length(coordinates) / 2., dim=1))
    loss.backward()
    rows = np.array([1, 7])
    actual = _fitting._numeric_row_grad(points, target, 2., rows, eps=1e-6)
    np.testing.assert_allclose(actual[rows], coordinates.grad.numpy()[rows], atol=1e-6, rtol=1e-5)
    assert np.all(actual[np.setdiff1d(np.arange(9), rows)] == 0)


def test_numpy_fit_uses_same_objective_across_coordinate_chunks(monkeypatch):
    monkeypatch.setattr(_fitting, 'torch', None)
    target = _target(5)
    full = _fitting.fit_disk_embedding(target, n_epochs=2, patience=2, random_state=3, batch_size=5)
    chunked = _fitting.fit_disk_embedding(target, n_epochs=2, patience=2, random_state=3, batch_size=1)
    np.testing.assert_allclose(full.coordinates, chunked.coordinates, atol=1e-12)
    np.testing.assert_allclose(full.loss_trace, chunked.loss_trace, atol=1e-12)
    assert _fitting.evaluate_alignment(full.coordinates, target, 2.) == pytest.approx(min(full.loss_trace), rel=1e-12)


def test_numpy_large_input_requires_torch(monkeypatch):
    monkeypatch.setattr(_fitting, 'torch', None)
    with pytest.raises(ImportError, match='PyTorch'):
        _fitting.fit_disk_embedding(np.eye(129))


@pytest.mark.parametrize('batch_size', [0, -2, 1.5, True])
def test_invalid_chunk_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        _fitting.fit_disk_embedding(_target(), batch_size=batch_size)


@pytest.mark.parametrize('target', [3., np.ones(4), np.ones((2, 3)), np.zeros((3, 3)), np.full((3, 3), np.nan)])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError):
        _fitting.fit_disk_embedding(target)


def test_nested_connectivity_lists_are_validated_without_attribute_error():
    target = _target(5)
    graph = (np.ones((5, 5)) - np.eye(5)).tolist()
    report = _fitting.fit_disk_embedding(target, connectivities=graph, n_epochs=1)
    assert report.coordinates.shape == (5, 2)
    with pytest.raises(ValueError, match='same shape'):
        _fitting.fit_disk_embedding(target, connectivities=[[0., 1.], [1., 0.]])


def test_sparse_spectral_seed_uses_low_laplacian_modes_and_seeded_solver(monkeypatch):
    seen = {}
    n = 2001
    def fake_solver(matrix, *, k, which, v0):
        seen.update(k=k, which=which, v0=v0.copy())
        x = np.linspace(-1., 1., n)
        return np.array([.8, .9, 1.]), np.column_stack([x**2, x, np.ones(n)])
    monkeypatch.setattr(_fitting, 'eigsh', fake_solver)
    graph = sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format='csr')
    first = _fitting._seed_from_connectivities(graph, 2, np.random.default_rng(4))
    v0 = seen['v0'].copy()
    second = _fitting._seed_from_connectivities(graph, 2, np.random.default_rng(4))
    assert seen['which'] == 'LA'
    assert seen['k'] == 3
    np.testing.assert_array_equal(v0, seen['v0'])
    np.testing.assert_array_equal(first, second)
