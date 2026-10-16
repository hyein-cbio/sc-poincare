"""Explicit bandwidth recovery, optimizer stopping, and fallback provenance."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse

from sc_poincare import _fitting, _proximity


def test_median_bandwidth_matches_explicit_width_and_feature_unit_changes():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(25, 4))
    original = features.copy()
    automatic = _proximity.estimate_proximity(features, 5, 'median')
    explicit = _proximity.estimate_proximity(features, 5, automatic.kernel_width)
    rescaled = _proximity.estimate_proximity(features * 13, 5, 'median')
    np.testing.assert_array_equal(features, original)
    np.testing.assert_allclose(automatic.weights.toarray(), explicit.weights.toarray(), atol=1e-12)
    np.testing.assert_allclose(automatic.accessibility, explicit.accessibility, atol=1e-12)
    np.testing.assert_allclose(automatic.accessibility, rescaled.accessibility, atol=1e-12)
    assert rescaled.kernel_width == pytest.approx(13 * automatic.kernel_width)
    assert automatic.diagnostics['sigma_effective'] == automatic.kernel_width


def test_median_bandwidth_can_recover_underflow_without_weak_bridge_floor():
    features = np.array([[0.], [.1], [100.], [100.1], [200.], [200.1]])
    with pytest.raises(ValueError, match='underflow'):
        _proximity.estimate_proximity(features, 5, 1.)
    with warnings.catch_warnings():
        warnings.simplefilter('error', _proximity.ProximityWarning)
        recovered = _proximity.estimate_proximity(features, 5, 'median')
    assert recovered.diagnostics['self_mass_gt_0_99_count'] == 0
    weights, lengths = _proximity.local_affinity(features, 5, 'median')
    positive = lengths.data[lengths.data > 0]
    assert recovered.kernel_width == np.median(positive)
    edges = lengths.tocoo()
    np.testing.assert_allclose(weights.toarray()[edges.row, edges.col],
                               np.exp(-.5 * (edges.data / recovered.kernel_width)**2), atol=0., rtol=1e-12)


def test_median_bandwidth_does_not_promise_to_recover_every_graph():
    # Most edges are tiny; one very distant component still cannot be connected
    # numerically with their median scale. Do not silently invent its weight.
    features = np.array([[0.], [.1], [.2], [1000.], [1000.1], [1000.2]])
    with pytest.raises(ValueError, match='underflow'):
        _proximity.estimate_proximity(features, 2, 'median')


def test_no_positive_edge_distances_need_explicit_bandwidth():
    with pytest.raises(ValueError, match='all graph edge distances are zero'):
        _proximity.estimate_proximity(np.zeros((6, 2)), 5, 'median')
    explicit = _proximity.estimate_proximity(np.zeros((6, 2)), 5, 1.)
    assert np.all(np.isfinite(explicit.accessibility))


@pytest.mark.parametrize('value', ['auto', '2', '', None, [], np.nan, np.inf, True])
def test_invalid_bandwidth_policy_rejected(value):
    with pytest.raises(ValueError, match='kernel_width'):
        _proximity.local_affinity(np.arange(4.)[:, None], 2, value)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_early_stopping_accumulates_small_improvements(monkeypatch, backend):
    # Each step improves .06, smaller than min_delta=.1. Every two steps
    # together improve .12, so patience=2 must not stop a steadily improving fit.
    count = iter(range(20))
    if backend == 'numpy':
        monkeypatch.setattr(_fitting, 'torch', None)
        monkeypatch.setattr(_fitting, 'evaluate_alignment', lambda *a, **k: 10 - .06 * next(count))
        monkeypatch.setattr(_fitting, '_numeric_row_grad', lambda coords, *a, **k: np.zeros_like(coords))
    else:
        pytest.importorskip('torch')
        monkeypatch.setattr(_fitting, '_torch_alignment_loss', lambda p, d, t, **k: d.sum() * 0. + 10 - .06 * next(count))
    report = _fitting.fit_disk_embedding(np.full((4, 4), .25), n_epochs=6, patience=2,
                                        min_delta=.1, burn_in_epochs=0, batch_size=4)
    assert report.n_epochs == 6
    assert report.stop_reason == 'epoch_limit'
    assert report.best_loss == pytest.approx(9.64)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_patience_starts_after_burn_in(monkeypatch, backend):
    if backend == 'numpy':
        monkeypatch.setattr(_fitting, 'torch', None)
        monkeypatch.setattr(_fitting, 'evaluate_alignment', lambda *a, **k: 10.)
        monkeypatch.setattr(_fitting, '_numeric_row_grad', lambda coords, *a, **k: np.zeros_like(coords))
    else:
        pytest.importorskip('torch')
        monkeypatch.setattr(_fitting, '_torch_alignment_loss', lambda p, d, t, **k: d.sum() * 0. + 10.)
    report = _fitting.fit_disk_embedding(np.full((4, 4), .25), n_epochs=12, patience=2,
                                        min_delta=.1, burn_in_epochs=5, batch_size=4)
    assert report.n_epochs == 7
    assert report.stop_reason == 'patience'
    assert report.best_loss == 10.


def test_finite_difference_near_training_boundary_matches_autograd():
    torch = pytest.importorskip('torch')
    points = np.array([[.99999, 0.], [.8, .1], [-.3, .2]])
    target = np.full((3, 3), 1 / 3)
    coords = torch.tensor(points, requires_grad=True)
    loss = _fitting._torch_sym_kl(torch.tensor(target), torch.softmax(-_fitting._torch_disk_length(coords) / 2, dim=1))
    loss.backward()
    actual = _fitting._numeric_row_grad(points, target, 2., np.arange(3))
    np.testing.assert_allclose(actual, coords.grad.numpy(), rtol=1e-5, atol=1e-6)


def test_finite_difference_close_points_matches_autograd():
    torch = pytest.importorskip('torch')
    points = np.array([[.9, 0.], [.9000001, .0000001], [-.3, .2]])
    target = np.array([[.5, .4, .1], [.4, .5, .1], [.1, .1, .8]])
    coords = torch.tensor(points, requires_grad=True)
    loss = _fitting._torch_sym_kl(torch.tensor(target), torch.softmax(-_fitting._torch_disk_length(coords) / 2, dim=1))
    loss.backward()
    actual = _fitting._numeric_row_grad(points, target, 2., np.arange(3))
    np.testing.assert_allclose(actual, coords.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_spectral_fallback_is_recorded_in_report(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('forced eigensolver failure')
    monkeypatch.setattr(np.linalg, 'eigh', fail)
    graph = sparse.csr_matrix(np.ones((6, 6)) - np.eye(6))
    with pytest.warns(UserWarning, match='eigensolver'):
        report = _fitting.fit_disk_embedding(np.eye(6), connectivities=graph, n_epochs=1)
    assert report.initialization == 'random_fallback'
    assert report.initialization_reason == 'eigensolver_failure'
    random = _fitting.fit_disk_embedding(np.eye(6), n_epochs=1)
    assert random.initialization == 'random'
    assert random.initialization_reason == ''


def test_anndata_preserves_requested_and_effective_bandwidth_and_stop_reason(tmp_path):
    ad = pytest.importorskip('anndata')
    from sc_poincare._anndata import layout_cells
    data = ad.AnnData(np.random.default_rng(1).normal(size=(12, 3)))
    layout_cells(data, method='rfa', sigma='median', n_neighbors=4, n_epochs=2)
    stored = data.uns['poincare']
    assert stored['params']['sigma_requested'] == 'median'
    assert isinstance(stored['params']['sigma'], float)
    assert stored['params']['sigma'] == stored['proximity_diagnostics']['sigma_effective']
    assert stored['stop_reason'] == 'epoch_limit'
    assert stored['params']['initialization'] == 'spectral'
    path = tmp_path / 'median.h5ad'
    data.write_h5ad(path)
    assert ad.read_h5ad(path).uns['poincare']['params']['sigma_requested'] == 'median'


def test_invalid_root_fails_before_graph_computation(monkeypatch):
    ad = pytest.importorskip('anndata')
    from sc_poincare import _anndata
    monkeypatch.setattr(_anndata, 'estimate_proximity', lambda *a, **k: pytest.fail('Graph should not run'))
    with pytest.raises(KeyError):
        _anndata.layout_cells(ad.AnnData(np.ones((8, 2))), root='absent')
