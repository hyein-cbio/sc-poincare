"""Small independent references for exact graph, RFA and global fitting."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances

from sc_poincare import _fitting, _proximity


def _target(n=9):
    rng = np.random.default_rng(4)
    matrix = rng.uniform(0.01, 1.0, (n, n))
    matrix = (matrix + matrix.T) / 2
    return matrix / matrix.sum(axis=1, keepdims=True)


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
