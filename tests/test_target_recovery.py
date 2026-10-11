"""Explicit bandwidth recovery, optimizer stopping, and fallback provenance."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse

from sc_poincare import _fitting, _proximity


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
        monkeypatch.setattr(_fitting, '_torch_sym_kl', lambda p, q: q.sum() * 0. + 10 - .06 * next(count))
    report = _fitting.fit_disk_embedding(np.eye(4), n_epochs=6, patience=2,
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
        monkeypatch.setattr(_fitting, '_torch_sym_kl', lambda p, q: q.sum() * 0. + 10.)
    report = _fitting.fit_disk_embedding(np.eye(4), n_epochs=12, patience=2,
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
