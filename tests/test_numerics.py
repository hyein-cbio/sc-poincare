"""Regression checks for close-point gradients, metric clipping and target scale."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from sc_poincare import _fitting, _proximity
from sc_poincare._disk import pairwise_disk_length, place_point_at_origin, translate_origin_to, length_from_origin


@pytest.mark.parametrize('radius', [0., .9, .999])
@pytest.mark.parametrize('delta', [1e-6, 1e-9, 1e-12])
def test_near_coincident_points_have_accurate_nonzero_distance(radius, delta):
    points = np.array([[radius, 0.], [radius + delta, delta]])
    separation = np.linalg.norm(points[0] - points[1])
    expected = 2 * np.arcsinh(separation / np.sqrt(np.prod(1 - np.sum(points**2, axis=1))))
    distance = pairwise_disk_length(points)
    assert distance[0, 1] == pytest.approx(expected, rel=1e-12)
    assert distance[0, 1] > 0
    np.testing.assert_array_equal(np.diag(distance), 0.)
    torch = pytest.importorskip('torch')
    coords = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    got = _fitting._torch_disk_length(coords)
    assert float(got[0, 1].detach()) == pytest.approx(expected, rel=1e-12)
    got[0, 1].backward()
    assert torch.isfinite(coords.grad).all()
    assert torch.linalg.vector_norm(coords.grad) > 1


def test_direct_kernel_stays_accurate_above_cdist_matrix_multiply_threshold():
    torch = pytest.importorskip('torch')
    points = np.column_stack([np.full(40, .9), np.arange(40) * 1e-10])
    distance = _fitting._torch_disk_length(torch.tensor(points, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(distance, pairwise_disk_length(points), rtol=1e-12, atol=0.)
    assert np.all(distance[np.triu_indices(40, 1)] > 0.)


def test_close_point_autograd_passes_finite_difference_gradcheck():
    torch = pytest.importorskip('torch')
    left = torch.tensor([[.9, .0]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[.9000001, .0000001]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: _fitting._torch_disk_length(a, b), (left, right),
        eps=1e-10, atol=1e-4, rtol=1e-4,
    )


def test_exact_duplicates_have_zero_distance_and_finite_subgradient():
    torch = pytest.importorskip('torch')
    coords = torch.tensor([[.8, .1], [.8, .1], [0., 0.]], dtype=torch.float64, requires_grad=True)
    distance = _fitting._torch_disk_length(coords)
    assert float(distance[0, 1].detach()) == 0.
    distance.sum().backward()
    assert torch.isfinite(coords.grad).all()
    np.testing.assert_allclose(distance.detach().numpy(), pairwise_disk_length(coords.detach().numpy()), rtol=1e-12)
    # Zero loss at identical distributions must not produce NaN via diagonal norms.
    coords.grad = None
    loss = _fitting._torch_sym_kl(torch.eye(3, dtype=torch.float64),
                                torch.softmax(-_fitting._torch_disk_length(coords), dim=1))
    loss.backward()
    assert torch.isfinite(coords.grad).all()


@pytest.mark.parametrize('cap', [None, .25, 2.5])
def test_clipping_is_in_riemannian_norm_and_matches_numpy(cap):
    torch = pytest.importorskip('torch')
    points = np.array([[0., 0.], [.5, 0.], [.99, 0.]])
    # Same Riemannian gradient norm (=10) at every radius.
    inverse_lambda = (1 - np.sum(points**2, axis=1, keepdims=True)) / 2
    gradient = np.column_stack([np.zeros(3), 10 / inverse_lambda[:, 0]])
    expected = inverse_lambda**2 * gradient
    if cap is not None:
        expected *= min(1., cap / 10.)
    step = 1e-4
    got_np = _fitting._riemannian_retract_numpy(points, gradient, step, cap)
    got_torch = _fitting._riemannian_retract(torch.tensor(points), torch.tensor(gradient), step, cap).numpy()
    np.testing.assert_allclose(got_np, points - step * expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(got_torch, got_np, atol=1e-15)
    actual_norm = np.linalg.norm((points - got_np) / step, axis=1) / inverse_lambda[:, 0]
    np.testing.assert_allclose(actual_norm, 10. if cap is None else cap, rtol=1e-12)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_gradients_fail_instead_of_becoming_zero(value):
    points = np.zeros((2, 2))
    gradient = np.array([[value, 0.], [1., 1.]])
    with pytest.raises(FloatingPointError, match='gradient'):
        _fitting._riemannian_retract_numpy(points, gradient, .01)
    torch = pytest.importorskip('torch')
    with pytest.raises(FloatingPointError, match='gradient'):
        _fitting._riemannian_retract(torch.tensor(points), torch.tensor(gradient), .01)


@pytest.mark.parametrize('cap', [0., -1., np.nan, np.inf, True])
def test_invalid_metric_clip_cap_is_rejected(cap):
    with pytest.raises(ValueError, match='max_grad_norm'):
        _fitting.fit_disk_embedding(np.eye(4), max_grad_norm=cap)


def test_recenter_does_not_apply_optimizer_radius_cap():
    points = np.array([[.999, 0.], [-.999, 0.], [.0, .7], [.2, .1]])
    distances = pairwise_disk_length(points)
    shifted = place_point_at_origin(points, 0)
    assert np.linalg.norm(shifted[1]) > 1 - 1e-5  # Must remain valid, not clipped.
    np.testing.assert_array_equal(shifted[0], np.zeros(2))
    np.testing.assert_allclose(pairwise_disk_length(shifted), distances, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(length_from_origin(shifted), distances[0], atol=1e-8, rtol=1e-9)
    restored = translate_origin_to(shifted, points[0])
    np.testing.assert_allclose(restored, points, atol=1e-9, rtol=1e-9)
    torch = pytest.importorskip('torch')
    got = _fitting._torch_disk_length(torch.tensor(shifted)).numpy()
    np.testing.assert_allclose(got, distances, atol=1e-8, rtol=1e-9)


@pytest.mark.parametrize('dimension', [2, 3, 5])
def test_translation_preserves_distances_in_all_supported_dimensions(dimension):
    rng = np.random.default_rng(7)
    points = rng.normal(size=(16, dimension))
    points *= rng.uniform(.1, .98, (16, 1)) / np.linalg.norm(points, axis=1, keepdims=True)
    moved = place_point_at_origin(points, 6)
    np.testing.assert_array_equal(moved[6], np.zeros(dimension))
    np.testing.assert_allclose(pairwise_disk_length(points), pairwise_disk_length(moved), rtol=1e-10, atol=1e-10)


def test_valid_boundary_near_points_are_not_projected_by_geometry():
    radius = 1 - 1e-10
    points = np.array([[0., 0.], [radius, 0.]])
    expected = 2 * np.arctanh(radius)
    assert pairwise_disk_length(points)[0, 1] == pytest.approx(expected, rel=1e-9)
    assert length_from_origin(points)[1] == pytest.approx(expected, rel=1e-12)
    torch = pytest.importorskip('torch')
    assert float(_fitting._torch_disk_length(torch.tensor(points))[0, 1]) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize('point', [[1., 0.], [1.1, 0.], [np.nan, 0.]])
def test_geometry_rejects_invalid_points_instead_of_hiding_them(point):
    points = np.array([point, [0., 0.]])
    with pytest.raises(ValueError):
        pairwise_disk_length(points)
    with pytest.raises(ValueError):
        place_point_at_origin(points, 0)
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError):
        _fitting._torch_disk_length(torch.tensor(points))


def test_numpy_and_torch_fits_agree_on_small_problem(monkeypatch):
    pytest.importorskip('torch')
    rng = np.random.default_rng(42)
    target = rng.uniform(.1, 1., size=(5, 5))
    target /= target.sum(axis=1, keepdims=True)
    torch_fit = _fitting.fit_disk_embedding(target, n_epochs=3, random_state=6, max_grad_norm=.1)
    monkeypatch.setattr(_fitting, 'torch', None)
    numpy_fit = _fitting.fit_disk_embedding(target, n_epochs=3, random_state=6, max_grad_norm=.1)
    np.testing.assert_allclose(numpy_fit.coordinates, torch_fit.coordinates, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(numpy_fit.loss_trace, torch_fit.loss_trace, atol=1e-7, rtol=1e-6)
