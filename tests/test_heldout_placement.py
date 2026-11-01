"""Held-out disk placement leaves the training coordinates fixed."""
import numpy as np
import pytest

from sc_poincare.evaluation import place_heldout_cells
from types import SimpleNamespace


def test_duplicate_cell_with_one_neighbor_stays_on_that_coordinate():
    training_features = np.array([
        [0., 0.],
        [1., 0.],
        [0., 2.],
        [3., -1.],
    ], dtype=np.float64)
    training_coordinates = np.array([
        [0.1, -0.2],
        [0.4, 0.1],
        [-0.3, 0.2],
        [0.0, 0.5],
    ], dtype=np.float64)
    frozen = training_coordinates.copy()
    training_coordinates.setflags(write=False)
    placed = place_heldout_cells(
        training_features, training_coordinates, training_features[[1, 3]], n_neighbors=1,
    )
    assert placed.shape == (2, 2)
    np.testing.assert_allclose(placed[0], frozen[1], atol=1e-6)
    np.testing.assert_allclose(placed[1], frozen[3], atol=1e-6)
    np.testing.assert_array_equal(training_coordinates, frozen)
    assert np.all(np.sum(placed * placed, axis=1) < 1.)


def test_heldout_placement_rejects_disk_and_shape_errors():
    features = np.eye(4, dtype=np.float64)
    coordinates = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="unit disk"):
        place_heldout_cells(features, np.ones((4, 2)), features[:1])
    with pytest.raises(ValueError, match="n_neighbors"):
        place_heldout_cells(features, coordinates, features[:1], n_neighbors=5)
    with pytest.raises(ValueError, match="at least one"):
        place_heldout_cells(features, coordinates, features[:0])
    with pytest.raises(ValueError, match="columns"):
        place_heldout_cells(features, coordinates, np.ones((1, 3)))


def test_heldout_placement_rejects_unfinished_optimization(monkeypatch):
    def unfinished(*args, **kwargs):
        return SimpleNamespace(x=np.array([.1, .2]), fun=1., success=False,
                               message='Iteration limit reached')

    monkeypatch.setattr('scipy.optimize.minimize', unfinished)
    with pytest.raises(RuntimeError, match='optimization failed.*Iteration limit'):
        place_heldout_cells(np.array([[0.]]), np.zeros((1, 2)), np.array([[1.]]), n_neighbors=1)


@pytest.mark.parametrize('point', [[np.sqrt(1. - 5e-9), 0.], [1., 0.], [np.nan, 0.]])
def test_heldout_placement_rejects_penalty_region_even_if_optimizer_reports_success(point, monkeypatch):
    monkeypatch.setattr('scipy.optimize.minimize', lambda *a, **k: SimpleNamespace(
        x=np.array(point), fun=1., success=True, message='converged'))
    with pytest.raises(FloatingPointError, match='disk'):
        place_heldout_cells(np.array([[0.]]), np.zeros((1, 2)), np.array([[1.]]), n_neighbors=1)


def test_heldout_placement_fits_nonduplicate_cells_without_moving_training():
    features = np.array([[0., 0.], [1., 0.], [0., 1.]])
    coordinates = np.array([[0., 0.], [.2, 0.], [0., .2]])
    frozen = coordinates.copy()
    placed = place_heldout_cells(features, coordinates, np.array([[.2, .3]]), n_neighbors=3)
    assert np.isfinite(placed).all()
    assert np.sum(placed * placed) < 1. - 1e-8
    np.testing.assert_array_equal(coordinates, frozen)


def test_iteration_limit_is_not_a_successful_heldout_result():
    rng = np.random.default_rng(37)
    features = rng.normal(size=(15, 4))
    coordinates = rng.normal(scale=.2, size=(15, 2))
    with pytest.raises(RuntimeError, match='optimization failed'):
        place_heldout_cells(features, coordinates, rng.normal(size=(2, 4)), n_neighbors=5, max_iter=1)


def test_heldout_placement_requires_training_cells():
    with pytest.raises(ValueError, match='training_features must contain'):
        place_heldout_cells(np.empty((0, 2)), np.empty((0, 2)), np.ones((1, 2)), n_neighbors=1)
