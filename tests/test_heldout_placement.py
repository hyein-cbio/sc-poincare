"""Held-out disk placement leaves the training coordinates fixed."""
import numpy as np
import pytest

from sc_poincare.evaluation import place_heldout_cells


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
