from __future__ import annotations

import numpy as np

from sc_poincare._disk import (
    confine_to_open_disk,
    length_from_origin,
    pairwise_disk_length,
    place_point_at_origin,
    polar_angles,
    translate_origin_to,
)


def test_distance_to_origin_matches_artanh():
    radii = np.array([0.0, 0.2, 0.5, 0.8, 0.95])
    points = np.stack([radii, np.zeros_like(radii)], axis=1)
    got = length_from_origin(points)
    expected = 2.0 * np.arctanh(radii)
    assert np.allclose(got, expected, atol=1e-8)


def test_pairwise_length_is_symmetric_and_zero_on_diagonal():
    rng = np.random.default_rng(1)
    pts = confine_to_open_disk(rng.normal(scale=0.2, size=(7, 2)))
    dist = pairwise_disk_length(pts)
    assert dist.shape == (7, 7)
    assert np.allclose(dist, dist.T, atol=1e-10)
    assert np.allclose(np.diag(dist), 0.0, atol=1e-10)


def test_translation_sends_origin_to_the_requested_point():
    v = np.array([0.3, -0.2])
    origin = np.zeros((1, 2))
    moved = translate_origin_to(origin, v)
    assert np.allclose(moved[0], v, atol=1e-10)


def test_recenter_is_an_isometry_and_pins_the_root():
    rng = np.random.default_rng(2)
    pts = confine_to_open_disk(rng.normal(scale=0.25, size=(12, 2)))
    before = pairwise_disk_length(pts)
    recentered = place_point_at_origin(pts, 3)
    after = pairwise_disk_length(recentered)
    assert np.linalg.norm(recentered[3]) < 1e-10
    assert np.allclose(before, after, atol=1e-8)
