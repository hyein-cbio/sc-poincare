from __future__ import annotations

import numpy as np

from sc_poincare._disk import (
    confine_to_open_disk,
    fan_dendrogram_angles,
    jitter_angles,
    length_from_origin,
    repel_points,
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


def test_zero_repel_is_exactly_the_embedding():
    rng = np.random.default_rng(5)
    pts = confine_to_open_disk(rng.normal(scale=0.3, size=(12, 2)))
    out = repel_points(pts, repel=0.0)
    assert out is not pts
    assert np.array_equal(out, pts)


def test_repel_separates_stacked_points_along_the_same_ray():
    pts = np.array([[0.0, 0.0], [0.4, 0.2], [0.4, 0.2], [0.4, 0.2]])
    out = repel_points(pts, repel=1.0)
    assert np.array_equal(out[0], [0.0, 0.0])
    original = pts[1]
    for row in out[1:]:
        assert abs(original[0] * row[1] - original[1] * row[0]) < 1e-9
        assert np.dot(original, row) > 0.0
    gaps = np.linalg.norm(out[1:, None, :] - out[None, 1:, :], axis=2)
    off = gaps[np.triu_indices(3, k=1)]
    assert np.min(off) > 0.02


def test_zero_jitter_is_exactly_the_embedding():
    rng = np.random.default_rng(6)
    pts = confine_to_open_disk(rng.normal(scale=0.3, size=(10, 2)))
    out = jitter_angles(pts, jitter=0.0)
    assert out is not pts
    assert np.array_equal(out, pts)


def test_jitter_changes_only_angle_and_stays_inside_the_window():
    pts = np.array([[0.0, 0.0], [0.6, 0.0], [0.0, 0.4]])
    out = jitter_angles(pts, jitter=0.1, random_state=1)
    assert np.array_equal(out[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(pts, axis=1))
    original = polar_angles(pts[1:])
    moved = polar_angles(out[1:])
    delta = np.abs(np.arctan2(np.sin(moved - original), np.cos(moved - original)))
    assert np.all(delta <= 0.1 + 1e-12)
    assert np.any(delta > 0.0)


def test_zero_spread_is_exactly_the_poincare_embedding():
    rng = np.random.default_rng(3)
    pts = confine_to_open_disk(rng.normal(scale=0.4, size=(30, 2)))
    groups = np.array(["a", "b", "c"] * 10)
    fanned = fan_dendrogram_angles(pts, groups, spread=0.0)
    assert fanned is not pts
    assert np.array_equal(fanned, pts)


def test_full_spread_keeps_radii_and_separates_overlapping_branches():
    rng = np.random.default_rng(4)
    branch_a = rng.normal(scale=0.02, size=(40, 2)) + np.array([0.55, 0.02])
    branch_b = rng.normal(scale=0.02, size=(40, 2)) + np.array([0.62, 0.0])
    pts = confine_to_open_disk(np.vstack([branch_a, branch_b]))
    groups = np.array(["a"] * 40 + ["b"] * 40)
    before = polar_angles(pts)
    fanned = fan_dendrogram_angles(pts, groups, spread=1.0)
    assert np.allclose(np.linalg.norm(fanned, axis=1), np.linalg.norm(pts, axis=1))
    after = polar_angles(fanned)
    before_gap = abs(before[:40].mean() - before[40:].mean())
    after_gap = abs(after[:40].mean() - after[40:].mean())
    after_gap = min(after_gap, 2 * np.pi - after_gap)
    before_gap = min(before_gap, 2 * np.pi - before_gap)
    assert after_gap > before_gap


def test_recenter_is_an_isometry_and_pins_the_root():
    rng = np.random.default_rng(2)
    pts = confine_to_open_disk(rng.normal(scale=0.25, size=(12, 2)))
    before = pairwise_disk_length(pts)
    recentered = place_point_at_origin(pts, 3)
    after = pairwise_disk_length(recentered)
    assert np.linalg.norm(recentered[3]) < 1e-10
    assert np.allclose(before, after, atol=1e-8)
