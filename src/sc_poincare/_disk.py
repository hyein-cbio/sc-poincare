"""Poincaré-disk geometry used by the embedding and downstream tools."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

_BOUNDARY_MARGIN = 1e-5


def confine_to_open_disk(points: np.ndarray, margin: float = _BOUNDARY_MARGIN) -> np.ndarray:
    """Project points onto a slightly shrunken closed disk so they stay in the open ball."""
    points = np.asarray(points, dtype=np.float64)
    radii = np.linalg.norm(points, axis=-1, keepdims=True)
    limit = 1.0 - margin
    scale = np.ones_like(radii)
    too_far = radii > limit
    scale[too_far] = limit / np.clip(radii[too_far], a_min=np.finfo(np.float64).tiny, a_max=None)
    return points * scale


def _checked_disk_points(points: np.ndarray) -> np.ndarray:
    """Validate geometry inputs without applying the optimizer's radius margin."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("Disk coordinates must be finite nonempty vectors.")
    if np.any(np.sum(points * points, axis=-1) >= 1.0):
        raise ValueError("Disk coordinates must lie strictly inside the unit ball.")
    return points


def pairwise_disk_length(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Geodesic length on the Poincaré disk.

    Equivalent forms:
    ``d(u, v) = arcosh(1 + 2 ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))``
    ``d(u, v) = 2 arcsinh(sqrt(α))`` with ``α = ||u-v||^2 / ((1-||u||^2)(1-||v||^2))``.
    The ``arcsinh`` form is used because it is exact at zero distance.
    """
    left = _checked_disk_points(left)
    square = right is None
    right = left if square else _checked_disk_points(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Pairwise coordinates must be matrices with matching dimensions.")

    left_sq = np.sum(left * left, axis=-1)
    right_sq = np.sum(right * right, axis=-1)
    # Direct differences avoid cancellation of nearly equal squared norms.
    # cdist needs only a pair matrix, not an N×M×dimension temporary.
    separation = cdist(left, right, metric="euclidean")
    denom = np.sqrt((1.0 - left_sq[:, None]) * (1.0 - right_sq[None, :]))
    dist = 2.0 * np.arcsinh(separation / denom)
    if square:
        np.fill_diagonal(dist, 0.0)
    return dist


def length_from_origin(points: np.ndarray) -> np.ndarray:
    """Hyperbolic distance from the disk origin to each point."""
    points = _checked_disk_points(points)
    radii = np.linalg.norm(points, axis=-1)
    # Valid translated points may be closer to the boundary than the optimizer
    # margin. Clipping them would change pseudotime and break isometry.
    return 2.0 * np.arctanh(radii)


def translate_origin_to(points: np.ndarray, new_origin: np.ndarray) -> np.ndarray:
    """Isometry sending the Euclidean origin to ``new_origin``.

    Paper Methods, translation in the Poincaré disk:

    ``τ(x, v) = [(1 + 2⟨v,x⟩ + ||x||²) v + (1 - ||v||²) x]
                 / [1 + 2⟨v,x⟩ + ||v||² ||x||²]``
    """
    x = _checked_disk_points(points)
    v = _checked_disk_points(np.asarray(new_origin, dtype=np.float64).reshape(-1))
    if x.ndim != 2 or x.shape[1] != len(v):
        raise ValueError("Translation requires a matrix and a matching origin vector.")
    v_sq = float(np.dot(v, v))
    if v_sq == 0.0:
        return x.copy()
    # Algebraically equivalent to the formula above, expanded around x=-v.
    # The root then maps to exact zero without subtracting large numerators.
    delta = x + v
    delta_sq = np.sum(delta * delta, axis=-1, keepdims=True)
    inner = delta @ v.reshape(-1, 1)
    one_minus = 1.0 - v_sq
    numerator = one_minus * delta + delta_sq * v
    perpendicular = delta - (inner / v_sq) * v
    denominator = (one_minus + inner)**2 + v_sq * np.sum(perpendicular**2, axis=-1, keepdims=True)
    translated = numerator / denominator
    if not np.all(np.isfinite(translated)) or np.any(np.sum(translated**2, axis=-1) >= 1.0):
        raise FloatingPointError(
            "Disk translation exceeded floating-point boundary precision. "
            "Use unshifted coordinates; silently clipping would change hyperbolic distances."
        )
    return translated


def place_point_at_origin(points: np.ndarray, index: int) -> np.ndarray:
    """Rigidly move the selected point to the origin, preserving pairwise lengths."""
    points = np.asarray(points, dtype=np.float64)
    return translate_origin_to(points, -points[index])


def polar_angles(points: np.ndarray) -> np.ndarray:
    """Argument of each 2-D disk coordinate, in ``(-π, π]``."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[1] < 2:
        raise ValueError("Angular lineages require at least a 2-D disk embedding.")
    return np.arctan2(points[:, 1], points[:, 0])


def circular_gap(theta: np.ndarray) -> np.ndarray:
    """Pairwise circular distance on the unit circle."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    delta = np.abs(theta[:, None] - theta[None, :])
    return np.minimum(delta, 2.0 * np.pi - delta)
