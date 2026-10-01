"""Poincaré-disk geometry used by the embedding and downstream tools."""

from __future__ import annotations

import numpy as np

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


def _alpha_from_sq(sep_sq: np.ndarray, left_sq: np.ndarray, right_sq: np.ndarray) -> np.ndarray:
    denom = np.clip(1.0 - left_sq[:, None], a_min=1e-12, a_max=None) * np.clip(
        1.0 - right_sq[None, :], a_min=1e-12, a_max=None
    )
    return np.clip(sep_sq / denom, 0.0, None)


def pairwise_disk_length(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Geodesic length on the Poincaré disk.

    Equivalent forms:
    ``d(u, v) = arcosh(1 + 2 ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))``
    ``d(u, v) = 2 arcsinh(sqrt(α))`` with ``α = ||u-v||^2 / ((1-||u||^2)(1-||v||^2))``.
    The ``arcsinh`` form is used because it is exact at zero distance.
    """
    left = confine_to_open_disk(np.asarray(left, dtype=np.float64))
    square = right is None
    if square:
        right = left
    else:
        right = confine_to_open_disk(np.asarray(right, dtype=np.float64))

    left_sq = np.sum(left * left, axis=-1)
    right_sq = np.sum(right * right, axis=-1)
    gram = left @ right.T
    sep_sq = np.clip(left_sq[:, None] + right_sq[None, :] - 2.0 * gram, 0.0, None)
    dist = 2.0 * np.arcsinh(np.sqrt(_alpha_from_sq(sep_sq, left_sq, right_sq)))
    if square:
        np.fill_diagonal(dist, 0.0)
    return dist


def length_from_origin(points: np.ndarray) -> np.ndarray:
    """Hyperbolic distance from the disk origin to each point."""
    points = confine_to_open_disk(np.asarray(points, dtype=np.float64))
    radii = np.linalg.norm(points, axis=-1)
    # 2 artanh(r) == 2 arcsinh(r / sqrt(1-r^2))
    return 2.0 * np.arctanh(np.clip(radii, 0.0, 1.0 - 1e-12))


def translate_origin_to(points: np.ndarray, new_origin: np.ndarray) -> np.ndarray:
    """Isometry sending the Euclidean origin to ``new_origin``.

    Paper Methods, translation in the Poincaré disk:

    ``τ(x, v) = [(1 + 2⟨v,x⟩ + ||x||²) v + (1 - ||v||²) x]
                 / [1 + 2⟨v,x⟩ + ||v||² ||x||²]``
    """
    points = np.asarray(points, dtype=np.float64)
    v = confine_to_open_disk(np.asarray(new_origin, dtype=np.float64).reshape(1, -1))[0]
    x = confine_to_open_disk(points)
    v_sq = float(np.dot(v, v))
    x_sq = np.sum(x * x, axis=-1, keepdims=True)
    inner = (x @ v.reshape(-1, 1))
    numerator = (1.0 + 2.0 * inner + x_sq) * v + (1.0 - v_sq) * x
    denominator = 1.0 + 2.0 * inner + v_sq * x_sq
    return confine_to_open_disk(numerator / np.clip(denominator, a_min=1e-12, a_max=None))


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
