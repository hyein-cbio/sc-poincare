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


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(theta: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    return float(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()))


def repel_points(
    coordinates: np.ndarray,
    repel: float = 0.0,
    *,
    iterations: int = 30,
    max_gap: float = 0.06,
) -> np.ndarray:
    """Separate overlapping display points by radius only.

    ``repel=0`` returns an exact copy. Larger values set a Euclidean gap of
    ``repel * max_gap``. Pairs closer than that gap are pushed apart along
    their original rays. Polar angle is not changed. A point that starts at
    the origin stays there. This is a layout, not a hyperbolic distance.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates must have at least two columns.")
    repel = float(repel)
    if not 0.0 <= repel <= 1.0:
        raise ValueError("repel must be between 0 and 1.")
    if repel == 0.0:
        return np.array(coordinates, copy=True)
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if max_gap <= 0.0:
        raise ValueError("max_gap must be positive.")

    from scipy.spatial import cKDTree

    points = np.array(coordinates, copy=True)
    radius = np.linalg.norm(points[:, :2], axis=1)
    origin = radius == 0.0
    unit = np.zeros_like(points[:, :2])
    unit[~origin] = points[~origin, :2] / radius[~origin, None]
    gap = repel * max_gap
    n_obs = points.shape[0]
    for _ in range(iterations):
        pairs = cKDTree(points[:, :2]).query_pairs(gap)
        if not pairs:
            break
        radial = np.zeros(n_obs, dtype=np.float64)
        hits = np.zeros(n_obs, dtype=np.float64)
        for i, j in pairs:
            if origin[i] and origin[j]:
                continue
            dist = float(np.linalg.norm(points[i, :2] - points[j, :2]))
            push = 0.5 * (gap - dist)
            outer, inner = (i, j) if radius[i] > radius[j] or (radius[i] == radius[j] and i > j) else (j, i)
            if origin[outer]:
                outer, inner = inner, outer
            if origin[inner]:
                radial[outer] += 2.0 * push
            else:
                radial[outer] += push
                radial[inner] -= push
            hits[i] += 1.0
            hits[j] += 1.0
        radius = radius + radial / np.clip(hits, 1.0, None)
        radius = np.clip(radius, 0.0, 1.0 - 1e-5)
        radius[origin] = 0.0
        points[:, :2] = unit * radius[:, None]
    return points


def jitter_angles(
    coordinates: np.ndarray,
    jitter: float = 0.0,
    *,
    random_state: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Add uniform angular noise without changing radius.

    ``jitter=0`` returns an exact copy. Otherwise each nonzero point's polar
    angle is shifted by an independent ``Uniform(-jitter, jitter)`` draw, in
    radians. Radius is unchanged, and a point at the origin stays there.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates must have at least two columns.")
    jitter = float(jitter)
    if jitter < 0.0 or jitter > np.pi:
        raise ValueError("jitter must be between 0 and pi radians.")
    if jitter == 0.0:
        return np.array(coordinates, copy=True)
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    points = np.array(coordinates, copy=True)
    radius = np.linalg.norm(points[:, :2], axis=1)
    origin = radius == 0.0
    if np.any(~origin):
        theta = np.arctan2(points[~origin, 1], points[~origin, 0])
        theta = theta + rng.uniform(-jitter, jitter, size=int(np.sum(~origin)))
        points[~origin, 0] = radius[~origin] * np.cos(theta)
        points[~origin, 1] = radius[~origin] * np.sin(theta)
    return points


def fan_dendrogram_angles(
    coordinates: np.ndarray,
    groups: np.ndarray,
    spread: float = 0.0,
) -> np.ndarray:
    """Spread branch angles while keeping each cell's radius.

    ``spread=0`` is the identity: the returned coordinates are an exact copy of
    the Poincaré embedding. ``spread=1`` places dendrogram leaves at equal
    angles, rotating each group rigidly about the origin. Intermediate values
    interpolate along the shortest arc. Pairwise hyperbolic distances are
    unchanged only at ``spread=0``; distance from the origin is unchanged for
    every value, because the radius is not modified.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates must have at least two columns.")
    groups = np.asarray(groups)
    if groups.shape[0] != coordinates.shape[0]:
        raise ValueError("groups must have one label per point.")
    spread = float(spread)
    if not 0.0 <= spread <= 1.0:
        raise ValueError("spread must be between 0 and 1.")
    if spread == 0.0:
        return np.array(coordinates, copy=True)

    labels = [label for label in dict.fromkeys(groups.tolist())]
    usable = []
    means = []
    for label in labels:
        mask = groups == label
        radii = np.linalg.norm(coordinates[mask, :2], axis=1)
        angled = radii > 0.0
        if not np.any(angled):
            continue
        usable.append(label)
        means.append(_circular_mean(polar_angles(coordinates[mask][angled])))
    if len(usable) < 2:
        return np.array(coordinates, copy=True)

    from scipy.cluster.hierarchy import leaves_list, linkage

    mean_angle = np.asarray(means, dtype=np.float64)
    circle = np.column_stack([np.cos(mean_angle), np.sin(mean_angle)])
    order = leaves_list(linkage(circle, method="average"))
    ordered_labels = [usable[int(i)] for i in order]
    ordered_angles = mean_angle[order]
    anchor = _circular_mean(ordered_angles)
    n_groups = len(ordered_labels)
    target = anchor + (np.arange(n_groups) - (n_groups - 1) / 2.0) * (2.0 * np.pi / n_groups)
    target = _wrap_angle(target)

    fanned = np.array(coordinates, copy=True)
    for label, current, dest in zip(ordered_labels, ordered_angles, target):
        phi = spread * float(_wrap_angle(dest - current))
        if phi == 0.0:
            continue
        mask = groups == label
        cosine, sine = np.cos(phi), np.sin(phi)
        xy = fanned[mask, :2]
        rotated = np.column_stack([cosine * xy[:, 0] - sine * xy[:, 1], sine * xy[:, 0] + cosine * xy[:, 1]])
        fanned[mask, :2] = rotated
    return fanned
