"""Deprecated display-only jitter/fan/repel layouts, isolated for future deletion.

Nothing in fitting, initialization or core geometry imports this module.
Only the three legacy tl exports use these wrappers. No warning is emitted on
import; each API call warns, including a zero-strength identity call.
GraphStressConfig.jitter is initialization noise and is not deprecated.
"""

from __future__ import annotations

import warnings
import numpy as np

from sc_poincare._disk import polar_angles

DEFAULT_BASIS = "X_poincare"


class DisplayLayoutDeprecationWarning(FutureWarning):
    """Visible warning for display layouts scheduled for separate removal."""


def _warn(api):
    warnings.warn(
        f"{api} is deprecated: it changes display coordinates, not the learned "
        "embedding. Keep raw coordinates and adjust marker size/alpha or zoom. "
        "Initialization jitter is unaffected.",
        DisplayLayoutDeprecationWarning,
        stacklevel=3,
    )


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(theta: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    return float(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()))


def _jitter_angles(
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


def _fan_dendrogram_angles(
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


def _repel_points(
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


def repel_embedding(
    adata,
    *,
    repel: float = 0.0,
    basis: str = DEFAULT_BASIS,
    key_added: str = "X_poincare_repel",
    copy: bool = False,
):
    """Deprecated: separate display overlaps along each ray.

    ``repel=0`` copies ``adata.obsm[basis]`` exactly. Positive values change
    radius only. Polar angle is left unchanged.
    """
    _warn('poincare_repel')
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    adata.obsm[key_added] = _repel_points(np.asarray(adata.obsm[basis]), repel=repel)
    return adata if copy else None


def jitter_embedding(
    adata,
    *,
    jitter: float = 0.0,
    basis: str = DEFAULT_BASIS,
    key_added: str = "X_poincare_jitter",
    random_state: int | np.random.Generator | None = 0,
    copy: bool = False,
):
    """Deprecated: write a display-only layout with angular jitter.

    ``jitter=0`` copies ``adata.obsm[basis]`` exactly. Positive values are
    the half-width, in radians, of a uniform angle shift. Radius is unchanged.
    """
    _warn('poincare_jitter')
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    adata.obsm[key_added] = _jitter_angles(
        np.asarray(adata.obsm[basis]),
        jitter=jitter,
        random_state=random_state,
    )
    return adata if copy else None


def fan_branches(
    adata,
    groupby: str,
    *,
    spread: float = 0.0,
    basis: str = DEFAULT_BASIS,
    key_added: str = "X_poincare_fan",
    copy: bool = False,
):
    """Deprecated: fan label groups in a display-only angular layout.

    ``spread=0`` copies ``adata.obsm[basis]`` exactly. Larger values rotate
    each group about the origin so dendrogram leaves occupy a wider fan.
    Radii, and therefore hyperbolic distance from the origin, stay put.
    """
    _warn('poincare_fan')
    if copy:
        adata = adata.copy()
    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} is missing. Run sc_poincare.tl.poincare_map first.")
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} is not a column of adata.obs.")
    coordinates = np.asarray(adata.obsm[basis])
    groups = adata.obs[groupby].astype(str).to_numpy()
    adata.obsm[key_added] = _fan_dendrogram_angles(coordinates, groups, spread=spread)
    return adata if copy else None
