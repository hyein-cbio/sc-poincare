"""Predeclared initialization and degree-aware probes. Defaults stay unchanged.

These helpers do not select a method from a score. ``fit_graph_stress`` remains
the classical-jitter exact path. A random disk start and a degree-aware teacher
are explicit alternate conditions.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from sc_poincare import experimental


def classical_ball(setup, dimension, random_state):
    """Classical scaling into ``dimension`` coordinates, then the usual jitter.

    Dimension 2 follows the same Gram construction as exact MU. A higher
    dimension is an explicit probe, not a default. The maximum radius is the
    configuration's ``initial_radius``.
    """
    dimension = int(dimension)
    n = setup.base_initial.shape[0]
    if isinstance(dimension, bool) or dimension < 2 or dimension >= n:
        raise ValueError("dimension must be an integer >= 2 and smaller than the number of cells.")
    config = setup.config
    geodesic = np.array(setup.target, dtype=np.float64, copy=True)
    geodesic *= setup.metadata["global_mean"] / config.distance_scale
    squared = geodesic * geodesic
    gram = -.5 * (squared - squared.mean(0) - squared.mean(1, keepdims=True) + squared.mean())
    from scipy.linalg import eigh
    values, vectors = eigh(gram, subset_by_index=[n - dimension, n - 1])
    initial = vectors[:, ::-1] * np.sqrt(np.maximum(values[::-1], 0.))
    radius = float(np.linalg.norm(initial, axis=1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Classical scaling did not produce a nondegenerate layout.")
    initial *= config.initial_radius / radius
    noise = np.random.default_rng(random_state).normal(scale=config.jitter, size=initial.shape)
    return np.ascontiguousarray(initial + noise, dtype=np.float64)


def classical_jitter(setup, random_state):
    """The exact MU start: classical scaling plus the configuration's jitter."""
    config = setup.config
    noise = np.random.default_rng(random_state).normal(scale=config.jitter, size=setup.base_initial.shape)
    return np.ascontiguousarray(setup.base_initial + noise, dtype=np.float64)


def random_disk(n_cells, radius, random_state):
    """Uniform-area disk sample, independent of the classical layout.

    The radius is strictly below ``radius`` and ``radius`` must be below 1.
    """
    if not np.isfinite(radius) or radius <= 0 or radius >= 1:
        raise ValueError("radius must be finite and strictly inside (0, 1).")
    n_cells = int(n_cells)
    if n_cells < 2:
        raise ValueError("At least two cells are required.")
    rng = np.random.default_rng(random_state)
    direction = rng.normal(size=(n_cells, 2))
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise RuntimeError("A random direction had zero length.")
    rho = radius * np.sqrt(rng.random(n_cells))
    return np.ascontiguousarray(direction / norm * rho[:, None], dtype=np.float64)


def degree_aware_teacher(features, config):
    """Fixed union-graph rule ``sigma_ij = sigma * sqrt(deg_i deg_j) / mean(deg)``.

    ``sigma`` is the existing global bandwidth. This does not replace
    ``prepare_graph_stress`` and is not a default.
    """
    if not isinstance(config, experimental.GraphStressConfig):
        raise TypeError("config must be a GraphStressConfig.")
    features = np.ascontiguousarray(features, dtype=np.float64)
    local = experimental._graph_lengths(features, config.n_neighbors, "union")
    positive = local.data[local.data > 0]
    if not len(positive) or not np.isfinite(positive).all():
        raise ValueError("A nondegenerate union graph is required.")
    sigma = max(float(np.quantile(positive, .95)), float(positive.max() / 5.))
    graph = local.tocoo()
    degree = np.bincount(graph.row, minlength=len(features)).astype(np.float64)
    if np.any(degree <= 0):
        raise ValueError("Degree-aware bandwidth requires a positive degree on every cell.")
    sigma_ij = sigma * np.sqrt(degree[graph.row] * degree[graph.col]) / float(degree.mean())
    weights = sparse.csr_matrix((np.exp(-.5 * (graph.data / sigma_ij) ** 2), (graph.row, graph.col)), shape=local.shape)
    weights.eliminate_zeros()
    row_sum = np.asarray(weights.sum(axis=1)).ravel()
    if np.any(row_sum <= 0) or not np.isfinite(row_sum).all():
        raise FloatingPointError("Degree-aware teacher has an empty or nonfinite row.")
    teacher = (sparse.diags(1. / row_sum) @ weights).tocsr()
    return teacher, dict(local_sigma=sigma, degree_mean=float(degree.mean()), rule="sigma_sqrt_degree_product")


def promotion_allowed(*, development_input, joint_improvements):
    """Development inputs cannot promote a rule, even if every measured score rose."""
    if development_input:
        return False
    return bool(joint_improvements) and all(joint_improvements)


def _validate_curvature(curvature):
    if isinstance(curvature, (bool, np.bool_)) or not isinstance(curvature, (int, float, np.floating)):
        raise ValueError("curvature must be finite and positive.")
    curvature = float(curvature)
    if not np.isfinite(curvature) or curvature <= 0:
        raise ValueError("curvature must be finite and positive.")
    return curvature


def _tangent_from_disk_curvature(points, curvature):
    radius = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(curvature * radius ** 2 >= 1):
        raise ValueError("Initialization must lie inside the curvature ball.")
    scale = np.ones_like(radius)
    positive = radius > 0
    scale[positive] = 2 * np.arctanh(np.sqrt(curvature) * radius[positive]) / (np.sqrt(curvature) * radius[positive])
    return points * scale


def _objective_curvature(u, target, teacher, config, chunk_size, curvature, *, backward):
    """MU objective with Poincaré curvature ``-curvature``. Curvature 1 matches the default disk."""
    torch = experimental._torch()
    curvature_t = torch.tensor(curvature, dtype=torch.float64)
    n = len(u)
    u.grad = None
    global_total, local_total = 0., 0.
    with torch.set_grad_enabled(backward):
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            rows = torch.arange(start, stop)
            radius = torch.linalg.vector_norm(u, dim=1, keepdim=True)
            factor = torch.where(
                radius > 1e-7,
                torch.tanh(torch.sqrt(curvature_t) * radius / 2) / (torch.sqrt(curvature_t) * radius.clamp_min(1e-15)),
                0.5 - curvature_t * radius.square() / 24,
            )
            x = u * factor
            sq = (x * x).sum(dim=1)
            if not torch.isfinite(x).all() or torch.any(curvature_t * sq >= 1):
                raise FloatingPointError("Curvature ball was left during the probe.")
            separation = torch.cdist(x[start:stop], x, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            denom = (1 - curvature_t * sq[start:stop])[:, None] * (1 - curvature_t * sq)[None, :]
            distance = 2 / torch.sqrt(curvature_t) * torch.asinh(
                torch.sqrt(curvature_t) * separation / torch.sqrt(denom)
            )
            distance = distance.clone()
            distance[torch.arange(stop - start), rows] = 0
            t = target[start:stop]
            error = ((distance - t) / config.distance_scale).square() / (
                t / config.distance_scale).clamp_min(config.relative_floor).square()
            global_loss = error.sum() / (n * (n - 1))
            mask = torch.zeros_like(distance, dtype=torch.bool)
            mask[torch.arange(stop - start), rows] = True
            logits = (-distance / config.gamma).masked_fill(mask, -torch.inf)
            logq = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).masked_fill(mask, 0)
            local_loss = -(teacher[start:stop] * logq).sum() / n
            loss = global_loss + config.local_weight * local_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite curvature-probe objective.")
            global_total += float(global_loss.detach())
            local_total += float(local_loss.detach())
            if backward:
                loss.backward()
    return global_total + config.local_weight * local_total, global_total, local_total


def fit_from_initial(setup, initial, *, epochs, chunk_size=256, geometry="poincare",
                     max_dense_bytes, teacher=None, optimizer_name="adam", curvature=1.):
    """Adam or plain SGD on an exact setup from a caller-supplied start.

    ``optimizer_name='adam'`` is the existing tangent Adam update. ``'sgd'``
    uses the same learning rate and is a separate ablation, not a new default.
    ``teacher=None`` uses the prepared teacher. A replacement teacher must be
    row-stochastic, finite, and shaped N-by-N. The prepared target is unchanged.
    No jitter is added here.
    """
    experimental._validate_setup(setup, max_dense_bytes)
    epochs = experimental._integer("epochs", epochs, 0)
    chunk_size = experimental._integer("chunk_size", chunk_size)
    if geometry not in ("poincare", "euclidean"):
        raise ValueError("geometry must be 'poincare' or 'euclidean'.")
    if optimizer_name not in ("adam", "sgd"):
        raise ValueError("optimizer_name must be 'adam' or 'sgd'.")
    curvature = _validate_curvature(curvature)
    if geometry != "poincare" and curvature != 1.:
        raise ValueError("Curvature probes use the Poincaré ball.")
    torch = experimental._torch()
    config = setup.config
    if geometry == "poincare":
        points = np.asarray(initial, dtype=np.float64)
        tangent = (experimental._tangent_from_disk(points) if curvature == 1.
                   else _tangent_from_disk_curvature(points, curvature))
    else:
        tangent = np.asarray(initial, dtype=np.float64)
        if tangent.shape != setup.base_initial.shape or not np.isfinite(tangent).all():
            raise ValueError("Euclidean initialization must be finite and shaped (N, 2).")
    if teacher is None:
        teacher_matrix = np.array(setup.teacher, dtype=np.float64, copy=True)
    else:
        teacher_matrix = np.array(teacher.toarray() if sparse.issparse(teacher) else teacher, dtype=np.float64, copy=True)
        if teacher_matrix.shape != setup.teacher.shape or not np.isfinite(teacher_matrix).all():
            raise ValueError("Replacement teacher must be finite and match the prepared shape.")
        if not np.allclose(teacher_matrix.sum(1), 1., rtol=0., atol=1e-8):
            raise ValueError("Replacement teacher rows must sum to 1.")
    u = torch.tensor(tangent, dtype=torch.float64, requires_grad=True)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([u], lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD([u], lr=config.learning_rate)
    target = torch.tensor(np.array(setup.target, dtype=np.float64, copy=True), dtype=torch.float64)
    teacher_t = torch.tensor(teacher_matrix, dtype=torch.float64)
    best_loss, best_epoch, trace = np.inf, 0, []
    current = None
    for epoch in range(epochs + 1):
        if curvature == 1.:
            loss, _, _ = experimental._objective(
                u, target, teacher_t, config, geometry, chunk_size, backward=epoch < epochs,
            )
            current = (experimental._disk_from_tangent(u) if geometry == "poincare" else u).detach().numpy().copy()
        else:
            loss, _, _ = _objective_curvature(
                u, target, teacher_t, config, chunk_size, curvature, backward=epoch < epochs,
            )
            radius = torch.linalg.vector_norm(u, dim=1, keepdim=True)
            factor = torch.where(
                radius > 1e-7,
                torch.tanh(torch.sqrt(torch.tensor(curvature)) * radius / 2) / (
                    torch.sqrt(torch.tensor(curvature)) * radius.clamp_min(1e-15)),
                0.5 - curvature * radius.square() / 24,
            )
            current = (u * factor).detach().numpy().copy()
        trace.append(float(loss))
        if loss < best_loss:
            best_loss, best_epoch = float(loss), epoch
        if epoch < epochs:
            optimizer.step()
    return dict(loss_trace=trace, current=current, best_loss=best_loss, best_epoch=best_epoch,
                epochs=epochs, optimizer_name=optimizer_name, curvature=curvature)


def fixed_multiscale_accessibility(affinity, *, max_dense_bytes):
    """Equal mixture of forest times 0.1 and 10, then row-normalized.

    This is not ``forest_tau=1`` and it is not selected as a default. Both
    components are built by the existing solver.
    """
    from sc_poincare._proximity import forest_accessibility
    parts = [forest_accessibility(affinity, max_dense_bytes=max_dense_bytes, forest_tau=tau)
             for tau in (0.1, 10.)]
    mixed = 0.5 * (parts[0] + parts[1])
    mixed = np.maximum(mixed, 0.)
    mixed /= np.clip(mixed.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    if not np.isfinite(mixed).all():
        raise FloatingPointError("Multiscale mixture is nonfinite.")
    return mixed


def lorentz_from_disk(points):
    """Isometric map from the Poincaré disk onto the forward hyperboloid."""
    points = np.asarray(points, dtype=np.float64)
    radius_sq = np.sum(points * points, axis=1, keepdims=True)
    if np.any(radius_sq >= 1):
        raise ValueError("Lorentz coordinates require an open-disk point.")
    denom = 1. - radius_sq
    time = (1. + radius_sq) / denom
    space = 2. * points / denom
    return np.concatenate([time, space], axis=1)


def disk_from_lorentz(points):
    """Inverse of ``lorentz_from_disk`` for the forward hyperboloid."""
    points = np.asarray(points, dtype=np.float64)
    return points[:, 1:] / (points[:, :1] + 1.)


def _minkowski(left, right):
    return -left[..., 0] * right[..., 0] + (left[..., 1:] * right[..., 1:]).sum(dim=-1)


def fit_lorentz(setup, initial, *, epochs, chunk_size=256, max_dense_bytes, learning_rate=None):
    """Stress on the hyperboloid chart, starting from the disk initialization.

    The step is a Minkowski retraction, not tangent Adam. Curvature stays -1.
    This does not replace the disk default.
    """
    experimental._validate_setup(setup, max_dense_bytes)
    epochs = experimental._integer("epochs", epochs, 0)
    chunk_size = experimental._integer("chunk_size", chunk_size)
    torch = experimental._torch()
    config = setup.config
    rate = config.learning_rate if learning_rate is None else float(learning_rate)
    point = torch.tensor(lorentz_from_disk(initial), dtype=torch.float64, requires_grad=True)
    target = torch.tensor(np.array(setup.target, dtype=np.float64, copy=True), dtype=torch.float64)
    teacher = torch.tensor(np.array(setup.teacher, dtype=np.float64, copy=True), dtype=torch.float64)

    def loss_at(coordinates):
        n = len(coordinates)
        total = 0.
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            rows = torch.arange(start, stop)
            left, right = coordinates[start:stop], coordinates
            inner = -left[:, :1] * right[:, 0] + left[:, 1:] @ right[:, 1:].T
            distance = torch.acosh(torch.clamp(-inner, min=1.))
            distance = distance.clone()
            distance[torch.arange(stop - start), rows] = 0
            t = target[start:stop]
            error = ((distance - t) / config.distance_scale).square() / (
                t / config.distance_scale).clamp_min(config.relative_floor).square()
            global_loss = error.sum() / (n * (n - 1))
            mask = torch.zeros_like(distance, dtype=torch.bool)
            mask[torch.arange(stop - start), rows] = True
            logits = (-distance / config.gamma).masked_fill(mask, -torch.inf)
            logq = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).masked_fill(mask, 0)
            local_loss = -(teacher[start:stop] * logq).sum() / n
            total = total + global_loss + config.local_weight * local_loss
        return total

    trace = []
    for epoch in range(epochs + 1):
        if point.grad is not None:
            point.grad = None
        value = loss_at(point)
        trace.append(float(value.detach()))
        if epoch == epochs:
            break
        value.backward()
        grad = point.grad
        inner = _minkowski(point.detach(), grad)
        tangent = grad + inner[:, None] * point.detach()
        step = -rate * tangent
        norm = torch.sqrt(_minkowski(step, step).clamp_min(1e-24))
        updated = torch.cosh(norm)[:, None] * point.detach() + torch.sinh(norm)[:, None] * step / norm[:, None]
        scale = torch.sqrt(torch.clamp(-_minkowski(updated, updated), min=1e-24))
        updated = updated / scale[:, None]
        updated = torch.where(updated[:, :1] < 0, -updated, updated)
        point = updated.detach().requires_grad_(True)
    current = point.detach()
    return dict(loss_trace=trace, current=current.numpy(), minkowski_diag=_minkowski(current, current).numpy(), epochs=epochs)


__all__ = [
    "classical_ball",
    "classical_jitter",
    "degree_aware_teacher",
    "disk_from_lorentz",
    "fit_from_initial",
    "fit_lorentz",
    "fixed_multiscale_accessibility",
    "lorentz_from_disk",
    "promotion_allowed",
    "random_disk",
]
