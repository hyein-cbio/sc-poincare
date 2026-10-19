"""Predeclared initialization and degree-aware probes. Defaults stay unchanged.

These helpers do not select a method from a score. ``fit_graph_stress`` remains
the classical-jitter exact path. A random disk start and a degree-aware teacher
are explicit alternate conditions.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from sc_poincare import experimental


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


def fit_from_initial(setup, initial, *, epochs, chunk_size=256, geometry="poincare",
                     max_dense_bytes, teacher=None, optimizer_name="adam"):
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
    torch = experimental._torch()
    config = setup.config
    if geometry == "poincare":
        tangent = experimental._tangent_from_disk(np.asarray(initial, dtype=np.float64))
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
        loss, _, _ = experimental._objective(
            u, target, teacher_t, config, geometry, chunk_size, backward=epoch < epochs,
        )
        current = (experimental._disk_from_tangent(u) if geometry == "poincare" else u).detach().numpy().copy()
        trace.append(float(loss))
        if loss < best_loss:
            best_loss, best_epoch = float(loss), epoch
        if epoch < epochs:
            optimizer.step()
    return dict(loss_trace=trace, current=current, best_loss=best_loss, best_epoch=best_epoch,
                epochs=epochs, optimizer_name=optimizer_name)


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


__all__ = [
    "classical_jitter",
    "degree_aware_teacher",
    "fit_from_initial",
    "fixed_multiscale_accessibility",
    "promotion_allowed",
    "random_disk",
]
