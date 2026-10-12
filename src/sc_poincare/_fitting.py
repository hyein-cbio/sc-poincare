"""Riemannian fitting of disk coordinates to a target proximity matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from sc_poincare._disk import confine_to_open_disk, pairwise_disk_length
from sc_poincare._objectives import (
    _numpy_alignment_loss, _numpy_kl_from_logs, _numpy_log_probabilities,
    _torch_alignment_loss, _torch_kl_from_logs,
    DEFAULT_OBJECTIVE, CONDITIONAL_OBJECTIVE, _validate_objective,
)
from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES, check_dense_budget

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-free envs
    torch = None


@dataclass
class FitReport:
    coordinates: np.ndarray
    loss_trace: list[float]
    n_epochs: int
    backend: str
    batch_size: int
    best_loss: float
    stop_reason: str
    initialization: str
    initialization_reason: str
    objective: str = DEFAULT_OBJECTIVE
    probability_computation: str = "log_softmax"
    zero_target_policy: str = "machine_tiny_logs_only"


@dataclass(frozen=True)
class FitProgress:
    """Independent checkpoint copies at an evaluated epoch (including epoch 0)."""

    epoch: int
    coordinates: np.ndarray
    loss: float
    best_loss: float
    best_coordinates: np.ndarray
    initialization: str
    initialization_reason: str


def _emit_progress(callback, epoch, coordinates, loss, best_loss, best_coordinates, init_status):
    if callback is not None:
        callback(FitProgress(
            epoch=epoch, coordinates=np.array(coordinates, copy=True), loss=float(loss),
            best_loss=float(best_loss), best_coordinates=np.array(best_coordinates, copy=True),
            **init_status,
        ))


def _row_softmax_neg_distance(distance: np.ndarray, temperature: float) -> np.ndarray:
    return np.exp(_numpy_log_probabilities(distance, temperature))


def _sym_kl(target: np.ndarray, predicted: np.ndarray) -> float:
    q = np.asarray(predicted, dtype=np.float64)
    q = q / q.sum(axis=1, keepdims=True)
    logq = np.log(np.where(q > 0, q, np.finfo(q.dtype).tiny))
    return _numpy_kl_from_logs(target, logq)


def _seed_coordinates(n_obs: int, n_components: int, rng: np.random.Generator) -> np.ndarray:
    """Scatter a small cloud with nonzero angular spread.

    Piling every cell at the origin makes all pairwise Poincaré lengths vanish,
    which is a singular point of the Euclidean gradient of ``arcosh``.
    """
    directions = rng.normal(size=(n_obs, n_components))
    directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12, None)
    radii = rng.uniform(0.03, 0.12, size=(n_obs, 1))
    return confine_to_open_disk(directions * radii)


def _seed_from_connectivities(
    connectivities,
    n_components: int,
    rng: np.random.Generator,
    radius: float = 0.2,
    *,
    status: dict[str, str] | None = None,
) -> np.ndarray:
    """Start from a spectral layout of the connectivity graph.

    Spectral initialization can preserve local structure better than random
    angles, but does not guarantee a good optimum. Non-trivial eigenvectors
    of symmetric normalized adjacency provide the starting layout. This is
    the same spectral principle as UMAP, not a reuse of its graph or code.
    """
    graph = sparse.csr_matrix(connectivities, dtype=np.float64)
    n_obs = graph.shape[0]
    if status is not None:
        status.update(initialization="spectral", initialization_reason="")

    def random_fallback(reason: str, message: str | None = None):
        if status is not None:
            status.update(initialization="random_fallback", initialization_reason=reason)
        if message:
            warnings.warn(message, UserWarning, stacklevel=3)
        return _seed_coordinates(n_obs, n_components, rng)

    if graph.shape[0] != graph.shape[1]:
        raise ValueError("connectivities must be square.")
    if not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0):
        raise ValueError("connectivities must be finite and nonnegative.")
    if n_obs <= n_components + 1:
        return random_fallback("too_few_cells")
    graph = (0.5 * (graph + graph.T)).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    if graph.nnz == 0 or np.any(degree <= 0.0):
        return random_fallback("isolated_cells", "Spectral initialization has isolated cells; using seeded random coordinates.")
    inverse_sqrt = sparse.diags(1.0 / np.sqrt(degree))
    normalized = (inverse_sqrt @ graph @ inverse_sqrt).tocsr()
    n_vectors = n_components + 1
    try:
        if n_obs <= 2000:
            evals, evecs = np.linalg.eigh(normalized.toarray())
        else:
            # Largest algebraic, not magnitude: negative modes of nearly
            # bipartite graphs are not the lowest Laplacian eigenmodes.
            evals, evecs = eigsh(normalized, k=n_vectors, which="LA", v0=rng.normal(size=n_obs))
    except (np.linalg.LinAlgError, ArpackNoConvergence):
        return random_fallback("eigensolver_failure", "Spectral eigensolver did not converge; using seeded random coordinates.")
    order = np.argsort(evals)[::-1]
    coords = np.asarray(evecs[:, order][:, 1:n_vectors], dtype=np.float64)
    if coords.shape[1] < n_components or not np.all(np.isfinite(coords)):
        return random_fallback("invalid_eigenvectors", "Invalid spectral coordinates; using seeded random initialization.")
    coords = coords - coords.mean(axis=0, keepdims=True)
    max_radius = float(np.linalg.norm(coords, axis=1).max())
    if not np.isfinite(max_radius) or max_radius < 1e-12:
        return random_fallback("degenerate_coordinates", "Degenerate spectral coordinates; using seeded random initialization.")
    coords = coords / max_radius * radius
    coords = coords + rng.normal(scale=radius * 1e-3, size=coords.shape)
    return confine_to_open_disk(coords)


def evaluate_alignment(
    coordinates: np.ndarray,
    target: np.ndarray,
    temperature: float,
    *,
    block_size: int = 256,
    objective: str = DEFAULT_OBJECTIVE,
) -> float:
    """Evaluate the selected global KL in row blocks, as used during fitting."""
    _validate_objective(objective)
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    total = 0.0
    for start in range(0, len(coordinates), block_size):
        stop = min(start + block_size, len(coordinates))
        distance = pairwise_disk_length(coordinates[start:stop], coordinates)
        distance[np.arange(stop - start), np.arange(start, stop)] = 0.0
        total += _numpy_alignment_loss(target[start:stop], distance, temperature,
                                       objective=objective, self_columns=np.arange(start, stop))
    return total


def _torch_sym_kl(target, predicted):
    q = predicted / predicted.sum(dim=1, keepdim=True)
    logq = torch.where(q > 0, q, torch.finfo(q.dtype).tiny).log()
    return _torch_kl_from_logs(target, logq)


def _shrink_torch(points, margin: float = 1e-5):
    radii = torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    limit = 1.0 - margin
    scale = torch.where(radii > limit, limit / radii.clamp_min(1e-12), torch.ones_like(radii))
    return points * scale


def _torch_disk_length(left, right=None, self_columns=None):
    """Pairwise length from direct Euclidean separations, with no distance floor.

    Use the direct cdist kernel: the matrix-multiplication kernel loses small
    separations to cancellation. At exact coincidences cdist uses a zero
    subgradient; distinct arbitrarily close points retain their gradients.
    """
    square = right is None
    right = left if square else right
    left_sq = (left * left).sum(dim=-1)
    right_sq = (right * right).sum(dim=-1)
    if (not torch.isfinite(left_sq).all() or not torch.isfinite(right_sq).all()
            or torch.any(left_sq >= 1.0) or torch.any(right_sq >= 1.0)):
        raise ValueError("Disk coordinates must be finite and strictly inside the unit ball.")
    separation = torch.cdist(left, right, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    denom = (1.0 - left_sq[:, None]) * (1.0 - right_sq[None, :])
    dist = 2.0 * torch.asinh(separation / torch.sqrt(denom))
    if square:
        self_columns = torch.arange(dist.shape[0], device=dist.device)
    if self_columns is not None:
        dist = dist.clone()
        dist[torch.arange(dist.shape[0], device=dist.device), self_columns] = 0.0
    return dist


def _riemannian_retract(points, euclid_grad, step_size: float, max_grad_norm: float | None = 2.5):
    """Projected first-order retraction, clipping in the Poincaré metric.

    With lambda_x = 2/(1-|x|²), grad_R = grad_E/lambda_x² and
    |grad_R|_g = |grad_E|/lambda_x. Clipping grad_E before this conversion
    incorrectly suppresses the naturally large gradients near the boundary.
    """
    if not torch.isfinite(euclid_grad).all():
        raise FloatingPointError("Non-finite Poincaré gradient; update aborted.")
    sqnorm = (points * points).sum(dim=-1, keepdim=True)
    inverse_lambda = (1.0 - sqnorm).clamp_min(0.0) / 2.0
    gradient = inverse_lambda.square() * euclid_grad
    if max_grad_norm is not None:
        metric_norm = inverse_lambda * torch.linalg.vector_norm(euclid_grad, dim=-1, keepdim=True)
        gradient *= torch.clamp(max_grad_norm / metric_norm.clamp_min(1e-300), max=1.0)
    return _shrink_torch(points - step_size * gradient)


def _riemannian_retract_numpy(points, euclid_grad, step_size: float, max_grad_norm: float | None = 2.5):
    """NumPy equivalent of the metric-clipped Torch update."""
    if not np.all(np.isfinite(euclid_grad)):
        raise FloatingPointError("Non-finite Poincaré gradient; update aborted.")
    inverse_lambda = np.clip(1.0 - np.sum(points * points, axis=-1, keepdims=True), 0.0, None) / 2.0
    gradient = inverse_lambda**2 * euclid_grad
    if max_grad_norm is not None:
        metric_norm = inverse_lambda * np.linalg.norm(euclid_grad, axis=-1, keepdims=True)
        gradient *= np.minimum(1.0, max_grad_norm / np.maximum(metric_norm, 1e-300))
    return confine_to_open_disk(points - step_size * gradient)


def _fit_with_torch(
    target: np.ndarray,
    n_components: int,
    temperature: float,
    n_epochs: int,
    learning_rate: float,
    batch_size: int | None,
    burn_in_epochs: int,
    min_delta: float,
    patience: int,
    rng: np.random.Generator,
    verbose: bool,
    connectivities=None,
    max_grad_norm: float | None = 2.5,
    initial_coordinates: np.ndarray | None = None,
    progress_callback: Callable[[FitProgress], None] | None = None,
    objective: str = DEFAULT_OBJECTIVE,
) -> FitReport:
    n_obs = target.shape[0]
    device = torch.device("cpu")
    target_t = torch.as_tensor(target, dtype=torch.float64, device=device)
    init_status = {"initialization": "random", "initialization_reason": ""}
    if initial_coordinates is not None:
        seed = initial_coordinates.copy()
        init_status["initialization"] = "provided"
    elif connectivities is None:
        seed = _seed_coordinates(n_obs, n_components, rng)
    else:
        seed = _seed_from_connectivities(connectivities, n_components, rng, status=init_status)
    coords = torch.tensor(
        seed,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    chunk_size = min(batch_size or 256, n_obs)
    loss_trace: list[float] = []
    best_loss = np.inf
    best_coords = coords.detach().cpu().numpy().copy()
    stall = 0
    stopping_loss = np.inf

    # Evaluate each updated checkpoint on the next pass, reusing that
    # forward pass for its gradient. The last pass needs no gradient.
    for epoch in range(n_epochs + 1):
        coords.grad = None
        total_loss = 0.0
        with torch.set_grad_enabled(epoch < n_epochs):
            for start in range(0, n_obs, chunk_size):
                stop = min(start + chunk_size, n_obs)
                self_columns = torch.arange(start, stop, device=device)
                distance = _torch_disk_length(coords[start:stop], coords, self_columns=self_columns)
                loss = _torch_alignment_loss(target_t[start:stop], distance, temperature,
                                             objective=objective, self_columns=self_columns)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite global Poincaré fitting loss.")
                total_loss += float(loss.detach().cpu())
                if epoch < n_epochs:
                    # Columns also depend on coords. Accumulate ALL coordinate
                    # gradients at the same checkpoint, then clip/update once.
                    loss.backward()

        if epoch > 0:
            loss_trace.append(total_loss)
            if verbose and epoch % max(n_epochs // 10, 1) == 0:
                print(f"poincare_map epoch {epoch}/{n_epochs}  loss={total_loss:.6f}")
        if total_loss < best_loss:
            best_loss = total_loss
            best_coords = coords.detach().cpu().numpy().copy()
        _emit_progress(progress_callback, epoch, coords.detach().cpu().numpy(),
                       total_loss, best_loss, best_coords, init_status)
        # Keep the checkpoint optimum separate from the last significant
        # improvement. Several small improvements can add up to min_delta.
        if epoch <= burn_in_epochs or total_loss + min_delta < stopping_loss:
            stopping_loss = total_loss
            stall = 0
        else:
            stall += 1
        if epoch == n_epochs or (epoch > burn_in_epochs and stall >= patience):
            break
        step = learning_rate / 10.0 if epoch < burn_in_epochs else learning_rate
        with torch.no_grad():
            coords.copy_(_riemannian_retract(coords, coords.grad, step, max_grad_norm))

    return FitReport(
        coordinates=confine_to_open_disk(best_coords),
        loss_trace=loss_trace,
        n_epochs=len(loss_trace),
        backend="torch",
        objective=objective,
        batch_size=chunk_size,
        best_loss=best_loss,
        stop_reason="epoch_limit" if len(loss_trace) == n_epochs else "patience",
        **init_status,
    )


def _numeric_row_grad(
    coordinates: np.ndarray,
    target: np.ndarray,
    temperature: float,
    rows: np.ndarray,
    eps: float = 1e-5,
    *,
    objective: str = DEFAULT_OBJECTIVE,
) -> np.ndarray:
    """Central-difference Euclidean gradient on a subset of rows (numpy fallback)."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Finite-difference eps must be finite and positive.")
    grad = np.zeros_like(coordinates)

    def _loss(pos: np.ndarray) -> float:
        # rows selects coordinates to differentiate, not a conditional target.
        return evaluate_alignment(pos, target, temperature, objective=objective)

    for i in rows:
        separation = np.linalg.norm(coordinates - coordinates[i], axis=1)
        positive = separation[separation > 0.0]
        # Do not cross a nearby distance kink during a coordinate difference.
        neighbor_step = .01 * positive.min() if positive.size else eps
        for axis in range(coordinates.shape[1]):
            other = np.delete(coordinates[i], axis)
            axis_limit = np.sqrt(max(0.0, 1.0 - float(other @ other)))
            clearance = axis_limit - abs(coordinates[i, axis])
            step = min(eps, .001 * clearance, neighbor_step)
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[i, axis] += step
            minus[i, axis] -= step
            actual_step = plus[i, axis] - minus[i, axis]
            if step <= 0 or actual_step <= 0:
                raise FloatingPointError("Insufficient precision for a finite-difference step; use PyTorch.")
            # Never project a perturbation: that differentiates the projection,
            # not the objective. Both perturbed points remain in the open ball.
            grad[i, axis] = (_loss(plus) - _loss(minus)) / actual_step
    if not np.all(np.isfinite(grad)):
        raise FloatingPointError("Non-finite finite-difference gradient.")
    return grad


def _fit_with_numpy(
    target: np.ndarray,
    n_components: int,
    temperature: float,
    n_epochs: int,
    learning_rate: float,
    batch_size: int | None,
    burn_in_epochs: int,
    min_delta: float,
    patience: int,
    rng: np.random.Generator,
    verbose: bool,
    connectivities=None,
    max_grad_norm: float | None = 2.5,
    initial_coordinates: np.ndarray | None = None,
    progress_callback: Callable[[FitProgress], None] | None = None,
    objective: str = DEFAULT_OBJECTIVE,
) -> FitReport:
    n_obs = target.shape[0]
    init_status = {"initialization": "random", "initialization_reason": ""}
    if initial_coordinates is not None:
        coords = initial_coordinates.copy()
        init_status["initialization"] = "provided"
    elif connectivities is None:
        coords = _seed_coordinates(n_obs, n_components, rng)
    else:
        coords = _seed_from_connectivities(connectivities, n_components, rng, status=init_status)
    # Exact finite differences are cubic; this fallback is only for tiny inputs.
    chunk_size = min(batch_size or 32, n_obs)
    loss_trace: list[float] = []
    best_loss = evaluate_alignment(coords, target, temperature, objective=objective)
    best_coords = coords.copy()
    stopping_loss = best_loss
    stall = 0
    _emit_progress(progress_callback, 0, coords, best_loss, best_loss, best_coords, init_status)

    for epoch in range(n_epochs):
        grad = np.zeros_like(coords)
        for start in range(0, n_obs, chunk_size):
            rows = np.arange(start, min(start + chunk_size, n_obs))
            grad += _numeric_row_grad(coords, target, temperature, rows, objective=objective)
        step = learning_rate / 10.0 if epoch < burn_in_epochs else learning_rate
        coords = _riemannian_retract_numpy(coords, grad, step, max_grad_norm)

        total_loss = evaluate_alignment(coords, target, temperature, objective=objective)
        if not np.isfinite(total_loss):
            raise FloatingPointError("Non-finite global Poincaré fitting loss.")
        loss_trace.append(total_loss)
        if verbose and (epoch + 1) % max(n_epochs // 10, 1) == 0:
            print(f"poincare_map epoch {epoch + 1}/{n_epochs}  loss={total_loss:.6f}")
        if total_loss < best_loss:
            best_loss = total_loss
            best_coords = coords.copy()
        _emit_progress(progress_callback, epoch + 1, coords, total_loss, best_loss, best_coords, init_status)
        if epoch + 1 <= burn_in_epochs or total_loss + min_delta < stopping_loss:
            stopping_loss = total_loss
            stall = 0
        else:
            stall += 1
        if epoch + 1 > burn_in_epochs and stall >= patience:
            break

    return FitReport(
        coordinates=confine_to_open_disk(best_coords),
        loss_trace=loss_trace,
        n_epochs=len(loss_trace),
        backend="numpy",
        objective=objective,
        batch_size=chunk_size,
        best_loss=best_loss,
        stop_reason="epoch_limit" if len(loss_trace) == n_epochs else "patience",
        **init_status,
    )


def fit_disk_embedding(
    target: np.ndarray,
    *,
    n_components: int = 2,
    temperature: float = 2.0,
    n_epochs: int = 250,
    learning_rate: float = 0.01,
    batch_size: int | None = None,
    burn_in_epochs: int = 30,
    min_delta: float = 1e-4,
    patience: int = 50,
    random_state: int | np.random.Generator | None = 0,
    verbose: bool = False,
    connectivities=None,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    max_grad_norm: float | None = 2.5,
    initial_coordinates: np.ndarray | None = None,
    progress_callback: Callable[[FitProgress], None] | None = None,
    objective: str = DEFAULT_OBJECTIVE,
) -> FitReport:
    """Match the exact global proximity objective, with row-chunked gradients.

    ``connectivities`` provides spectral initialization. ``batch_size``
    bounds row workspace, NOT the set of comparison cells: all columns
    are always retained and a single global update is made per epoch.
    The target remains dense and this is not an approximate large-N method.
    ``objective='symmetric_kl'`` includes self probabilities (default).
    Experimental ``'conditional_symmetric_kl'`` excludes self from both
    distributions but keeps all other columns. It requires positive off-diagonal
    target mass in every row and does not preserve isolation/self-accessibility.
    ``max_grad_norm`` clips each accumulated gradient's Riemannian norm;
    None disables clipping. It does not cap coordinate radii.

    ``initial_coordinates`` explicitly warm-starts a new run; it must match
    the target's row count and fit within the optimizer margin (radius <=
    1-1e-5). This is not a complete restoration of the stopping/schedule state.
    ``progress_callback`` receives independent current/best coordinate copies
    after evaluation, including epoch 0. Exceptions propagate to the caller.
    Callbacks are observational and their return values are ignored.
    """
    _validate_objective(objective)
    shape = np.shape(target)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("target proximity must be a square matrix.")
    check_dense_budget(shape[0], max_dense_bytes)
    target = np.asarray(target, dtype=np.float64)
    if target.shape[0] < 2 or not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError("target must have at least two rows and finite nonnegative entries.")
    with np.errstate(over="ignore"):
        row_mass = target.sum(axis=1)
    if np.any(row_mass <= 0) or not np.all(np.isfinite(row_mass)):
        raise ValueError("target rows must have finite positive mass.")
    if n_components < 2:
        raise ValueError("n_components must be at least 2.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("gamma / temperature must be finite and positive.")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           for value in (n_epochs, patience, burn_in_epochs, n_components)):
        raise ValueError("Epoch counts, patience and n_components must be integers.")
    if n_epochs < 1 or patience < 1 or burn_in_epochs < 0:
        raise ValueError("n_epochs and patience must be positive; burn_in_epochs must be nonnegative.")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("learning_rate must be positive and min_delta nonnegative, both finite.")
    if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer or None.")
    if connectivities is not None and np.shape(connectivities) != target.shape:
        raise ValueError("connectivities and target must have the same shape.")
    if max_grad_norm is not None and (isinstance(max_grad_norm, bool) or not np.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be finite and positive, or None.")
    if initial_coordinates is not None:
        initial_coordinates = np.asarray(initial_coordinates, dtype=np.float64)
        if (initial_coordinates.shape != (target.shape[0], n_components)
                or not np.all(np.isfinite(initial_coordinates))):
            raise ValueError("initial_coordinates must be finite with shape (n_obs, n_components).")
        margin_limit = 1.0 - 1e-5 + 8.0 * np.finfo(np.float64).eps
        if np.any(np.linalg.norm(initial_coordinates, axis=1) > margin_limit):
            raise ValueError("initial_coordinates exceed the optimizer radius margin; use raw training coordinates, not recentered display coordinates.")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None.")
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if torch is None and target.shape[0] > 128:
        raise ImportError("Install PyTorch for more than 128 cells; the exact NumPy fallback uses cubic finite differences.")

    has_zero = False
    if objective == CONDITIONAL_OBJECTIVE:
        # Sum off-diagonal mass directly: total-minus-diagonal cancels when
        # self mass rounds to 1 while other entries remain tiny and positive.
        for start in range(0, len(target), 256):
            stop = min(start + 256, len(target))
            block = target[start:stop].copy()
            block[np.arange(stop - start), np.arange(start, stop)] = 0.
            mass = block.sum(axis=1)
            if np.any(mass <= 0):
                raise ValueError("Conditional objective requires positive off-diagonal mass in every target row.")
            has_zero |= np.count_nonzero(block == 0) > stop - start
    else:
        has_zero = bool(np.any(target == 0))
    if has_zero:
        warnings.warn(
            "Zero target probabilities use machine-tiny values in logarithms only. "
            "The reverse KL is numerically regularized, not exact for these zeros.",
            UserWarning, stacklevel=2,
        )

    kwargs = dict(
        target=target,
        n_components=n_components,
        temperature=temperature,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        burn_in_epochs=burn_in_epochs,
        min_delta=min_delta,
        patience=patience,
        rng=rng,
        verbose=verbose,
        connectivities=connectivities,
        max_grad_norm=max_grad_norm,
        initial_coordinates=initial_coordinates,
        progress_callback=progress_callback,
        objective=objective,
    )
    if torch is not None:
        return _fit_with_torch(**kwargs)
    return _fit_with_numpy(**kwargs)
