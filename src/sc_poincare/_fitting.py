"""Riemannian fitting of disk coordinates to a target proximity matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sc_poincare._disk import confine_to_open_disk, pairwise_disk_length
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


def _row_softmax_neg_distance(distance: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -distance / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    unnormalized = np.exp(scaled)
    return unnormalized / np.clip(unnormalized.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)


def _sym_kl(target: np.ndarray, predicted: np.ndarray) -> float:
    floor = 1e-12
    p = np.clip(target, floor, None)
    q = np.clip(predicted, floor, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return float(np.sum(p * np.log(p / q) + q * np.log(q / p)))


def _seed_coordinates(n_obs: int, n_components: int, rng: np.random.Generator) -> np.ndarray:
    """Scatter a small cloud with nonzero angular spread.

    Piling every cell at the origin makes all pairwise Poincaré lengths vanish,
    which is a singular point of the Euclidean gradient of ``arcosh``.
    """
    directions = rng.normal(size=(n_obs, n_components))
    directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12, None)
    radii = rng.uniform(0.03, 0.12, size=(n_obs, 1))
    return confine_to_open_disk(directions * radii)


def evaluate_alignment(
    coordinates: np.ndarray,
    target: np.ndarray,
    temperature: float,
    *,
    block_size: int = 256,
) -> float:
    """Global symmetric KL in row blocks, using the same distances as fitting."""
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    total = 0.0
    for start in range(0, len(coordinates), block_size):
        stop = min(start + block_size, len(coordinates))
        distance = pairwise_disk_length(coordinates[start:stop], coordinates)
        distance[np.arange(stop - start), np.arange(start, stop)] = 0.0
        predicted = _row_softmax_neg_distance(distance, temperature)
        total += _sym_kl(target[start:stop], predicted)
    return total


def _torch_sym_kl(target, predicted):
    floor = 1e-12
    p = target.clamp_min(floor)
    q = predicted.clamp_min(floor)
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    return torch.sum(p * (p.log() - q.log()) + q * (q.log() - p.log()))


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
    max_grad_norm: float | None = 2.5,
) -> FitReport:
    n_obs = target.shape[0]
    device = torch.device("cpu")
    target_t = torch.as_tensor(target, dtype=torch.float64, device=device)
    init_status = {"initialization": "random", "initialization_reason": ""}
    seed = _seed_coordinates(n_obs, n_components, rng)
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
                predicted = torch.softmax(-distance / temperature, dim=1)
                loss = _torch_sym_kl(target_t[start:stop], predicted)
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
) -> np.ndarray:
    """Central-difference Euclidean gradient on a subset of rows (numpy fallback)."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Finite-difference eps must be finite and positive.")
    grad = np.zeros_like(coordinates)

    def _loss(pos: np.ndarray) -> float:
        # rows selects coordinates to differentiate, not a conditional target.
        return evaluate_alignment(pos, target, temperature)

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
    max_grad_norm: float | None = 2.5,
) -> FitReport:
    n_obs = target.shape[0]
    init_status = {"initialization": "random", "initialization_reason": ""}
    coords = _seed_coordinates(n_obs, n_components, rng)
    # Exact finite differences are cubic; this fallback is only for tiny inputs.
    chunk_size = min(batch_size or 32, n_obs)
    loss_trace: list[float] = []
    best_loss = evaluate_alignment(coords, target, temperature)
    best_coords = coords.copy()
    stopping_loss = best_loss
    stall = 0

    for epoch in range(n_epochs):
        grad = np.zeros_like(coords)
        for start in range(0, n_obs, chunk_size):
            rows = np.arange(start, min(start + chunk_size, n_obs))
            grad += _numeric_row_grad(coords, target, temperature, rows)
        step = learning_rate / 10.0 if epoch < burn_in_epochs else learning_rate
        coords = _riemannian_retract_numpy(coords, grad, step, max_grad_norm)

        total_loss = evaluate_alignment(coords, target, temperature)
        if not np.isfinite(total_loss):
            raise FloatingPointError("Non-finite global Poincaré fitting loss.")
        loss_trace.append(total_loss)
        if verbose and (epoch + 1) % max(n_epochs // 10, 1) == 0:
            print(f"poincare_map epoch {epoch + 1}/{n_epochs}  loss={total_loss:.6f}")
        if total_loss < best_loss:
            best_loss = total_loss
            best_coords = coords.copy()
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
    max_grad_norm: float | None = 2.5,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
) -> FitReport:
    """Match the exact global proximity objective, with row-chunked gradients.

    ``batch_size`` bounds row workspace, NOT the set of comparison cells: all columns
    are always retained and a single global update is made per epoch.
    The target remains dense and this is not an approximate large-N method.
    ``max_grad_norm`` clips each accumulated gradient's Riemannian norm;
    None disables clipping. It does not cap coordinate radii.
    """
    shape = np.shape(target)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("target proximity must be a square matrix.")
    check_dense_budget(shape[0], max_dense_bytes)
    target = np.asarray(target, dtype=np.float64)
    if target.shape[0] < 2 or not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError("target must have at least two rows and finite nonnegative entries.")
    if np.any(target.sum(axis=1) <= 0):
        raise ValueError("target rows must have positive mass.")
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
    if max_grad_norm is not None and (isinstance(max_grad_norm, bool) or not np.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be finite and positive, or None.")
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if torch is None and target.shape[0] > 128:
        raise ImportError("Install PyTorch for more than 128 cells; the exact NumPy fallback uses cubic finite differences.")

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
        max_grad_norm=max_grad_norm,
    )
    if torch is not None:
        return _fit_with_torch(**kwargs)
    return _fit_with_numpy(**kwargs)
