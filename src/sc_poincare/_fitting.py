"""Riemannian fitting of disk coordinates to a target proximity matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sc_poincare._disk import confine_to_open_disk, pairwise_disk_length

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


def evaluate_alignment(coordinates: np.ndarray, target: np.ndarray, temperature: float) -> float:
    distance = pairwise_disk_length(coordinates)
    predicted = _row_softmax_neg_distance(distance, temperature)
    return _sym_kl(target, predicted)


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


def _clip_rows(matrix, max_norm: float = 5.0):
    norms = torch.linalg.vector_norm(matrix, dim=-1, keepdim=True).clamp_min(1e-12)
    return matrix * torch.clamp(max_norm / norms, max=1.0)


def _riemannian_retract(points, euclid_grad, step_size: float):
    euclid_grad = torch.nan_to_num(euclid_grad, nan=0.0, posinf=0.0, neginf=0.0)
    euclid_grad = _clip_rows(euclid_grad, max_norm=5.0)
    sqnorm = (points * points).sum(dim=-1, keepdim=True)
    conformal = ((1.0 - sqnorm).clamp_min(0.0) ** 2) / 4.0
    relocated = points - step_size * conformal * euclid_grad
    return _shrink_torch(relocated)


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
) -> FitReport:
    n_obs = target.shape[0]
    device = torch.device("cpu")
    target_t = torch.as_tensor(target, dtype=torch.float64, device=device)
    coords = torch.tensor(
        _seed_coordinates(n_obs, n_components, rng),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    full_batch = batch_size is None or batch_size >= n_obs
    loss_trace: list[float] = []
    best_loss = np.inf
    best_coords = coords.detach().cpu().numpy().copy()
    stall = 0
    order = np.arange(n_obs)

    for epoch in range(n_epochs):
        if full_batch:
            batches = [order]
        else:
            rng.shuffle(order)
            batches = [order[i : i + batch_size] for i in range(0, n_obs, batch_size)]
            batches = [b for b in batches if len(b) >= 2]

        epoch_loss = 0.0
        seen = 0
        step = learning_rate / 10.0 if epoch < burn_in_epochs else learning_rate
        for batch in batches:
            idx = torch.as_tensor(np.ascontiguousarray(batch), dtype=torch.long, device=device)
            piece = coords[idx]
            local_target = target_t.index_select(0, idx).index_select(1, idx)
            local_target = local_target / local_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
            distance = _torch_disk_length(piece)
            logits = -distance / temperature
            predicted = torch.softmax(logits, dim=1)
            loss = _torch_sym_kl(local_target, predicted)
            if not torch.isfinite(loss):
                continue
            if coords.grad is not None:
                coords.grad.zero_()
            loss.backward()
            with torch.no_grad():
                grad = torch.zeros_like(coords)
                if coords.grad is not None:
                    grad[idx] = coords.grad[idx]
                coords.copy_(_riemannian_retract(coords, grad, step))
                if coords.grad is not None:
                    coords.grad.zero_()
            epoch_loss += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)

        if seen == 0:
            if not loss_trace:
                raise FloatingPointError(
                    "poincare fitting produced a non-finite loss on every batch of the first epoch."
                )
            break
        mean_loss = epoch_loss / seen
        loss_trace.append(mean_loss)
        if verbose and (epoch + 1) % max(n_epochs // 10, 1) == 0:
            print(f"poincare_map epoch {epoch + 1}/{n_epochs}  loss={mean_loss:.6f}")

        if mean_loss + min_delta < best_loss:
            best_loss = mean_loss
            best_coords = coords.detach().cpu().numpy().copy()
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    return FitReport(
        coordinates=confine_to_open_disk(best_coords),
        loss_trace=loss_trace,
        n_epochs=len(loss_trace),
        backend="torch",
    )


def _numeric_row_grad(
    coordinates: np.ndarray,
    target: np.ndarray,
    temperature: float,
    rows: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference Euclidean gradient on a subset of rows (numpy fallback)."""
    grad = np.zeros_like(coordinates)
    tgt = target[np.ix_(rows, rows)]
    tgt = tgt / np.clip(tgt.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)

    def _loss(pos: np.ndarray) -> float:
        dist = pairwise_disk_length(pos[rows])
        pred = _row_softmax_neg_distance(dist, temperature)
        return _sym_kl(tgt, pred)

    for i in rows:
        for axis in range(coordinates.shape[1]):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[i, axis] += eps
            minus[i, axis] -= eps
            plus = confine_to_open_disk(plus)
            minus = confine_to_open_disk(minus)
            grad[i, axis] = (_loss(plus) - _loss(minus)) / (2.0 * eps)
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


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
) -> FitReport:
    n_obs = target.shape[0]
    coords = _seed_coordinates(n_obs, n_components, rng)
    # The finite-difference gradient is O(rows^3); keep rows small.
    batch_size = min(batch_size or 32, 32)
    full_batch = batch_size >= n_obs
    loss_trace: list[float] = []
    best_loss = np.inf
    best_coords = coords.copy()
    stall = 0
    order = np.arange(n_obs)

    for epoch in range(n_epochs):
        if full_batch:
            batches = [order]
        else:
            rng.shuffle(order)
            batches = [order[i : i + batch_size] for i in range(0, n_obs, batch_size)]
            batches = [b for b in batches if len(b) >= 2]

        epoch_loss = 0.0
        seen = 0
        step = learning_rate / 10.0 if epoch < burn_in_epochs else learning_rate
        for batch in batches:
            local_target = target[np.ix_(batch, batch)]
            local_target = local_target / np.clip(local_target.sum(axis=1, keepdims=True), 1e-12, None)
            local_coords = coords[batch]
            dist = pairwise_disk_length(local_coords)
            pred = _row_softmax_neg_distance(dist, temperature)
            epoch_loss += _sym_kl(local_target, pred) * len(batch)
            seen += len(batch)

            grad = _numeric_row_grad(coords, target, temperature, np.asarray(batch))
            row_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = grad * np.clip(5.0 / np.clip(row_norm, 1e-12, None), a_min=None, a_max=1.0)
            sqnorm = np.sum(coords * coords, axis=1, keepdims=True)
            conformal = (np.clip(1.0 - sqnorm, 0.0, None) ** 2) / 4.0
            coords = confine_to_open_disk(coords - step * conformal * grad)

        mean_loss = epoch_loss / max(seen, 1)
        loss_trace.append(mean_loss)
        if verbose and (epoch + 1) % max(n_epochs // 10, 1) == 0:
            print(f"poincare_map epoch {epoch + 1}/{n_epochs}  loss={mean_loss:.6f}")
        if mean_loss + min_delta < best_loss:
            best_loss = mean_loss
            best_coords = coords.copy()
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    return FitReport(
        coordinates=confine_to_open_disk(best_coords),
        loss_trace=loss_trace,
        n_epochs=len(loss_trace),
        backend="numpy",
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
) -> FitReport:
    """Match softmaxed disk distances to a row-stochastic proximity matrix."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("target proximity must be a square matrix.")
    if n_components < 2:
        raise ValueError("n_components must be at least 2.")
    if temperature <= 0:
        raise ValueError("gamma / temperature must be positive.")
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    n_obs = target.shape[0]
    if batch_size is None and n_obs > 4000:
        batch_size = 256

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
    )
    if torch is not None:
        return _fit_with_torch(**kwargs)
    return _fit_with_numpy(**kwargs)
