"""Opt-in research fitter: RFA KL plus graph-geodesic distance stress.

This is an independent extension, NOT the authors' PoincareMaps algorithm or
Riemannian Adam. It leaves sc_poincare.tl.poincare_map defaults unchanged.
Requires the torch extra. Features must already be preprocessed. No labels or
reference coordinates are accepted. Dense shortest paths and classical scaling
make this a small-data experiment, not a scalable alternative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import dijkstra

from sc_poincare._fitting import _torch_disk_length, _torch_sym_kl
from sc_poincare._proximity import estimate_proximity, local_affinity
from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES, check_dense_budget


@dataclass
class ExperimentalFit:
    coordinates: np.ndarray
    stages: list[dict]
    parameters: dict


def fit_geodesic_regularized(
    features: np.ndarray,
    *,
    n_neighbors: int = 15,
    sigma: float | str = "median",
    gamma: float = .5,
    learning_rate: float = .02,
    stages: tuple[tuple[int, float], ...] = ((800, 10.),),
    distance_scale: float = 3.,
    random_state: int = 0,
    chunk_size: int = 256,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
) -> ExperimentalFit:
    """Fit mean row-KL + weight * mean normalized squared geodesic error.

    Graph distances are divided by their off-diagonal mean, then multiplied by
    distance_scale. Initialization is 2-D classical scaling of the same graph
    distances, shrunk near the origin and given small seeded noise.

    Adam acts on free spatial coordinates u with x=u/(1+sqrt(1+|u|²)). Each stage
    is (epochs, positive stress weight), starts from the preceding stage's best
    checkpoint and resets Adam's state. The loss is not comparable across stages
    of different weights. There is no metric-based checkpoint selection.
    """
    import torch

    for name, value in [("gamma", gamma), ("learning_rate", learning_rate),
                        ("distance_scale", distance_scale)]:
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if not stages:
        raise ValueError("At least one stage is required.")
    for epochs, weight in stages:
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("Stage epochs must be positive integers.")
        if isinstance(weight, bool) or not np.isfinite(weight) or weight <= 0:
            raise ValueError("Stress weights must be finite and positive; use the normal fitter for pure KL.")
    shape = np.shape(features)
    if len(shape) != 2:
        raise ValueError("features must be a matrix.")
    check_dense_budget(shape[0], max_dense_bytes)
    features = np.asarray(features, dtype=np.float64)
    proximity = estimate_proximity(features, n_neighbors, sigma, max_dense_bytes=max_dense_bytes)
    # Only lengths enter shortest paths. A wide kernel here avoids irrelevant
    # underflow while obtaining the same stitched edge topology.
    wide_sigma = 2 * float(np.linalg.norm(features, axis=1).max()) + 1.
    _, lengths = local_affinity(features, n_neighbors, wide_sigma)
    geodesics = dijkstra(lengths, directed=False)
    n = len(features)
    mean_distance = float(geodesics.sum() / (n * (n - 1)))
    if not np.all(np.isfinite(geodesics)) or not np.isfinite(mean_distance) or mean_distance <= 0:
        raise ValueError("Graph distances must have a finite positive off-diagonal mean.")
    d2 = geodesics**2
    gram = -.5 * (d2 - d2.mean(0) - d2.mean(1, keepdims=True) + d2.mean())
    values, vectors = eigh(gram, subset_by_index=[n - 2, n - 1])
    initial = vectors[:, ::-1] * np.sqrt(np.maximum(values[::-1], 0.))
    initial *= .2 / max(np.linalg.norm(initial, axis=1).max(), 1e-12)
    initial += np.random.default_rng(random_state).normal(scale=.0002, size=initial.shape)
    del d2, gram
    target = torch.tensor(proximity.accessibility, dtype=torch.float64)
    global_target = torch.tensor(geodesics / mean_distance * distance_scale, dtype=torch.float64)
    stage_reports = []
    for epochs, weight in stages:
        u = torch.tensor(2 * initial / (1 - np.sum(initial**2, axis=1, keepdims=True)),
                         dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=learning_rate)
        best_loss = np.inf
        best = initial.copy()
        trace = []
        for epoch in range(epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            with torch.set_grad_enabled(epoch < epochs):
                for start in range(0, n, chunk_size):
                    stop = min(start + chunk_size, n)
                    coords = u / (1 + torch.sqrt(1 + (u * u).sum(-1, keepdim=True)))
                    distances = _torch_disk_length(coords[start:stop], coords,
                                                   self_columns=torch.arange(start, stop))
                    kl = _torch_sym_kl(target[start:stop], torch.softmax(-distances / gamma, dim=1)) / n
                    stress_sum = ((distances - global_target[start:stop]) / distance_scale).square().sum()
                    loss = kl + weight * stress_sum / (n * (n - 1))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite experimental objective.")
                    total += float(loss.detach())
                    if epoch < epochs:
                        loss.backward()
            if epoch > 0:
                trace.append(total)
            if total < best_loss:
                best_loss = total
                best = coords.detach().numpy().copy()
            if epoch == epochs:
                break
            if not torch.isfinite(u.grad).all():
                raise FloatingPointError("Non-finite experimental gradient.")
            optimizer.step()
        stage_reports.append({"epochs": int(epochs), "weight": float(weight),
                              "best_loss": best_loss, "loss_trace": np.asarray(trace)})
        initial = best
    return ExperimentalFit(initial, stage_reports, {
        "objective": "mean_RFA_KL_plus_normalized_geodesic_stress",
        "optimizer": "spatial_coordinate_adam", "initialization": "geodesic_classical_scaling",
        "n_neighbors": n_neighbors, "sigma_requested": sigma,
        "sigma": proximity.kernel_width, "gamma": gamma, "distance_scale": distance_scale,
        "learning_rate": learning_rate, "random_state": random_state,
        "proximity_diagnostics": proximity.diagnostics,
    })
