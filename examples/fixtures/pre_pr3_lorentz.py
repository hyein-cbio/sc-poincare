"""Frozen fit_lorentz from cd6ae2d6a9754b34e934d72328ac7cb9ec1a17d8.

Function text is verbatim; imports resolve unchanged helpers in the current package.
This is an intentionally unsafe historical research control, never a model default.
"""
import numpy as np
from sc_poincare import experimental
from sc_poincare.controlled_probe import lorentz_from_disk, _minkowski


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
