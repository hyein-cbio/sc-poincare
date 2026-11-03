"""Explicit research controls; no default fitter imports this module.

``acosh_exact`` copies the pre-13ce854 distance operations, without changing their
backward behavior. ``acosh_masked`` is a new stabilized reconstruction, not a
historical replay: self and clamped coincident pairs get zero derivatives.
All factorial arms use current chunk-wise backward and the same retraction.
"""
from __future__ import annotations

import numpy as np

from sc_poincare import controlled_probe as cp, experimental

DISTANCES = ("disk_stable", "acosh_exact", "acosh_masked")
GRADIENTS = ("corrected", "legacy")
HISTORICAL_COMMIT = "cd6ae2d6a9754b34e934d72328ac7cb9ec1a17d8"
HISTORICAL_FUNCTION_SHA256 = "97bde9824263931ceea71f286eadc6cefd97c85d1c15b912817e2d0d36830d2a"


def distance_rows(point, start, stop, mode):
    torch = experimental._torch()
    rows = torch.arange(start, stop)
    if mode == "disk_stable":
        from sc_poincare._fitting import _torch_disk_length
        disk = point[:, 1:] / (point[:, :1] + 1.)
        return _torch_disk_length(disk[start:stop], disk, self_columns=rows)
    if mode not in DISTANCES:
        raise ValueError(f"Unknown distance mode: {mode}")
    left, right = point[start:stop], point
    inner = -left[:, :1] * right[:, 0] + left[:, 1:] @ right[:, 1:].T
    if mode == "acosh_exact":
        # Verbatim distance operations from the pinned historical function.
        distance = torch.acosh(torch.clamp(-inner, min=1.))
        distance = distance.clone()
        distance[torch.arange(stop - start), rows] = 0
        return distance
    argument = torch.clamp(-inner, min=1.)
    zero = argument <= 1.
    zero[torch.arange(stop - start), rows] = True
    # Mask BEFORE acosh: post-hoc zeroing alone still backpropagates 0 * inf.
    return torch.acosh(argument.masked_fill(zero, 2.)).masked_fill(zero, 0.)


def objective(point, target, teacher, config, chunk_size, *, distance_mode,
              backward=False, backward_mode="chunked"):
    """Same MU terms/normalization, selectable distance and backward schedule."""
    if distance_mode not in DISTANCES or backward_mode not in ("chunked", "full"):
        raise ValueError("Unknown distance or backward mode")
    torch = experimental._torch()
    n = len(point)
    point.grad = None
    total = 0.
    full = 0.
    with torch.set_grad_enabled(backward):
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            rows = torch.arange(start, stop)
            distance = distance_rows(point, start, stop, distance_mode)
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
                raise FloatingPointError("Nonfinite ablation objective")
            total += float(loss.detach())
            if backward:
                if backward_mode == "chunked":
                    loss.backward()
                else:
                    full = full + global_loss + config.local_weight * local_loss
        if backward and backward_mode == "full":
            full.backward()
    return total


def tangent_gradient(point, gradient, mode):
    if mode == "corrected":
        return cp._lorentz_tangent_gradient(point, gradient)
    if mode == "legacy":
        return gradient + cp._minkowski(point, gradient)[:, None] * point
    raise ValueError(f"Unknown gradient mode: {mode}")


def geometry_diagnostics(point):
    torch = experimental._torch()
    disk = point[:, 1:] / (point[:, :1] + 1.)
    radius_sq = (disk * disk).sum(1)
    if (not torch.isfinite(point).all() or not torch.isfinite(disk).all()
            or torch.any(radius_sq >= 1.) or torch.any(point[:, 0] <= 0)):
        raise FloatingPointError("Invalid ablation coordinates")
    return dict(minkowski_max_abs_error=float((cp._minkowski(point, point) + 1).abs().max()),
                min_time_coordinate=float(point[:, 0].min()),
                max_unit_disk_radius=float(radius_sq.max().sqrt()),
                min_unit_disk_margin=float((1 - radius_sq).min()))


def fit_control(setup, initial, *, distance_mode, gradient_mode, epochs=20,
                chunk_size=256, max_dense_bytes, backward_mode="chunked"):
    """Return a labeled failure at the last finite checkpoint; never repair it."""
    if distance_mode not in DISTANCES or gradient_mode not in GRADIENTS:
        raise ValueError("Unknown ablation arm")
    experimental._validate_setup(setup, max_dense_bytes)
    epochs = experimental._integer("epochs", epochs, 0)
    chunk_size = experimental._integer("chunk_size", chunk_size)
    torch = experimental._torch()
    point = torch.tensor(cp.lorentz_from_disk(initial), dtype=torch.float64, requires_grad=True)
    target = torch.tensor(np.array(setup.target), dtype=torch.float64)
    teacher = torch.tensor(np.array(setup.teacher), dtype=torch.float64)
    trace, diagnostics = [], []
    failure = None
    completed = 0
    for epoch in range(epochs + 1):
        with torch.no_grad():
            diagnostics.append(dict(epoch=epoch, **geometry_diagnostics(point)))
        value = objective(point, target, teacher, setup.config, chunk_size,
                          distance_mode=distance_mode, backward=epoch < epochs,
                          backward_mode=backward_mode)
        trace.append(value)
        if epoch == epochs:
            break
        count = int((~torch.isfinite(point.grad)).sum())
        diagnostics[-1]["nonfinite_gradient_entries"] = count
        if count:
            failure = dict(epoch=epoch, stage="backward", nonfinite_gradient_entries=count)
            break
        tangent = tangent_gradient(point.detach(), point.grad, gradient_mode)
        diagnostics[-1]["gradient_max_abs"] = float(point.grad.abs().max())
        diagnostics[-1]["tangent_constraint_max_abs"] = float(cp._minkowski(point.detach(), tangent).abs().max())
        step = -setup.config.learning_rate * tangent
        norm = torch.sqrt(cp._minkowski(step, step).clamp_min(1e-24))
        updated = torch.cosh(norm)[:, None] * point.detach() + torch.sinh(norm)[:, None] * step / norm[:, None]
        scale = torch.sqrt(torch.clamp(-cp._minkowski(updated, updated), min=1e-24))
        updated = updated / scale[:, None]
        updated = torch.where(updated[:, :1] < 0, -updated, updated)
        try:
            with torch.no_grad():
                geometry_diagnostics(updated)
        except FloatingPointError:
            failure = dict(epoch=epoch, stage="update")
            break
        point = updated.detach().requires_grad_(True)
        completed += 1
    current = point.detach()
    return dict(status="completed" if failure is None else "failed_numerical", failure=failure,
                distance_mode=distance_mode, gradient_mode=gradient_mode, backward_mode=backward_mode,
                requested_epochs=epochs, epochs_completed=completed, loss_trace=trace,
                current=current.numpy(), minkowski_diag=cp._minkowski(current, current).numpy(),
                trajectory_diagnostics=diagnostics)
