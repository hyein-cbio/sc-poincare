"""Numerically stable row-wise symmetric KL objectives.

Positive target probabilities, including subnormals, are not floored. Exact
zeros use the dtype's smallest normal positive value *in logarithms only*;
this is a finite numerical regularization of otherwise infinite reverse KL,
not probability smoothing or an exact divergence when zeros are present.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

DEFAULT_OBJECTIVE = "symmetric_kl"
CONDITIONAL_OBJECTIVE = "conditional_symmetric_kl"


def _validate_objective(objective):
    if not isinstance(objective, str) or objective not in (DEFAULT_OBJECTIVE, CONDITIONAL_OBJECTIVE):
        raise ValueError("objective must be 'symmetric_kl' or experimental 'conditional_symmetric_kl'.")


def _self_columns(target, self_columns):
    if self_columns is None:
        if target.shape[0] != target.shape[1]:
            raise ValueError("Conditional row blocks require explicit self_columns.")
        return np.arange(target.shape[0])
    return self_columns


def _numpy_log_probabilities(distance, temperature):
    logits = -np.asarray(distance, dtype=np.float64) / temperature
    return logits - logsumexp(logits, axis=1, keepdims=True)


def _numpy_kl_from_logs(target, logq, *, self_columns=None):
    p = np.asarray(target, dtype=np.float64)
    mass = p.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Objective target rows must have finite positive comparison mass.")
    p = p / mass
    logp = np.log(np.where(p > 0, p, np.finfo(p.dtype).tiny))
    q = np.exp(logq)
    if self_columns is not None:
        # The excluded Q entry is zero; avoid 0 * -inf in the sum.
        logq = logq.copy()
        logq[np.arange(len(p)), self_columns] = 0.
    return float(np.sum(p * (logp - logq) + q * (logq - logp)))


def _numpy_alignment_loss(target, distance, temperature, *, objective=DEFAULT_OBJECTIVE, self_columns=None):
    _validate_objective(objective)
    if objective == DEFAULT_OBJECTIVE:
        return _numpy_kl_from_logs(target, _numpy_log_probabilities(distance, temperature))
    columns = _self_columns(target, self_columns)
    p = np.array(target, dtype=np.float64, copy=True)
    logits = -np.asarray(distance, dtype=np.float64) / temperature
    p[np.arange(len(p)), columns] = 0.
    logits[np.arange(len(p)), columns] = -np.inf
    logq = logits - logsumexp(logits, axis=1, keepdims=True)
    return _numpy_kl_from_logs(p, logq, self_columns=columns)


def _torch_kl_from_logs(target, logq, *, self_columns=None):
    import torch

    mass = target.sum(dim=1, keepdim=True)
    if not torch.isfinite(mass).all() or torch.any(mass <= 0):
        raise ValueError("Objective target rows must have finite positive comparison mass.")
    p = target / mass
    logp = torch.where(p > 0, p, torch.finfo(p.dtype).tiny).log()
    q = logq.exp()
    if self_columns is not None:
        logq = logq.clone()
        logq[torch.arange(len(p), device=p.device), self_columns] = 0.
    return torch.sum(p * (logp - logq) + q * (logq - logp))


def _torch_alignment_loss(target, distance, temperature, *, objective=DEFAULT_OBJECTIVE, self_columns=None):
    import torch

    _validate_objective(objective)
    if objective == DEFAULT_OBJECTIVE:
        return _torch_kl_from_logs(target, torch.log_softmax(-distance / temperature, dim=1))
    columns = torch.as_tensor(_self_columns(target, self_columns), dtype=torch.long, device=target.device)
    rows = torch.arange(len(target), device=target.device)
    p = target.clone()
    p[rows, columns] = 0.
    logits = -distance / temperature
    logits[rows, columns] = -torch.inf
    return _torch_kl_from_logs(p, torch.log_softmax(logits, dim=1), self_columns=columns)
