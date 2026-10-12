"""Numerically stable row-wise symmetric KL objectives.

Positive target probabilities, including subnormals, are not floored. Exact
zeros use the dtype's smallest normal positive value *in logarithms only*;
this is a finite numerical regularization of otherwise infinite reverse KL,
not probability smoothing or an exact divergence when zeros are present.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _numpy_log_probabilities(distance, temperature):
    logits = -np.asarray(distance, dtype=np.float64) / temperature
    return logits - logsumexp(logits, axis=1, keepdims=True)


def _numpy_kl_from_logs(target, logq):
    p = np.asarray(target, dtype=np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    logp = np.log(np.where(p > 0, p, np.finfo(p.dtype).tiny))
    q = np.exp(logq)
    return float(np.sum(p * (logp - logq) + q * (logq - logp)))


def _numpy_alignment_loss(target, distance, temperature):
    return _numpy_kl_from_logs(target, _numpy_log_probabilities(distance, temperature))


def _torch_kl_from_logs(target, logq):
    import torch

    p = target / target.sum(dim=1, keepdim=True)
    logp = torch.where(p > 0, p, torch.finfo(p.dtype).tiny).log()
    q = logq.exp()
    return torch.sum(p * (logp - logq) + q * (logq - logp))


def _torch_alignment_loss(target, distance, temperature):
    import torch

    return _torch_kl_from_logs(target, torch.log_softmax(-distance / temperature, dim=1))
