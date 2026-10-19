"""LEGACY ONLY: retired NumPy probability-space convenience wrappers.

Preserved for historical inspection, not imported by fitting, initialization,
AnnData routing or the public tools API. Current numerical paths use log-domain
objectives directly. No redirect aliases are kept in _fitting.
"""

from __future__ import annotations

import numpy as np

from sc_poincare._objectives import _numpy_kl_from_logs, _numpy_log_probabilities

# BEGIN LEGACY NUMPY PROBABILITY WRAPPERS (no production consumers)

def _row_softmax_neg_distance(distance: np.ndarray, temperature: float) -> np.ndarray:
    return np.exp(_numpy_log_probabilities(distance, temperature))


def _sym_kl(target: np.ndarray, predicted: np.ndarray) -> float:
    q = np.asarray(predicted, dtype=np.float64)
    q = q / q.sum(axis=1, keepdims=True)
    logq = np.log(np.where(q > 0, q, np.finfo(q.dtype).tiny))
    return _numpy_kl_from_logs(target, logq)

# END LEGACY NUMPY PROBABILITY WRAPPERS
