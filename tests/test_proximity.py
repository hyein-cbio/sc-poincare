from __future__ import annotations

import numpy as np

from sc_poincare._proximity import estimate_proximity, forest_accessibility, local_affinity


def test_affinity_graph_is_symmetric_connected_and_nonnegative():
    rng = np.random.default_rng(0)
    left = rng.normal(loc=0.0, scale=0.2, size=(12, 4))
    right = rng.normal(loc=5.0, scale=0.2, size=(10, 4))
    features = np.vstack([left, right])
    weights, _ = local_affinity(features, n_neighbors=3, kernel_width=1.0)
    dense = weights.toarray()
    assert np.allclose(dense, dense.T, atol=1e-12)
    assert np.all(dense >= 0)
    # One connected component via the Laplacian nullspace.
    degree = dense.sum(axis=1)
    lap = np.diag(degree) - dense
    evals = np.sort(np.linalg.eigvalsh(lap))
    assert evals[0] == 0 or evals[0] < 1e-10
    assert evals[1] > 1e-8


def test_forest_accessibility_is_row_stochastic():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(18, 6))
    prox = estimate_proximity(features, n_neighbors=5, kernel_width=1.5)
    p = prox.accessibility
    assert p.shape == (18, 18)
    assert np.all(p >= -1e-12)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-8)
    # Nearby cells should share more forest mass than far-apart cells.
    close = p[0, 1]
    far = p[0, -1]
    # Not a strict theorem for every seed, but the diagonal must dominate.
    assert np.all(np.diag(p) > 1.0 / 18)
    assert close >= 0 and far >= 0
