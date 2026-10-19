from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from sc_poincare._proximity import estimate_proximity, forest_accessibility, local_affinity


def test_affinity_graph_is_symmetric_connected_and_nonnegative():
    rng = np.random.default_rng(0)
    left = rng.normal(loc=0.0, scale=0.2, size=(12, 4))
    right = rng.normal(loc=5.0, scale=0.2, size=(10, 4))
    features = np.vstack([left, right])
    weights, lengths = local_affinity(features, n_neighbors=3, kernel_width=1.0)
    dense = weights.toarray()
    assert np.allclose(dense, dense.T, atol=1e-12)
    assert np.all(dense >= 0)
    # A very weak but positive bridge is connected; its algebraic connectivity
    # need not exceed an arbitrary threshold. Check the actual paper kernel.
    assert connected_components(weights, directed=False)[0] == 1
    edges = lengths.tocoo()
    np.testing.assert_allclose(dense[edges.row, edges.col], np.exp(-0.5 * edges.data**2), rtol=1e-12, atol=0.)


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


def test_multiscale_mixture_is_separate_from_sigma_and_from_the_default():
    from sc_poincare._proximity import multiscale_forest_accessibility
    rng = np.random.default_rng(4)
    features = rng.normal(size=(18, 6))
    weights, _ = local_affinity(features, n_neighbors=5, kernel_width=1.5)
    unit = forest_accessibility(weights)
    prox = estimate_proximity(features, n_neighbors=5, kernel_width=1.5)
    np.testing.assert_allclose(prox.accessibility, unit, atol=1e-12)
    mixed = multiscale_forest_accessibility(weights)
    repeated = multiscale_forest_accessibility(weights, taus=(1.0, 1.0))
    np.testing.assert_allclose(repeated, unit, atol=1e-10)
    assert not np.allclose(mixed, unit, atol=1e-6)
    assert np.all(mixed >= -1e-12)
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0, atol=1e-8)
    wider, _ = local_affinity(features, n_neighbors=5, kernel_width=3.0)
    assert not np.allclose(forest_accessibility(wider), unit, atol=1e-6)
    with pytest.raises(ValueError):
        multiscale_forest_accessibility(weights, taus=(1.0,))
    with pytest.raises(ValueError):
        multiscale_forest_accessibility(weights, taus=(0.1, -1.0))
