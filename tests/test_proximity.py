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


@pytest.mark.parametrize(
    "features,k,expected",
    [
        (np.array([[0.], [1.], [2.]]), 2, (1, 0, 0)),
        (np.array([[0.], [.1], [1.], [1.1]]), 1, (2, 0, 1)),
        (np.array([[0.], [1.], [3.]]), 1, (2, 1, 1)),
        (np.array([[0.], [.1], [1.], [1.1], [3.]]), 1, (3, 1, 2)),
        # Explicit zero-length edges connect duplicate cells.
        (np.zeros((4, 2)), 3, (1, 0, 0)),
        # k is capped to n_obs - 1, as in the existing graph builder.
        (np.zeros((4, 2)), 20, (1, 0, 0)),
    ],
    ids=["connected", "two-components", "isolated", "multiple-bridges", "duplicates", "capped-k"],
)
def test_graph_construction_diagnostics(features, k, expected):
    original = features.copy()
    plain_weights, plain_lengths = local_affinity(features, k, 2.)
    diagnostics = {}
    weights, lengths = local_affinity(features, k, 2., diagnostics=diagnostics)
    keys = ("n_components_before_stitching", "n_isolated_before_stitching", "n_bridges_added")
    assert tuple(diagnostics[key] for key in keys) == expected
    assert all(type(diagnostics[key]) is int for key in keys)
    assert diagnostics['n_bridges_added'] == diagnostics['n_components_before_stitching'] - 1
    assert connected_components(weights, directed=False)[0] == 1
    # Reading topology must not mutate features, drop zero-length edges, or
    # change the sparse graph's structure, order or values.
    np.testing.assert_array_equal(features, original)
    for actual, plain in [(weights, plain_weights), (lengths, plain_lengths)]:
        for attribute in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(actual, attribute), getattr(plain, attribute))
    prox = estimate_proximity(features, k, 2.)
    assert all(prox.diagnostics[key] == diagnostics[key] for key in keys)
    np.testing.assert_array_equal(prox.accessibility, forest_accessibility(plain_weights))


def test_failed_graph_does_not_publish_partial_diagnostics():
    diagnostics = {'existing': 7}
    with pytest.raises(ValueError, match='underflow'):
        local_affinity(np.array([[0.], [100.], [200.]]), 1, 1., diagnostics=diagnostics)
    assert diagnostics == {'existing': 7}


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
