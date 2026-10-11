from __future__ import annotations

import numpy as np
from scipy import sparse

from sc_poincare._fitting import _seed_from_connectivities


def test_connectivity_seed_keeps_a_tight_block_on_one_side():
    left = np.ones((6, 6))
    right = np.ones((6, 6))
    graph = sparse.block_diag((sparse.csr_matrix(left), sparse.csr_matrix(right)), format="lil")
    # One bridge, as a stitched connectivity graph would have.
    graph[5, 6] = 0.2
    graph[6, 5] = 0.2
    graph = graph.tocsr()
    coords = _seed_from_connectivities(graph, 2, np.random.default_rng(0))
    within = np.linalg.norm(coords[:6, None, :] - coords[None, :6, :], axis=-1)
    across = np.linalg.norm(coords[:6, None, :] - coords[None, 6:, :], axis=-1)
    assert within[np.triu_indices(6, 1)].mean() < across.mean()
