"""Local neighborhood graph and global forest-accessibility proximities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES, check_dense_budget


@dataclass(frozen=True)
class ManifoldProximity:
    """Locally weighted neighborhood plus the dense accessibility matrix."""

    weights: sparse.csr_matrix
    accessibility: np.ndarray
    metric: str
    n_neighbors: int
    kernel_width: float


def _union_find(n: int) -> tuple[np.ndarray, callable, callable]:
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> bool:
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1
        return True

    return parent, find, union


def _mutual_knn_mask(neighbor_index: np.ndarray) -> list[set[int]]:
    """``neighbor_index[i]`` lists the k nearest others of i (no self)."""
    bags = [set(row.tolist()) for row in neighbor_index]
    mutual = []
    for i, bag in enumerate(bags):
        kept = {j for j in bag if i in bags[j]}
        mutual.append(kept)
    return mutual


def _stitch_components(
    adjacency: sparse.csr_matrix,
    features: np.ndarray,
    metric: str = "euclidean",
    block_size: int = 256,
) -> sparse.csr_matrix:
    """Exact component MST bridges via bounded-memory Borůvka rounds.

    Each component proposes its shortest outgoing edge. This preserves the
    minimum total bridge length without allocating and sorting all pairs.
    Distance work can still be quadratic; it is skipped for connected graphs.
    """
    n = adjacency.shape[0]
    n_parts, labels = connected_components(adjacency, directed=False)
    if n_parts <= 1:
        return adjacency.tocsr()
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    _, find, union = _union_find(n_parts)
    extra_i: list[int] = []
    extra_j: list[int] = []
    extra_d: list[float] = []
    while n_parts > 1:
        components = np.array([find(int(label)) for label in labels])
        candidates: dict[int, tuple[float, int, int]] = {}
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            distances = pairwise_distances(features[start:stop], features, metric=metric)
            distances[components[start:stop, None] == components[None, :]] = np.inf
            nearest = np.argmin(distances, axis=1)
            for offset, j in enumerate(nearest):
                i = start + offset
                distance = float(distances[offset, j])
                if not np.isfinite(distance):
                    continue
                edge = (distance, min(i, int(j)), max(i, int(j)))
                component = int(components[i])
                if component not in candidates or edge < candidates[component]:
                    candidates[component] = edge
        added = 0
        for distance, i, j in sorted(set(candidates.values())):
            if union(int(labels[i]), int(labels[j])):
                extra_i.extend((i, j))
                extra_j.extend((j, i))
                extra_d.extend((distance, distance))
                n_parts -= 1
                added += 1
        if not added:
            raise ValueError("Cannot connect graph components with finite distances.")
    # Construct directly: sparse addition can drop explicit zero-length bridges.
    edges = adjacency.tocoo()
    return sparse.coo_matrix(
        (np.concatenate([edges.data, extra_d]),
         (np.concatenate([edges.row, extra_i]), np.concatenate([edges.col, extra_j]))),
        shape=(n, n),
    ).tocsr()


def local_affinity(
    features: np.ndarray,
    n_neighbors: int,
    kernel_width: float,
    metric: str = "euclidean",
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return local affinities and sparse stitched edge lengths (not all pairs)."""
    features = np.asarray(features, dtype=np.float64)
    n_obs = features.shape[0]
    if n_obs < 3:
        raise ValueError("Need at least 3 observations to build a neighborhood graph.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive.")
    if kernel_width <= 0:
        raise ValueError("sigma / kernel_width must be positive.")

    k = min(n_neighbors, n_obs - 1)
    knn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    knn.fit(features)
    distances, indices = knn.kneighbors(features)
    # Remove self by index, not position: duplicate cells can precede self.
    neighbor_index = np.empty((n_obs, k), dtype=np.int64)
    neighbor_lengths = []
    for i in range(n_obs):
        keep = indices[i] != i
        neighbor_index[i] = indices[i, keep][:k]
        neighbor_lengths.append(dict(zip(neighbor_index[i], distances[i, keep][:k])))
    mutual = _mutual_knn_mask(neighbor_index)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, partners in enumerate(mutual):
        for j in partners:
            if j <= i:
                continue
            length = float(neighbor_lengths[i][j])
            rows.extend((i, j))
            cols.extend((j, i))
            data.extend((length, length))

    length_graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_obs)).tocsr()
    stitched = _stitch_components(length_graph, features, metric=metric)

    # Kernelize local edges first. Bridges added to restore connectivity can sit
    # far beyond σ and would otherwise vanish numerically, leaving the graph
    # disconnected for all practical purposes. Give those bridges a weight just
    # below the weakest local affinity so RFA still sees a single component.
    local = length_graph.tocoo()
    local_w = np.exp(-(local.data**2) / (2.0 * kernel_width**2))
    affinity = sparse.coo_matrix(
        (local_w, (local.row, local.col)),
        shape=(n_obs, n_obs),
    ).tocsr()

    stitched_topology = stitched.copy()
    stitched_topology.data[:] = 1.0
    local_topology = length_graph.copy()
    local_topology.data[:] = 1.0
    bridge = stitched_topology - local_topology
    bridge.eliminate_zeros()
    if bridge.nnz:
        floor = float(np.min(local_w)) if local_w.size else 1e-6
        bridge = bridge.tocoo()
        bridge.data = np.full(bridge.data.shape, 0.5 * max(floor, 1e-6))
        affinity = (affinity + bridge.tocsr()).tocsr()

    return affinity, stitched


def forest_accessibility(
    affinity: sparse.spmatrix,
    *,
    block_size: int = 128,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
) -> np.ndarray:
    """Exact ``P = (I + L)^{-1}`` with blocked RHS solves; output is dense."""
    if affinity.shape[0] != affinity.shape[1] or affinity.shape[0] == 0:
        raise ValueError("affinity must be a nonempty square matrix.")
    check_dense_budget(affinity.shape[0], max_dense_bytes)
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    affinity = affinity.tocsr().astype(np.float64)
    if not np.all(np.isfinite(affinity.data)) or np.any(affinity.data < 0):
        raise ValueError("affinity must be finite and nonnegative.")
    n_obs = affinity.shape[0]
    degree = np.asarray(affinity.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree) - affinity
    shifted = (sparse.eye(n_obs, format="csc") + laplacian.tocsc()).tocsc()
    factorization = splu(shifted)
    accessibility = np.empty((n_obs, n_obs), dtype=np.float64)
    for start in range(0, n_obs, block_size):
        stop = min(start + block_size, n_obs)
        rhs = np.zeros((n_obs, stop - start), dtype=np.float64)
        rhs[np.arange(start, stop), np.arange(stop - start)] = 1.0
        accessibility[:, start:stop] = factorization.solve(rhs)
    # Tiny negative values can appear from round-off; they break the KL.
    np.maximum(accessibility, 0.0, out=accessibility)
    row_mass = accessibility.sum(axis=1, keepdims=True)
    accessibility /= np.clip(row_mass, a_min=1e-12, a_max=None)
    return accessibility


def estimate_proximity(
    features: np.ndarray,
    n_neighbors: int,
    kernel_width: float,
    metric: str = "euclidean",
    *,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
) -> ManifoldProximity:
    check_dense_budget(features.shape[0], max_dense_bytes)
    weights, _ = local_affinity(features, n_neighbors, kernel_width, metric=metric)
    accessibility = forest_accessibility(weights, max_dense_bytes=max_dense_bytes)
    return ManifoldProximity(
        weights=weights,
        accessibility=accessibility,
        metric=metric,
        n_neighbors=n_neighbors,
        kernel_width=kernel_width,
    )
