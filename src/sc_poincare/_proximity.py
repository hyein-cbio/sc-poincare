"""Local neighborhood graph and global forest-accessibility proximities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


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
    sample_distance: np.ndarray,
) -> sparse.csr_matrix:
    """Add the shortest inter-component links until the graph is connected."""
    n = adjacency.shape[0]
    _, find, union = _union_find(n)
    coo = adjacency.tocoo()
    n_parts = n
    for i, j in zip(coo.row.tolist(), coo.col.tolist()):
        if i < j and union(i, j):
            n_parts -= 1
    if n_parts <= 1:
        return adjacency.tocsr()

    tri_i, tri_j = np.triu_indices(n, k=1)
    order = np.argsort(sample_distance[tri_i, tri_j], kind="mergesort")
    extra_i: list[int] = []
    extra_j: list[int] = []
    extra_d: list[float] = []
    for pos in order:
        i = int(tri_i[pos])
        j = int(tri_j[pos])
        if union(i, j):
            extra_i.extend((i, j))
            extra_j.extend((j, i))
            extra_d.extend((float(sample_distance[i, j]), float(sample_distance[i, j])))
            n_parts -= 1
            if n_parts <= 1:
                break

    if not extra_i:
        return adjacency.tocsr()
    extra = sparse.coo_matrix((extra_d, (extra_i, extra_j)), shape=(n, n))
    return (adjacency + extra).tocsr()


def local_affinity(
    features: np.ndarray,
    n_neighbors: int,
    kernel_width: float,
    metric: str = "euclidean",
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Connected mutual kNN graph with a Gaussian kernel on edge lengths."""
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
    neighbor_index = knn.kneighbors(features, return_distance=False)[:, 1:]
    mutual = _mutual_knn_mask(neighbor_index)

    sample_distance = pairwise_distances(features, metric=metric)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, partners in enumerate(mutual):
        for j in partners:
            if j <= i:
                continue
            length = float(sample_distance[i, j])
            rows.extend((i, j))
            cols.extend((j, i))
            data.extend((length, length))

    length_graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_obs)).tocsr()
    stitched = _stitch_components(length_graph, sample_distance)

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

    bridge = (stitched > 0).astype(np.float64) - (length_graph > 0).astype(np.float64)
    bridge.eliminate_zeros()
    if bridge.nnz:
        floor = float(np.min(local_w)) if local_w.size else 1e-6
        bridge = bridge.tocoo()
        bridge.data = np.full(bridge.data.shape, 0.5 * max(floor, 1e-6))
        affinity = (affinity + bridge.tocsr()).tocsr()

    return affinity, sample_distance


def forest_accessibility(affinity: sparse.spmatrix) -> np.ndarray:
    """Relative forest accessibility ``P = (I + L)^{-1}``, ``L = D - A``."""
    affinity = affinity.tocsr().astype(np.float64)
    n_obs = affinity.shape[0]
    degree = np.asarray(affinity.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree) - affinity
    shifted = (sparse.eye(n_obs, format="csc") + laplacian.tocsc()).tocsc()
    identity = np.eye(n_obs, dtype=np.float64)
    accessibility = np.asarray(spsolve(shifted, identity), dtype=np.float64)
    if accessibility.ndim == 1:
        accessibility = accessibility.reshape(n_obs, n_obs)
    # Tiny negative values can appear from round-off; they break the KL.
    accessibility = np.clip(accessibility, 0.0, None)
    row_mass = accessibility.sum(axis=1, keepdims=True)
    accessibility = accessibility / np.clip(row_mass, a_min=1e-12, a_max=None)
    return accessibility


def estimate_proximity(
    features: np.ndarray,
    n_neighbors: int,
    kernel_width: float,
    metric: str = "euclidean",
) -> ManifoldProximity:
    weights, _ = local_affinity(features, n_neighbors, kernel_width, metric=metric)
    accessibility = forest_accessibility(weights)
    return ManifoldProximity(
        weights=weights,
        accessibility=accessibility,
        metric=metric,
        n_neighbors=n_neighbors,
        kernel_width=kernel_width,
    )
