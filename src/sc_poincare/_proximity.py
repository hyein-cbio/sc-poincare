"""Local neighborhood graph and global forest-accessibility proximities."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

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
    diagnostics: dict[str, float | int]
    graph_mode: str = "mutual"


class ProximityWarning(UserWarning):
    """The chosen feature scale / sigma gives a numerically weak target."""


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


def _resolve_kernel_width(lengths: sparse.csr_matrix, kernel_width: float | str) -> float:
    """Resolve an explicitly requested global bandwidth; never change features."""
    if isinstance(kernel_width, str):
        if kernel_width != "median":
            raise ValueError("sigma / kernel_width must be positive or 'median'.")
        positive = lengths.data[lengths.data > 0.0]
        if not positive.size:
            raise ValueError("Cannot infer median sigma: all graph edge distances are zero. Supply a positive numeric sigma.")
        width = float(np.median(positive))
    else:
        if isinstance(kernel_width, (bool, np.bool_)) or not np.isscalar(kernel_width):
            raise ValueError("sigma / kernel_width must be positive or 'median'.")
        try:
            width = float(kernel_width)
        except (TypeError, ValueError):
            raise ValueError("sigma / kernel_width must be positive or 'median'.") from None
    if not np.isfinite(width) or width <= 0:
        raise ValueError("sigma / kernel_width must be finite and positive.")
    return width


def _validate_graph_mode(graph_mode):
    if not isinstance(graph_mode, str) or graph_mode not in ("mutual", "union"):
        raise ValueError("graph_mode must be 'mutual' or 'union'.")


def local_affinity(
    features: np.ndarray,
    n_neighbors: int,
    kernel_width: float | str,
    metric: str = "euclidean",
    *,
    diagnostics: dict[str, int] | None = None,
    graph_mode: str = "mutual",
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return local affinities and sparse stitched edge lengths (not all pairs).

    If supplied, ``diagnostics`` receives graph-construction counts on success.
    Counts describe the selected graph before Gaussian weighting; explicit
    zero-length edges count as connections, not isolated cells. Collecting
    diagnostics never changes graph construction or kernel weights.

    ``mutual`` (default) requires reciprocal neighbors. Experimental ``union``
    keeps either direction, then uses the same MST stitching and actual Gaussian
    weights. No extra degree normalization or affinity floor is introduced.
    """
    _validate_graph_mode(graph_mode)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("features must be a finite 2-D matrix with at least one feature.")
    n_obs = features.shape[0]
    if n_obs < 3:
        raise ValueError("Need at least 3 observations to build a neighborhood graph.")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if isinstance(kernel_width, str) and kernel_width == "median":
        pass  # Resolve after constructing the graph's edge lengths.
    else:
        _resolve_kernel_width(sparse.csr_matrix((0, 0)), kernel_width)

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
    if graph_mode == "mutual":
        partners_by_cell = _mutual_knn_mask(neighbor_index)
    else:
        partners_by_cell = [set(row.tolist()) for row in neighbor_index]
        for i, neighbors in enumerate(neighbor_index):
            for j in neighbors:
                partners_by_cell[j].add(i)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, partners in enumerate(partners_by_cell):
        for j in partners:
            if j <= i:
                continue
            # In union mode the edge may exist only in j's directed list.
            length = float(neighbor_lengths[i][j] if j in neighbor_lengths[i]
                           else neighbor_lengths[j][i])
            rows.extend((i, j))
            cols.extend((j, i))
            data.extend((length, length))

    length_graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_obs)).tocsr()
    if diagnostics is not None:
        n_components_before = int(connected_components(length_graph, directed=False)[0])
        n_isolated_before = int(np.count_nonzero(np.diff(length_graph.indptr) == 0))
    stitched = _stitch_components(length_graph, features, metric=metric)
    width = _resolve_kernel_width(stitched, kernel_width)

    # Paper Eq. 1: ALL edges, including bridges, use their actual distance.
    # Preserve explicit zero distances (duplicates -> affinity 1), but never
    # manufacture a bridge strength to hide an unsuitable feature scale.
    affinity = stitched.copy()
    with np.errstate(over="ignore", under="ignore"):
        affinity.data = np.exp(-0.5 * (affinity.data / width)**2)
    n_underflow = int(np.count_nonzero(affinity.data == 0.0))
    affinity.eliminate_zeros()
    n_parts = connected_components(affinity, directed=False)[0]
    if n_parts > 1:
        raise ValueError(
            f"Gaussian affinity underflow disconnected the graph into {n_parts} components. "
            f"Resolved sigma = {width:.6g}. Check feature units/preprocessing or increase sigma explicitly; "
            "bridge weights are not artificially floored."
        )
    if n_underflow:
        warnings.warn(
            f"{n_underflow // 2} undirected Gaussian edge weights underflowed to zero. "
            "The graph is still connected; inspect feature scale and sigma.",
            ProximityWarning, stacklevel=2,
        )
    if diagnostics is not None:
        diagnostics.update(
            n_components_before_stitching=n_components_before,
            n_isolated_before_stitching=n_isolated_before,
            n_bridges_added=int((stitched.nnz - length_graph.nnz) // 2),
        )
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
    kernel_width: float | str,
    metric: str = "euclidean",
    *,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    graph_mode: str = "mutual",
) -> ManifoldProximity:
    _validate_graph_mode(graph_mode)
    check_dense_budget(features.shape[0], max_dense_bytes)
    graph_diagnostics: dict[str, int] = {}
    weights, lengths = local_affinity(
        features, n_neighbors, kernel_width, metric=metric, diagnostics=graph_diagnostics,
        graph_mode=graph_mode,
    )
    width = _resolve_kernel_width(lengths, kernel_width)
    accessibility = forest_accessibility(weights, max_dense_bytes=max_dense_bytes)
    self_mass = np.diag(accessibility)
    n_self_dominated = int(np.count_nonzero(self_mass > 0.99))
    diagnostics = {
        **graph_diagnostics,
        "n_obs": int(weights.shape[0]),
        "n_edges": int(weights.nnz // 2),
        "edge_weight_min": float(weights.data.min()),
        "edge_weight_median": float(np.median(weights.data)),
        "edge_weight_max": float(weights.data.max()),
        "edge_distance_median": float(np.median(lengths.data)),
        "edge_distance_over_sigma_median": float(np.median(lengths.data) / width),
        "sigma_effective": width,
        "self_mass_median": float(np.median(self_mass)),
        "self_mass_gt_0_99_count": n_self_dominated,
        "self_mass_gt_0_99_fraction": float(n_self_dominated / weights.shape[0]),
    }
    if diagnostics["self_mass_gt_0_99_fraction"] >= 0.1:
        warnings.warn(
            f"RFA is self-dominated: {n_self_dominated}/{weights.shape[0]} cells assign "
            "more than 99% mass to themselves. This weakens the trajectory signal. "
            f"Median edge distance / sigma = {diagnostics['edge_distance_over_sigma_median']:.3g}. "
            "Inspect feature units/preprocessing and sigma before interpreting the embedding. "
            "No automatic rescaling has been applied.",
            ProximityWarning, stacklevel=2,
        )
    return ManifoldProximity(
        weights=weights,
        accessibility=accessibility,
        metric=metric,
        n_neighbors=n_neighbors,
        kernel_width=width,
        diagnostics=diagnostics,
        graph_mode=graph_mode,
    )
