"""Opt-in exact graph-stress experiments; the public RFA workflow is unchanged.

These input-only methods have limited development-data evidence, not a claim of
general biological improvement or uniquely hyperbolic benefit. Targets and
classical scaling require quadratic storage despite row-chunked optimization.
Torch is required only for fitting. No labels, reference coordinates or roots
are accepted by the model API; callers must explicitly preprocess features.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import numbers
import operator
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

from sc_poincare._proximity import _stitch_components, local_affinity
from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES, check_dense_budget


def _integer(name, value, minimum=1):
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from None
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _real(name, value, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return float(value)


@dataclass(frozen=True)
class GraphStressConfig:
    """Explicit experimental recipe; ``global`` retains the fixed MU control.

    ``local_radius`` changes only the union-graph Gaussian bandwidth to
    sqrt(r_i*r_j), with r_i the kth nonself feature-neighbor distance. Radii
    are floored at 1e-6 times their positive median. Empty teacher rows are
    rejected, not silently repaired. Changing these parameters defines a new
    experimental condition, not a previously validated quality result.
    """

    n_neighbors: int = 15
    local_bandwidth: str = "global"
    distance_scale: float = 2.0
    gamma: float = .1 * 2 / 3
    local_weight: float = .03
    relative_floor: float = .01
    initial_radius: float = .2
    jitter: float = .0002
    learning_rate: float = .02 * 2 / 3

    def __post_init__(self):
        object.__setattr__(self, "n_neighbors", _integer("n_neighbors", self.n_neighbors))
        if self.local_bandwidth not in ("global", "local_radius"):
            raise ValueError("local_bandwidth must be 'global' or 'local_radius'.")
        for name in ("distance_scale", "gamma", "relative_floor", "initial_radius", "learning_rate"):
            object.__setattr__(self, name, _real(name, getattr(self, name)))
        for name in ("local_weight", "jitter"):
            object.__setattr__(self, name, _real(name, getattr(self, name), allow_zero=True))
        if self.initial_radius >= 1:
            raise ValueError("initial_radius must be strictly below 1.")


@dataclass(frozen=True)
class GraphStressSetup:
    """Prepared input-only target, teacher and disk initializer.

    Arrays are read-only snapshots. Fitting rechecks their hashes; mutating a
    setup or using it after changing its numerical implementation is rejected.
    """

    target: np.ndarray
    teacher: np.ndarray
    base_initial: np.ndarray
    config: GraphStressConfig
    metadata: dict


@dataclass(frozen=True)
class GraphStressProgress:
    """Independent copies at an evaluated epoch, before its Adam update.

    ``state`` includes Adam moments and can be saved with ``torch.save`` and
    loaded with ``torch.load(..., weights_only=True)``. Callback mutations do
    not change the running fit. Resume uses total desired epochs, not an
    additional-epoch count.
    """

    epoch: int
    loss: float
    global_stress: float
    local_ce: float
    current: np.ndarray
    best: np.ndarray
    best_loss: float
    best_epoch: int
    state: dict


@dataclass
class GraphStressResult:
    """Own-objective best and final coordinates; epoch zero is eligible.

    ``initial`` is the disk initialization in both geometries. The Euclidean
    control starts at ``initial_tangent`` instead. Equal tangent arrays do not
    imply identical initial pair distances or equal optimization convergence.
    """

    coordinates: np.ndarray
    current: np.ndarray
    initial: np.ndarray
    initial_tangent: np.ndarray
    loss_trace: np.ndarray
    n_epochs: int
    geometry: str
    best_loss: float
    best_epoch: int
    current_loss: float
    state: dict


def _digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _source_hashes():
    folder = Path(__file__).parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("experimental.py", "_proximity.py", "_fitting.py", "_disk.py")}


def _dense_budget(n, limit):
    try:
        check_dense_budget(n, limit)
    except MemoryError as exc:
        raise MemoryError(
            "Exact experimental graph stress exceeds the shared dense allocation "
            "budget. Row chunks do not remove its O(N²) targets/initialization. "
            "Subsample or explicitly increase max_dense_bytes."
        ) from exc


def _graph_lengths(features, k, topology):
    if topology == "mutual":
        width = 2 * float(np.linalg.norm(features, axis=1).max()) + 1
        if not np.isfinite(width):
            raise ValueError("Feature magnitudes overflow Euclidean graph distances.")
        return local_affinity(features, k, width)[1]
    distances, indices = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(features).kneighbors(features)
    edges = {}
    for i in range(len(features)):
        valid = np.flatnonzero(indices[i] != i)[:k]
        for q in valid:
            j = int(indices[i, q])
            pair = min(i, j), max(i, j)
            edges[pair] = min(edges.get(pair, np.inf), float(distances[i, q]))
    rows, columns, values = [], [], []
    for (i, j), distance in edges.items():
        rows.extend([i, j]); columns.extend([j, i]); values.extend([distance, distance])
    return _stitch_components(coo_matrix((values, (rows, columns)),
                                       shape=(len(features), len(features))).tocsr(), features)


def _teacher_diagnostics(teacher):
    """Describe a row-stochastic zero-diagonal teacher without changing it.

    Entropy is in nats. Its mean lower-bounds nonself cross entropy over
    probability distributions, but need not be attainable by the geometry.
    Incoming mass and asymmetry describe normalization, not population balance.
    """
    n = len(teacher)
    row_mass = teacher.sum(axis=1)
    if not np.allclose(row_mass, 1., rtol=0., atol=1e-12) or np.any(np.diag(teacher) != 0.):
        raise ValueError('A normalized nonself teacher is required for entropy diagnostics.')
    entropy = np.empty(n)
    max_asymmetry = 0.
    for start in range(0, n, 128):
        rows = teacher[start:start+128]
        if not np.isfinite(rows).all() or np.any(rows < 0.):
            raise ValueError('A finite nonnegative teacher is required for entropy diagnostics.')
        max_asymmetry = max(max_asymmetry, float(np.max(np.abs(rows-teacher[:, start:start+128].T))))
        for offset, row in enumerate(rows):
            positive = row[row > 0.]
            entropy[start+offset] = -float(np.dot(positive, np.log(positive)))
    levels = [0., .1, .5, .9, 1.]
    return dict(teacher_diagnostic_quantiles=levels, teacher_entropy_mean=float(entropy.mean()),
        teacher_entropy_quantiles=np.quantile(entropy, levels).tolist(),
        teacher_incoming_mass_quantiles=np.quantile(teacher.sum(axis=0), levels).tolist(),
        teacher_max_asymmetry=max_asymmetry,
        teacher_max_row_sum_error=float(np.max(np.abs(row_mass-1.))))



def prepare_graph_stress(features, *, config=None,
                         max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
    """Prepare exact mutual-geodesic stress and a union-neighbor teacher.

    Mutual edge lengths become log1p(length/positive-edge-median). Shortest
    paths are normalized to a nonself mean of ``distance_scale``. Global
    teacher bandwidth is max(q95_positive_edge, maxedge/5). The alternative
    local-radius teacher records its floor and underflow diagnostics. Metadata
    also describes teacher entropy (nats), incoming mass and normalization;
    these input-only statistics do not change targets or fitting.

    ``features`` must be a finite real N×D matrix with N > n_neighbors.
    Labels must be removed before calling this function or computing PCA.
    The shared budget estimates four float64 N×N arrays, not total RSS.
    ``None`` explicitly disables that estimate; it is not a scalability claim.
    """
    config = GraphStressConfig() if config is None else config
    if not isinstance(config, GraphStressConfig):
        raise TypeError("config must be a GraphStressConfig.")
    raw = np.asarray(features)
    if raw.ndim != 2 or raw.shape[1] == 0 or raw.shape[0] <= config.n_neighbors:
        raise ValueError("features must be N×D with D > 0 and N > n_neighbors.")
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError("features must contain real numeric values, with labels excluded.")
    _dense_budget(len(raw), max_dense_bytes)
    f = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(f).all():
        raise ValueError("features must be finite.")
    lengths = _graph_lengths(f, config.n_neighbors, "mutual")
    positive = lengths.data[lengths.data > 0]
    if not len(positive) or not np.isfinite(positive).all():
        raise ValueError("A nondegenerate graph with finite positive edge lengths is required.")
    median = float(np.median(positive))
    transformed = lengths.copy()
    transformed.data = np.log1p(transformed.data / median)
    geodesic = dijkstra(transformed, directed=False)
    n = len(f)
    mean = float(geodesic.sum() / (n * (n - 1)))
    if not np.isfinite(geodesic).all() or not np.isfinite(mean) or mean <= 0:
        raise ValueError("Finite geodesics with a positive nonself mean are required.")
    squared = geodesic * geodesic
    gram = -.5 * (squared - squared.mean(0) - squared.mean(1, keepdims=True) + squared.mean())
    values, vectors = eigh(gram, subset_by_index=[n - 2, n - 1])
    initial = vectors[:, ::-1] * np.sqrt(np.maximum(values[::-1], 0))
    radius = np.linalg.norm(initial, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("A nondegenerate classical-scaling initialization is required.")
    initial *= config.initial_radius / radius
    del squared, gram
    local = _graph_lengths(f, config.n_neighbors, "union")
    positive = local.data[local.data > 0]
    sigma = max(float(np.quantile(positive, .95)), float(positive.max() / 5))
    weights = local.copy()
    weights.data = np.exp(-.5 * (weights.data / sigma)**2)
    teacher = weights.toarray()
    teacher /= teacher.sum(1, keepdims=True)
    diagnostics = dict(local_sigma=sigma)
    if config.local_bandwidth == "local_radius":
        control_hash = _digest(teacher)
        distances, indices = NearestNeighbors(n_neighbors=config.n_neighbors + 1, n_jobs=1).fit(f).kneighbors(f)
        radii = np.array([distances[i, np.flatnonzero(indices[i] != i)[:config.n_neighbors][-1]]
                          for i in range(n)])
        positive = radii[radii > 0]
        if not len(positive):
            raise ValueError("A nonzero local neighbor radius is required.")
        floor = float(np.median(positive) * 1e-6)
        zero_count = int(np.sum(radii <= 0))
        radii = np.maximum(radii, floor)
        graph = local.tocoo()
        exponent = -.5 * graph.data**2 / (radii[graph.row] * radii[graph.col])
        weight = np.exp(exponent)
        teacher = np.zeros((n, n), dtype=np.float64)
        teacher[graph.row, graph.col] = weight
        row_sum = teacher.sum(1)
        if np.any(row_sum <= 0) or not np.isfinite(row_sum).all():
            raise FloatingPointError("Local-radius teacher has an empty or nonfinite row; rescale/review the input.")
        teacher /= row_sum[:, None]
        diagnostics = dict(control_teacher_sha256=control_hash, control_local_sigma=sigma,
                           radius_floor=floor, zero_radius_count=zero_count,
                           underflow_edge_count=int(np.sum(weight == 0)),
                           minimum_exponent=float(exponent.min()),
                           radius_quantiles=np.quantile(radii, [0, .1, .5, .9, 1]).tolist())
    target = geodesic / mean * config.distance_scale
    np.fill_diagonal(target, 0)
    metadata = dict(n_cells=n, n_features=f.shape[1], feature_sha256=_digest(f),
                    target_sha256=_digest(target), teacher_sha256=_digest(teacher),
                    base_initial_sha256=_digest(initial), edge_median=median,
                    global_mean=mean, global_edges=lengths.nnz // 2,
                    local_edges=local.nnz // 2, n_neighbors=config.n_neighbors,
                    parameters=asdict(config), source_hashes=_source_hashes(), **diagnostics)
    for array in (target, teacher, initial):
        if not np.isfinite(array).all():
            raise FloatingPointError("Prepared target, teacher and initializer must be finite.")
        array.setflags(write=False)
    metadata.update(_teacher_diagnostics(teacher))
    return GraphStressSetup(target, teacher, initial, config, metadata)


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("Experimental graph-stress fitting requires sc-poincare[torch].") from exc


def _disk_from_tangent(u):
    torch = _torch()
    radius = torch.linalg.vector_norm(u, dim=1, keepdim=True)
    factor = torch.where(radius > 1e-7, torch.tanh(radius / 2) / radius.clamp_min(1e-15),
                         .5 - radius.square() / 24 + radius.pow(4) / 240)
    return u * factor


def _tangent_from_disk(x):
    radius = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(radius >= 1):
        raise ValueError("Initialization including jitter must lie in the open unit disk.")
    return x * np.divide(2 * np.arctanh(radius), radius,
                         out=np.full_like(radius, 2.), where=radius > 0)


def _lorentz_spatial_from_disk(disk):
    """Spatial coordinates of the hyperboloid point isometric to a disk point."""
    disk = np.asarray(disk, dtype=np.float64)
    radius_sq = np.sum(disk * disk, axis=1, keepdims=True)
    if not np.isfinite(disk).all() or np.any(radius_sq >= 1):
        raise ValueError("Lorentz initialization must lie in the open unit disk.")
    return 2 * disk / (1 - radius_sq)


def _disk_from_lorentz(u):
    """Inverse of the disk-to-hyperboloid spatial map."""
    torch = _torch()
    time = torch.sqrt(1 + u.square().sum(dim=1, keepdim=True))
    return u / (time + 1)


def _lorentz_distance(rows, points, self_columns):
    """Hyperboloid distance acosh(-<x,y>). Diagonal entries are exactly zero."""
    torch = _torch()
    time_rows = torch.sqrt(1 + rows.square().sum(dim=1))
    time_points = torch.sqrt(1 + points.square().sum(dim=1))
    minkowski = -time_rows[:, None] * time_points[None, :] + rows @ points.T
    distance = torch.acosh(torch.clamp(-minkowski, min=1))
    distance = distance.clone()
    distance[torch.arange(distance.shape[0], device=distance.device), self_columns] = 0
    return distance


def _objective(u, target, teacher, config, geometry, chunk_size, *, backward):
    """Exact all-column objective; each row chunk contributes both endpoints."""
    torch = _torch()
    from sc_poincare._fitting import _torch_disk_length
    n = len(u)
    u.grad = None
    global_total, local_total = 0., 0.
    with torch.set_grad_enabled(backward):
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            rows = torch.arange(start, stop)
            if geometry == "poincare":
                x = _disk_from_tangent(u)
            elif geometry == "lorentz":
                x = _disk_from_lorentz(u)
            else:
                x = u
            if not torch.isfinite(x).all() or (geometry == "poincare" and torch.any(torch.linalg.vector_norm(x, dim=1) >= 1)):
                raise FloatingPointError("Nonfinite coordinates or a point outside the open disk.")
            if geometry == "poincare":
                d = _torch_disk_length(x[start:stop], x, self_columns=rows)
            elif geometry == "lorentz":
                d = _lorentz_distance(u[start:stop], u, rows)
            else:
                d = torch.cdist(x[start:stop], x, compute_mode="donot_use_mm_for_euclid_dist")
            t = target[start:stop]
            error = ((d - t) / config.distance_scale).square() / (t / config.distance_scale).clamp_min(config.relative_floor).square()
            global_loss = error.sum() / (n * (n - 1))
            mask = torch.zeros_like(d, dtype=torch.bool)
            mask[torch.arange(stop - start), rows] = True
            logits = (-d / config.gamma).masked_fill(mask, -torch.inf)
            logq = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).masked_fill(mask, 0)
            local_loss = -(teacher[start:stop] * logq).sum() / n
            loss = global_loss + config.local_weight * local_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite graph-stress objective.")
            global_total += float(global_loss.detach())
            local_total += float(local_loss.detach())
            if backward:
                loss.backward()
    if backward and not torch.isfinite(u.grad).all():
        raise FloatingPointError("Nonfinite graph-stress gradient.")
    return global_total + config.local_weight * local_total, global_total, local_total


def _checkpoint_hash(state):
    digest = hashlib.sha256()
    def add(value):
        if hasattr(value, "detach"):
            a = value.detach().cpu().contiguous().numpy()
            add((str(a.dtype), list(a.shape), _digest(a)))
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
                if key != "checkpoint_sha256":
                    add(key); add(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            add(len(value))
            for item in value:
                add(item)
        else:
            raw = json.dumps(value, allow_nan=False, sort_keys=True).encode()
            digest.update(str(len(raw)).encode() + b":" + raw)
    add(state)
    return digest.hexdigest()


def _validate_setup(setup, max_dense_bytes):
    if not isinstance(setup, GraphStressSetup):
        raise TypeError("setup must be returned by prepare_graph_stress.")
    if not isinstance(setup.config, GraphStressConfig) or setup.metadata.get("parameters") != asdict(setup.config):
        raise ValueError("Prepared setup configuration was changed.")
    n = setup.metadata["n_cells"]
    _dense_budget(n, max_dense_bytes)
    for name, key, shape in (("target", "target_sha256", (n, n)),
                             ("teacher", "teacher_sha256", (n, n)),
                             ("base_initial", "base_initial_sha256", (n, 2))):
        a = getattr(setup, name)
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all() or _digest(a) != setup.metadata[key]:
            raise ValueError(f"Prepared {name} was changed or has invalid shape/dtype/values.")
    if setup.metadata["source_hashes"] != _source_hashes():
        raise ValueError("Prepared setup source hashes differ from this implementation.")


def fit_graph_stress(setup, *, epochs=1500, random_state=0, geometry="poincare",
                    chunk_size=256, max_dense_bytes=DEFAULT_MAX_DENSE_BYTES,
                    resume=None, callback: Callable[[GraphStressProgress], None] | None = None,
                    checkpoint_every=100, optimizer_name="adam"):
    """Fit exact relative stress + weighted forward conditional cross-entropy.

    Full nonself denominators are retained in every row chunk. Tangent Adam
    uses the setup configuration. Best selection includes epoch zero and uses
    only this objective. ``epochs`` is the total target epoch on resume.
    Checkpoints contain independent CPU tensors, Adam moments, configuration,
    source/input hashes and an integrity digest. Resume requires these to match
    (including geometry, random seed, chunk size and Torch version). Save only
    these generated state dictionaries, not an untrusted pickled Python object.

    No process-wide thread settings, deadlines, cgroup paths or RSS policy are
    changed. Callers manage execution limits; the dense allocation estimate is
    checked again before fitting: a larger preparation budget must explicitly
    be supplied again here. This is an experimental exact small-N API.
    """
    epochs = _integer("epochs", epochs, 0)
    random_state = _integer("random_state", random_state, 0)
    chunk_size = _integer("chunk_size", chunk_size)
    checkpoint_every = _integer("checkpoint_every", checkpoint_every)
    if geometry not in ("poincare", "euclidean", "lorentz"):
        raise ValueError("geometry must be 'poincare', 'euclidean', or 'lorentz'.")
    if optimizer_name not in ("adam", "sgd"):
        raise ValueError("optimizer_name must be 'adam' or 'sgd'.")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None.")
    _validate_setup(setup, max_dense_bytes)
    torch = _torch()
    config = setup.config
    initial = setup.base_initial + np.random.default_rng(random_state).normal(scale=config.jitter, size=setup.base_initial.shape)
    initial_u = _tangent_from_disk(initial)
    chart = _lorentz_spatial_from_disk(initial) if geometry == "lorentz" else initial_u
    u = torch.tensor(chart, dtype=torch.float64, requires_grad=True)
    def make_optimizer(parameter):
        if optimizer_name == "sgd":
            return torch.optim.SGD([parameter], lr=config.learning_rate)
        return torch.optim.Adam([parameter], lr=config.learning_rate)
    optimizer = make_optimizer(u)
    target = torch.tensor(setup.target, dtype=torch.float64)
    teacher = torch.tensor(setup.teacher, dtype=torch.float64)
    fit_config = dict(geometry=geometry, random_state=random_state, chunk_size=chunk_size)
    if optimizer_name != "adam":
        fit_config["optimizer"] = optimizer_name
    best = initial.copy() if geometry in ("poincare", "lorentz") else initial_u.copy()
    best_loss, best_epoch, trace, start_epoch = np.inf, 0, [], 0
    if resume is not None:
        if not isinstance(resume, dict) or resume.get("schema_version") != 1:
            raise ValueError("Unsupported graph-stress checkpoint schema.")
        if resume.get("checkpoint_sha256") != _checkpoint_hash(resume):
            raise ValueError("Checkpoint integrity hash mismatch.")
        if (resume["metadata"] != setup.metadata or resume["parameters"] != asdict(config)
                or resume["fit_configuration"] != fit_config or resume["torch_version"] != str(torch.__version__)):
            raise ValueError("Checkpoint input, configuration, source or Torch version mismatch.")
        start_epoch = _integer("checkpoint epoch", resume["epoch"], 0)
        if epochs <= start_epoch:
            raise ValueError("epochs must exceed the checkpoint's evaluated epoch.")
        for name in ("u", "best"):
            value = resume[name]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or tuple(value.shape) != chart.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint {name} tensor.")
        saved_trace = np.asarray(resume["trace"], dtype=float)
        if (saved_trace.shape != (start_epoch + 1,) or not np.isfinite(saved_trace).all()
                or float(saved_trace.min()) != resume["best_loss"]
                or int(saved_trace.argmin()) != resume["best_epoch"]):
            raise ValueError("Checkpoint objective history is inconsistent.")
        u = resume["u"].detach().cpu().clone().requires_grad_(True)
        optimizer = make_optimizer(u)
        optimizer.load_state_dict(deepcopy(resume["optimizer"]))
        best = resume["best"].detach().cpu().numpy().copy()
        best_loss, best_epoch = resume["best_loss"], resume["best_epoch"]
        trace = list(resume["trace"][:-1])
    for epoch in range(start_epoch, epochs + 1):
        loss, global_loss, local_loss = _objective(u, target, teacher, config, geometry, chunk_size, backward=epoch < epochs)
        if geometry == "poincare":
            current = _disk_from_tangent(u).detach().numpy().copy()
        elif geometry == "lorentz":
            current = _disk_from_lorentz(u).detach().numpy().copy()
        else:
            current = u.detach().numpy().copy()
        trace.append(loss)
        if loss < best_loss:
            best_loss, best_epoch, best = loss, epoch, current.copy()
        if epoch == start_epoch or epoch % checkpoint_every == 0 or epoch == epochs:
            state = dict(schema_version=1, u=u.detach().cpu().clone(),
                         optimizer=deepcopy(optimizer.state_dict()), epoch=epoch,
                         best=torch.tensor(best.copy(), dtype=torch.float64),
                         best_loss=best_loss, best_epoch=best_epoch, trace=trace.copy(),
                         metadata=deepcopy(setup.metadata), parameters=asdict(config),
                         fit_configuration=fit_config.copy(), torch_version=str(torch.__version__))
            state["checkpoint_sha256"] = _checkpoint_hash(state)
            if callback is not None:
                callback(GraphStressProgress(epoch, loss, global_loss, local_loss,
                                              current.copy(), best.copy(), best_loss,
                                              best_epoch, deepcopy(state)))
        if epoch < epochs:
            optimizer.step()
    return GraphStressResult(best, current, initial, chart, np.asarray(trace),
                             epochs, geometry, best_loss, best_epoch, loss, state)


__all__ = ["GraphStressConfig", "GraphStressSetup", "GraphStressProgress", "GraphStressResult",
           "prepare_graph_stress", "fit_graph_stress"]
