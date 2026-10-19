"""Landmark reference check against exact mutual-graph stress.

This module does not change ``prepare_graph_stress`` or ``fit_graph_stress``.
It is deliberately outside the hashed fitting sources so existing MU checkpoints
keep their source contract.

The check builds the exact dense target under the shared allocation guard, then
compares an independently computed landmark shortest-path set with that target.
It does not fit, disable the guard, or certify an unseen biological dataset.
A triangle completion still materializes an N-by-N matrix; matching the exact
reference here is not an atlas-scale method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import dijkstra

from sc_poincare import experimental
from sc_poincare._research_policy import ensure_active_paths

OBJECTIVE = "exact_mu_relative_stress_plus_local_ce"
APPROXIMATION = "landmark_triangle"
REFERENCE_ROLE = "reference_check"
UNSEEN_ROLE = "unseen_biological"

NUMPY_METRICS = (
    "landmark_column_max_abs",
    "triangle_excess_max",
    "completion_mean_relative_error",
    "init_procrustes_rmse",
)
GRADIENT_METRICS = (
    "completion_gradient_relative_l2",
    "column_objective_gradient_relative_l2",
)
REFERENCE_METRICS = NUMPY_METRICS + GRADIENT_METRICS

# Normalized names named by the validation-readiness audit and the roadmap.
# An identifier outside this list is not a certified blind holdout.
_DEVELOPMENT_INPUTS = frozenset({
    "paul",
    "moignard",
    "moignard2015",
    "pancreas",
    "nestorowa",
})


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _normalize_input_id(value):
    if not isinstance(value, str) or not value or len(value) > 80:
        raise ValueError("input_id must be a string of 1 to 80 characters.")
    if any(character.isspace() for character in value):
        raise ValueError("input_id must not contain whitespace.")
    return "".join(character for character in value.lower() if character.isalnum())


def _canonical_features(features, n_neighbors):
    raw = np.asarray(features)
    if raw.ndim != 2 or raw.shape[1] == 0 or raw.shape[0] <= n_neighbors:
        raise ValueError("features must be N×D with D > 0 and N > n_neighbors.")
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError("features must contain real numeric values, with labels excluded.")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("features must be finite.")
    return array


def _parameters_json(config):
    if not isinstance(config, experimental.GraphStressConfig):
        raise TypeError("config must be a GraphStressConfig.")
    return json.dumps(asdict(config), allow_nan=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FrozenComparison:
    """Input, metric, objective and resource limits fixed before the check runs.

    ``max_epochs`` is zero because this check evaluates the exact initializer.
    It does not spend a fitting budget. ``role='unseen_biological'`` may be
    declared only for an identifier that is not a known development collection;
    the reference check still refuses to execute that role.
    """

    input_id: str
    role: str
    n_cells: int
    feature_sha256: str
    parameters_json: str
    n_landmarks: int
    max_dense_bytes: int
    max_epochs: int
    seeds: tuple
    primary_metric: str
    sensitivity_metrics: tuple
    labels_excluded: bool = True
    approximation: str = APPROXIMATION
    objective: str = OBJECTIVE
    schema_version: int = 1

    def to_dict(self):
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["sensitivity_metrics"] = list(self.sensitivity_metrics)
        return payload


@dataclass(frozen=True)
class LandmarkReference:
    """JSON report plus the initializer arrays the manifest stores.

    ``report['fits_executed']`` is zero. ``report['biological_validation']``
    is false. The arrays are copies.
    """

    report: dict
    arrays: dict


def freeze_comparison(features, config, *, input_id, role, n_landmarks, max_dense_bytes,
                      max_epochs, seeds, primary_metric, sensitivity_metrics,
                      labels_excluded=True):
    """Bind the feature hash and exact MU configuration before comparison.

    The returned object does not prepare targets or fit. Call
    ``compare_landmark_reference`` with the same features and configuration.
    """
    if role not in (REFERENCE_ROLE, UNSEEN_ROLE):
        raise ValueError("role must be 'reference_check' or 'unseen_biological'.")
    if labels_excluded is not True:
        raise ValueError("labels_excluded must be True. Labels cannot enter the reference check.")
    normalized = _normalize_input_id(input_id)
    if role == UNSEEN_ROLE and normalized in _DEVELOPMENT_INPUTS:
        raise ValueError(
            f"{input_id!r} matches a known development collection ({normalized}). "
            "It cannot be declared as unseen biological data."
        )
    if not isinstance(config, experimental.GraphStressConfig):
        raise TypeError("config must be a GraphStressConfig.")
    array = _canonical_features(features, config.n_neighbors)
    n_landmarks = _integer("n_landmarks", n_landmarks, 2)
    if n_landmarks > len(array):
        raise ValueError("n_landmarks cannot exceed the number of cells.")
    if isinstance(max_dense_bytes, bool) or max_dense_bytes is None:
        raise ValueError("max_dense_bytes must be a positive integer. The dense guard stays enabled.")
    max_dense_bytes = _integer("max_dense_bytes", max_dense_bytes, 1)
    max_epochs = _integer("max_epochs", max_epochs, 0)
    if max_epochs != 0:
        raise ValueError(
            "max_epochs must be 0. This reference check does not fit; a positive epoch "
            "budget is a separate experiment and is not authorized here."
        )
    if isinstance(seeds, (str, bytes)) or not seeds:
        raise ValueError("seeds must be a non-empty sequence of integers.")
    seeds = tuple(_integer("seed", seed, 0) for seed in seeds)
    if primary_metric not in REFERENCE_METRICS:
        raise ValueError(f"primary_metric must be one of {REFERENCE_METRICS}.")
    if isinstance(sensitivity_metrics, str) or not sensitivity_metrics:
        raise ValueError("sensitivity_metrics must be a non-empty sequence of metric names.")
    sensitivity = tuple(sensitivity_metrics)
    if len(set(sensitivity)) != len(sensitivity):
        raise ValueError("sensitivity_metrics must not contain duplicates.")
    if any(metric not in REFERENCE_METRICS for metric in sensitivity):
        raise ValueError(f"sensitivity_metrics must be chosen from {REFERENCE_METRICS}.")
    if primary_metric in sensitivity:
        raise ValueError("primary_metric is declared separately from sensitivity_metrics.")
    return FrozenComparison(
        input_id=input_id,
        role=role,
        n_cells=len(array),
        feature_sha256=experimental._digest(array),
        parameters_json=_parameters_json(config),
        n_landmarks=n_landmarks,
        max_dense_bytes=max_dense_bytes,
        max_epochs=max_epochs,
        seeds=seeds,
        primary_metric=primary_metric,
        sensitivity_metrics=sensitivity,
        labels_excluded=True,
    )


def _transformed_mutual(features, n_neighbors):
    lengths = experimental._graph_lengths(features, n_neighbors, "mutual")
    positive = lengths.data[lengths.data > 0]
    if not len(positive) or not np.isfinite(positive).all():
        raise ValueError("A nondegenerate graph with finite positive edge lengths is required.")
    transformed = lengths.copy()
    transformed.data = np.log1p(transformed.data / float(np.median(positive)))
    return transformed


def _farthest_landmarks(graph, n_landmarks, seed):
    """Farthest-point order on the transformed mutual graph.

    The first landmark is ``default_rng(seed).integers(n)``. Later landmarks
    maximize graph distance to the selected set. Ties keep the smallest index.
    """
    n = graph.shape[0]
    first = int(np.random.default_rng(seed).integers(n))
    selected = [first]
    nearest = np.asarray(dijkstra(graph, directed=False, indices=first), dtype=np.float64)
    if nearest.shape != (n,) or not np.isfinite(nearest).all():
        raise ValueError("Finite distances from the first landmark are required.")
    while len(selected) < n_landmarks:
        masked = nearest.copy()
        masked[selected] = -np.inf
        selected.append(int(np.argmax(masked)))
        nearest = np.minimum(nearest, dijkstra(graph, directed=False, indices=selected[-1]))
        if not np.isfinite(nearest).all():
            raise ValueError("Finite distances to every selected landmark are required.")
    indices = np.asarray(selected, dtype=np.int64)
    if len(np.unique(indices)) != n_landmarks:
        raise ValueError("Landmark selection produced a duplicate index.")
    distances = np.asarray(dijkstra(graph, directed=False, indices=indices), dtype=np.float64)
    if distances.shape != (n_landmarks, n) or not np.isfinite(distances).all():
        raise ValueError("Landmark shortest paths must be finite and shaped (landmarks, cells).")
    return indices, distances


def _triangle_completion(distances):
    """Shortest paths through one landmark. This is an upper bound on the graph metric."""
    n = distances.shape[1]
    completion = np.full((n, n), np.inf, dtype=np.float64)
    for row in np.asarray(distances, dtype=np.float64):
        completion = np.minimum(completion, row[:, None] + row[None, :])
    if not np.isfinite(completion).all():
        raise ValueError("Landmark triangle completion left a nonfinite pair distance.")
    np.fill_diagonal(completion, 0.)
    return completion


def _landmark_initialization(distances, indices, initial_radius):
    """Landmark MDS, scaled to the same maximum radius as exact classical scaling."""
    d_nl = np.asarray(distances.T, dtype=np.float64)
    d_ll = np.asarray(distances[:, indices], dtype=np.float64)
    d_ll = 0.5 * (d_ll + d_ll.T)
    np.fill_diagonal(d_ll, 0.)
    squared = d_ll * d_ll
    gram = -0.5 * (squared - squared.mean(0) - squared.mean(1, keepdims=True) + squared.mean())
    width = len(d_ll)
    values, vectors = eigh(gram, subset_by_index=[width - 2, width - 1])
    lam = np.maximum(values[::-1], 0.)
    basis = vectors[:, ::-1]
    if not np.any(lam > 0):
        raise ValueError("A nondegenerate landmark classical-scaling initialization is required.")
    d2 = d_nl * d_nl
    cross = -0.5 * (d2 - d2.mean(1, keepdims=True) - squared.mean(0) + squared.mean())
    scale = np.zeros_like(lam)
    scale[lam > 0] = 1. / np.sqrt(lam[lam > 0])
    initial = cross @ basis * scale
    radius = float(np.linalg.norm(initial, axis=1).max())
    if not np.isfinite(initial).all() or radius <= 0:
        raise ValueError("A nondegenerate landmark classical-scaling initialization is required.")
    return np.ascontiguousarray(initial * (initial_radius / radius), dtype=np.float64)


def _procrustes_rmse(reference, candidate):
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError("Procrustes inputs must share shape (N, 2).")
    left = left - left.mean(0)
    right = right - right.mean(0)
    rotation_source, _, rotation_target = np.linalg.svd(left.T @ right, full_matrices=False)
    rotation = rotation_source @ rotation_target
    residual = left @ rotation - right
    return float(np.sqrt(np.mean(residual * residual)))


def _relative_l2(candidate, reference):
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(np.asarray(candidate) - np.asarray(reference)))
    if denominator == 0.:
        return numerator
    return numerator / denominator


def _column_objective(u, distances, landmark_index, mean, teacher, config, geometry, chunk_size, *, backward):
    """Exact local CE plus global stress on pairs that touch a landmark.

    Both directed endpoints are included when either cell is a landmark.
    The denominator stays ``N*(N-1)``, so omitted pairs contribute nothing
    rather than a rescaled mean. When every cell is a landmark this is the
    exact graph-stress objective.
    """
    torch = experimental._torch()
    from sc_poincare._fitting import _torch_disk_length
    n = int(u.shape[0])
    index = torch.as_tensor(np.asarray(landmark_index, dtype=np.int64))
    position = torch.full((n,), -1, dtype=torch.long)
    position[index] = torch.arange(len(landmark_index))
    is_landmark = position >= 0
    target_ln = torch.as_tensor(np.asarray(distances, dtype=np.float64) * (config.distance_scale / mean))
    teacher_t = teacher if hasattr(teacher, "detach") else torch.as_tensor(teacher)
    u.grad = None
    global_total, local_total = 0., 0.
    with torch.set_grad_enabled(backward):
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            rows = torch.arange(start, stop)
            x = experimental._disk_from_tangent(u) if geometry == "poincare" else u
            if not torch.isfinite(x).all() or (geometry == "poincare" and torch.any(torch.linalg.vector_norm(x, dim=1) >= 1)):
                raise FloatingPointError("Nonfinite coordinates or a point outside the open disk.")
            d = (_torch_disk_length(x[start:stop], x, self_columns=rows) if geometry == "poincare"
                 else torch.cdist(x[start:stop], x, compute_mode="donot_use_mm_for_euclid_dist"))
            row_pos = position[start:stop]
            # Landmark columns for every row, then the full landmark row so both
            # directed endpoints are present for every pair that touches a landmark.
            t = torch.zeros_like(d)
            t[:, index] = target_ln[:, rows].T
            landmark_rows = row_pos >= 0
            if bool(landmark_rows.any()):
                t[landmark_rows] = target_ln[row_pos[landmark_rows]]
            included = is_landmark[start:stop, None] | is_landmark[None, :]
            included = included.clone()
            included[torch.arange(stop - start), rows] = False
            scale = config.distance_scale
            error = ((d - t) / scale).square() / (t / scale).clamp_min(config.relative_floor).square()
            error = torch.where(included, error, torch.zeros_like(error))
            global_loss = error.sum() / (n * (n - 1))
            self_mask = torch.zeros_like(d, dtype=torch.bool)
            self_mask[torch.arange(stop - start), rows] = True
            logits = (-d / config.gamma).masked_fill(self_mask, -torch.inf)
            logq = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).masked_fill(self_mask, 0)
            local_loss = -(teacher_t[start:stop] * logq).sum() / n
            loss = global_loss + config.local_weight * local_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite landmark-column objective.")
            global_total += float(global_loss.detach())
            local_total += float(local_loss.detach())
            if backward:
                loss.backward()
    if backward and not torch.isfinite(u.grad).all():
        raise FloatingPointError("Nonfinite landmark-column gradient.")
    return global_total + config.local_weight * local_total, global_total, local_total


def _gradient_metrics(setup, completion, distances, landmark_index, geometry="poincare", chunk_size=7):
    torch = experimental._torch()
    config = setup.config
    mean = setup.metadata["global_mean"]
    tangent = experimental._tangent_from_disk(setup.base_initial)
    teacher = torch.tensor(np.array(setup.teacher, dtype=np.float64, copy=True), dtype=torch.float64)
    exact_target = torch.tensor(np.array(setup.target, dtype=np.float64, copy=True), dtype=torch.float64)
    completed = completion / mean * config.distance_scale
    np.fill_diagonal(completed, 0.)
    completed_target = torch.tensor(completed, dtype=torch.float64)

    def grad_of(objective, argument):
        u = torch.tensor(tangent, dtype=torch.float64, requires_grad=True)
        objective(u, argument)
        return u.grad.detach().numpy().copy()

    exact = grad_of(
        lambda u, target: experimental._objective(u, target, teacher, config, geometry, chunk_size, backward=True),
        exact_target,
    )
    completed_grad = grad_of(
        lambda u, target: experimental._objective(u, target, teacher, config, geometry, chunk_size, backward=True),
        completed_target,
    )
    column = grad_of(
        lambda u, _ignored: _column_objective(
            u, distances, landmark_index, mean, teacher, config, geometry, chunk_size, backward=True,
        ),
        None,
    )
    column_full = grad_of(
        lambda u, _ignored: _column_objective(
            u, distances, landmark_index, mean, teacher, config, geometry, len(tangent), backward=True,
        ),
        None,
    )
    if not np.allclose(column, column_full, rtol=1e-12, atol=1e-12):
        raise FloatingPointError("Landmark-column gradient changed with chunk size.")
    return {
        "completion_gradient_relative_l2": _relative_l2(completed_grad, exact),
        "column_objective_gradient_relative_l2": _relative_l2(column, exact),
    }


def compare_landmark_reference(features, protocol, config, *, geometry="poincare"):
    """Compare landmark paths, initialization and endpoint gradients with exact MU.

    The exact preparation applies ``protocol.max_dense_bytes``. ``None`` cannot
    be smuggled in through the protocol. No optimizer step is taken. The
    evaluated point is the exact classical-scaling initializer, where every
    included pair contributes both directed endpoints.
    """
    if not isinstance(protocol, FrozenComparison):
        raise TypeError("protocol must be a FrozenComparison from freeze_comparison.")
    if protocol.role != REFERENCE_ROLE:
        raise ValueError(
            "compare_landmark_reference executes only role='reference_check'. "
            "An unseen-biological declaration is not a fit and is not certified here."
        )
    if protocol.approximation != APPROXIMATION or protocol.objective != OBJECTIVE:
        raise ValueError("Frozen objective or approximation does not match this reference check.")
    if protocol.labels_excluded is not True or protocol.max_epochs != 0:
        raise ValueError("The frozen protocol does not match the no-label, no-fit reference check.")
    if geometry != "poincare":
        raise ValueError("The reference gradient check uses the Poincaré initializer.")
    if _parameters_json(config) != protocol.parameters_json:
        raise ValueError("config does not match the frozen GraphStressConfig.")
    array = _canonical_features(features, config.n_neighbors)
    if len(array) != protocol.n_cells or experimental._digest(array) != protocol.feature_sha256:
        raise ValueError("features do not match the frozen input hash or cell count.")
    declared = {protocol.primary_metric, *protocol.sensitivity_metrics}
    if any(metric in GRADIENT_METRICS for metric in declared):
        experimental._torch()
    setup = experimental.prepare_graph_stress(
        array, config=config, max_dense_bytes=protocol.max_dense_bytes,
    )
    indices, distances = _farthest_landmarks(
        _transformed_mutual(array, config.n_neighbors), protocol.n_landmarks, protocol.seeds[0],
    )
    exact = np.array(setup.target, dtype=np.float64, copy=True)
    exact *= setup.metadata["global_mean"] / config.distance_scale
    column_error = float(np.max(np.abs(distances.T - exact[:, indices])))
    completion = _triangle_completion(distances)
    excess = float(np.max(completion - exact))
    completion_mean = float(completion.sum() / (len(array) * (len(array) - 1)))
    mean_error = abs(completion_mean - setup.metadata["global_mean"]) / setup.metadata["global_mean"]
    initial = _landmark_initialization(distances, indices, config.initial_radius)
    metrics = {
        "landmark_column_max_abs": column_error,
        "triangle_excess_max": excess,
        "completion_mean_relative_error": float(mean_error),
        "init_procrustes_rmse": _procrustes_rmse(setup.base_initial, initial),
    }
    if any(metric in GRADIENT_METRICS for metric in declared):
        metrics.update(_gradient_metrics(setup, completion, distances, indices))
    missing = declared.difference(metrics)
    if missing:
        raise RuntimeError(f"Declared metrics were not computed: {sorted(missing)}.")
    report = {
        "schema_version": 1,
        "fits_executed": 0,
        "biological_validation": False,
        "normalization": "exact_nonself_mean",
        "landmark_seed": protocol.seeds[0],
        "executed_landmark_seeds": [protocol.seeds[0]],
        "declared_seeds": list(protocol.seeds),
        "primary_metric": protocol.primary_metric,
        "primary_value": metrics[protocol.primary_metric],
        "sensitivity": {name: metrics[name] for name in protocol.sensitivity_metrics},
        "metrics": metrics,
        "protocol": protocol.to_dict(),
        "n_cells": protocol.n_cells,
        "n_landmarks": protocol.n_landmarks,
        "global_mean": setup.metadata["global_mean"],
        "completion_nonself_mean": completion_mean,
    }
    arrays = {
        "landmark_index": indices.copy(),
        "exact_initial": np.array(setup.base_initial, dtype=np.float64, copy=True),
        "landmark_initial": initial,
    }
    return LandmarkReference(report, arrays)


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_reference_tree(directory, reference):
    """Write the report, initializer arrays and a checksum manifest.

    The directory must not already exist. Closed research trees are rejected.
    The manifest covers the payload files, and a reread must reproduce it.
    """
    if not isinstance(reference, LandmarkReference):
        raise TypeError("reference must be a LandmarkReference.")
    if reference.report.get("fits_executed") != 0 or reference.report.get("biological_validation") is not False:
        raise ValueError("Only an unfitted, non-biological reference report can be written.")
    path = Path(directory)
    ensure_active_paths(path)
    if path.exists():
        raise FileExistsError(f"Reference tree already exists: {path}.")
    path.mkdir()
    try:
        report_bytes = json.dumps(reference.report, allow_nan=False, sort_keys=True, indent=2).encode()
        _atomic_bytes(path / "report.json", report_bytes)
        buffer = path / "arrays.npz"
        temporary = path / "arrays.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, **reference.arrays)
        os.replace(temporary, buffer)
        files = {}
        for name in ("report.json", "arrays.npz"):
            payload = path / name
            files[name] = {"sha256": _sha256_file(payload), "bytes": payload.stat().st_size}
        manifest = {"schema_version": 1, "files": files}
        _atomic_bytes(path / "MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2).encode())
        verify_reference_tree(path)
    except Exception:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
        raise
    return path


def verify_reference_tree(directory):
    """Recompute payload checksums and reload the initializer arrays.

    A mismatch raises ``ValueError``. This is a local reread, not an external
    backup or a new evaluation of a fitted model.
    """
    path = Path(directory)
    ensure_active_paths(path)
    manifest = json.loads((path / "MANIFEST.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != {"report.json", "arrays.npz"}:
        raise ValueError("Reference manifest schema or file list is invalid.")
    for name, record in manifest["files"].items():
        payload = path / name
        if _sha256_file(payload) != record["sha256"] or payload.stat().st_size != record["bytes"]:
            raise ValueError(f"Reference payload checksum mismatch: {name}.")
    report = json.loads((path / "report.json").read_text())
    if report.get("fits_executed") != 0 or report.get("biological_validation") is not False:
        raise ValueError("Reference report claims fitting or biological validation.")
    n_cells = report["n_cells"]
    n_landmarks = report["n_landmarks"]
    with np.load(path / "arrays.npz") as arrays:
        if arrays["landmark_index"].shape != (n_landmarks,):
            raise ValueError("Stored landmark index does not match the report.")
        for name in ("exact_initial", "landmark_initial"):
            values = arrays[name]
            if values.shape != (n_cells, 2) or values.dtype != np.float64 or not np.isfinite(values).all():
                raise ValueError(f"Stored {name} is missing, nonfinite or the wrong shape.")
    if report["primary_metric"] not in report["metrics"]:
        raise ValueError("Reference report primary metric is absent.")
    if report.get("executed_landmark_seeds") != [report["landmark_seed"]]:
        raise ValueError("Reference report executed seeds do not match its landmark seed.")
    return report


__all__ = [
    "APPROXIMATION",
    "GRADIENT_METRICS",
    "NUMPY_METRICS",
    "OBJECTIVE",
    "REFERENCE_METRICS",
    "FrozenComparison",
    "LandmarkReference",
    "compare_landmark_reference",
    "freeze_comparison",
    "verify_reference_tree",
    "write_reference_tree",
]
