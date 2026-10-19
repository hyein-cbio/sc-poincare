"""Post-fit neighborhood diagnostics, separate from fitting and model selection.

Distances are Euclidean in the explicitly supplied feature representation and
hyperbolic in the raw disk embedding. These are descriptive diagnostics, not
held-out prediction, biological quality scores or the paper's Qlocal/Qglobal.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numbers
import operator

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from sc_poincare._disk import pairwise_disk_length


@dataclass(frozen=True)
class NeighborhoodEvaluation:
    """Summary plus per-cell scores, with copies of the evaluated plot inputs."""

    coordinates: np.ndarray
    labels: np.ndarray
    summary: dict
    purity: dict[int, np.ndarray]
    input_recall: dict[int, np.ndarray]
    cell_ids: np.ndarray | None = None


def _array_hash(values):
    array = np.ascontiguousarray(values)
    header = json.dumps([array.dtype.str, array.shape]).encode()
    return hashlib.sha256(header + b'\0' + array.tobytes()).hexdigest()


def _numeric_label(value):
    """Canonicalize numeric equality without rounding categories to float64."""
    if isinstance(value, (numbers.Integral, np.bool_)):
        return str(int(value))
    try:
        numerator, denominator = value.as_integer_ratio()
    except (AttributeError, OverflowError, ValueError):
        raise ValueError('Numeric labels must be finite real categories.') from None
    if denominator == 1:
        return str(numerator)
    # Keep ordinary labels readable, but do not merge high-precision values
    # or large rationals merely because their float64 conversions coincide.
    try:
        approximate = float(value)
        if np.isfinite(approximate) and approximate.as_integer_ratio() == (numerator, denominator):
            return str(approximate)
    except OverflowError:
        pass
    return f'{numerator}/{denominator}'


def _canonical_labels(values, n):
    # Object dtype preserves missing values rather than converting NaN to 'nan'.
    group = np.asarray(values, dtype=object)
    if group.shape != (n,):
        raise ValueError('labels must have one entry per cell, in the same order.')
    if all(isinstance(value, (str, np.str_)) for value in group):
        group = group.astype(str)
    elif all(isinstance(value, (numbers.Real, np.bool_)) for value in group):
        group = np.asarray([_numeric_label(value) for value in group], dtype=str)
    else:
        raise ValueError('labels must be nonempty strings or finite numeric categories, without missing values.')
    if any(not value.strip() for value in group):
        raise ValueError('labels must not contain empty strings.')
    return group


def _validate_snapshot(result):
    """Reject input-array edits that would detach cached scores from a plot."""
    if not isinstance(result, NeighborhoodEvaluation):
        raise TypeError('result must be returned by evaluate_neighborhoods.')
    names = ('coordinates', 'labels') + (() if result.cell_ids is None else ('cell_ids',))
    for name in names:
        if _array_hash(getattr(result, name)) != result.summary['provenance'][f'{name}_sha256']:
            raise ValueError(f'Evaluated {name} changed after evaluation.')


def _cell_ids(values, n):
    ids = np.asarray(values, dtype=object)
    if ids.shape != (n,) or not all(isinstance(x, (str, np.str_)) and x.strip() for x in ids):
        raise ValueError("cell_ids must contain one nonempty string per cell.")
    ids = ids.astype(str)
    if len(np.unique(ids)) != n:
        raise ValueError("cell_ids must be unique.")
    return ids


def _positive_integer(value, name):
    try:
        integer = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer.") from None
    if isinstance(value, (bool, np.bool_)) or integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _matrix(values, name):
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in 'biuf':
        raise ValueError(f"{name} must be a finite real numeric matrix.")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real numeric matrix.")
    return result


def evaluate_neighborhoods(features, coordinates, labels, *, ks=(15, 30), block_size=256,
                           cell_ids=None, global_n_anchors=128, global_random_state=714,
                           geometry='poincare'):
    """Measure self-excluding kNN purity and preservation without changing inputs.

    All inputs must have the same cell order. ``features`` is the caller's
    fitted representation (e.g. PCA scores), with labels/identifiers excluded.
    ``coordinates`` must be finite N-by-D points with D >= 2. Default ``geometry='poincare'``
    requires points strictly inside the open unit disk; explicit ``'euclidean'``
    evaluates native Euclidean coordinates without disk projection or rescaling.
    Labels may be nonempty strings or finite numeric categories; missing labels
    are rejected, not dropped. Equal numeric values (e.g. 0 and 0.0) denote
    the same category without float64 rounding of distinct values. Labels are
    used only after neighbor retrieval.

    For each requested k (1 <= k < N), overall purity averages the fraction of
    same-label embedding neighbors across cells. Macro purity averages the
    per-population means equally. Input-neighbor recall is |N_input ∩ N_disk|/k.
    Population counts and the attainable purity ceiling min(n_group-1,k)/k
    accompany each mean: small groups cannot reach purity 1 at a large k.
    Singleton groups are included, not silently removed from the macro mean.

    Distances are computed in row blocks against ALL cells. Workspace is
    O(block_size*N), excluding inputs/results and sorting workspace. There is
    no sampling, feature scaling, label-based tuning or display transform.
    Exact distance ties are broken by original row index (stable sorting), so
    duplicates are valid but tie results depend on cell order. These diagnostics
    do not establish convergence or biological correctness.

    A fixed label-independent sample of up to ``global_n_anchors`` rows also
    measures mean row-wise Spearman correlation of feature and disk distances
    to all other cells. Constant-distance rows are undefined, not zero; their
    count is reported. This is a feature-distance diagnostic, not a graph or
    biological-order metric. Optional unique string ``cell_ids`` permit checked
    comparisons; input hashes do not certify upstream cell identity.
    """
    if not isinstance(geometry, str) or geometry not in ('poincare', 'euclidean'):
        raise ValueError("geometry must be 'poincare' or 'euclidean'.")
    features = _matrix(features, 'features')
    points = _matrix(coordinates, 'coordinates')
    n = len(features)
    if n < 2 or points.shape[0] != n or points.shape[1] < 2:
        raise ValueError("coordinates must be N-by-D, D >= 2, with the same N >= 2 as features.")
    if geometry == 'poincare' and np.any(np.sum(points * points, axis=1) >= 1):
        raise ValueError("coordinates must lie strictly inside the unit disk.")
    group = _canonical_labels(labels, n)
    try:
        sizes = sorted({_positive_integer(k, 'k') for k in ks})
    except TypeError:
        raise ValueError("ks must be a nonempty iterable of positive integers.") from None
    if not sizes or sizes[-1] >= n:
        raise ValueError("Each k must satisfy 1 <= k < number of cells.")
    block_size = _positive_integer(block_size, 'block_size')
    n_anchors = min(n, _positive_integer(global_n_anchors, 'global_n_anchors'))
    try:
        anchor_seed = operator.index(global_random_state)
    except TypeError:
        raise ValueError("global_random_state must be a nonnegative integer.") from None
    if isinstance(global_random_state, (bool, np.bool_)) or anchor_seed < 0:
        raise ValueError("global_random_state must be a nonnegative integer.")
    ids = None if cell_ids is None else _cell_ids(cell_ids, n)
    anchors = np.sort(np.random.default_rng(anchor_seed).choice(n, n_anchors, replace=False))
    global_correlations = []
    purity = {k: np.empty(n) for k in sizes}
    recall = {k: np.empty(n) for k in sizes}
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        input_distance = cdist(features[start:stop], features, metric='euclidean')
        disk_distance = (pairwise_disk_length(points[start:stop], points) if geometry == 'poincare'
                         else cdist(points[start:stop], points, metric='euclidean'))
        if not np.isfinite(input_distance).all() or not np.isfinite(disk_distance).all():
            raise FloatingPointError("Nonfinite evaluation distances; review feature/coordinate scale.")
        for anchor in anchors[(anchors >= start) & (anchors < stop)]:
            keep = np.arange(n) != anchor
            left = input_distance[anchor - start, keep]
            right = disk_distance[anchor - start, keep]
            if np.all(left == left[0]) or np.all(right == right[0]):
                continue
            correlation = float(spearmanr(left, right).statistic)
            if np.isfinite(correlation):
                global_correlations.append(correlation)
        rows = np.arange(stop - start)
        columns = np.arange(start, stop)
        input_distance[rows, columns] = np.inf
        disk_distance[rows, columns] = np.inf
        input_neighbors = np.argsort(input_distance, axis=1, kind='stable')[:, :sizes[-1]]
        disk_neighbors = np.argsort(disk_distance, axis=1, kind='stable')[:, :sizes[-1]]
        for k in sizes:
            purity[k][start:stop] = (group[disk_neighbors[:, :k]] == group[start:stop, None]).mean(axis=1)
            recall[k][start:stop] = [
                np.intersect1d(a[:k], b[:k], assume_unique=True).size / k
                for a, b in zip(input_neighbors, disk_neighbors)
            ]
    metrics = {}
    for k in sizes:
        populations = []
        for label in np.unique(group):
            mask = group == label
            count = int(mask.sum())
            populations.append(dict(label=str(label), n_cells=count,
                purity=float(purity[k][mask].mean()), input_neighbor_recall=float(recall[k][mask].mean()),
                purity_ceiling=min(count - 1, k) / k))
        metrics[str(k)] = dict(overall_purity=float(purity[k].mean()),
            macro_purity=float(np.mean([item['purity'] for item in populations])),
            input_neighbor_recall=float(recall[k].mean()), populations=populations)
    summary = dict(n_cells=n, n_features=features.shape[1], ks=sizes,
                   input_metric='euclidean', embedding_metric=geometry,
                   self_excluded=True, tie_policy='distance_then_row_index', metrics=metrics,
                   global_feature_rank=dict(
                       mean_spearman=float(np.mean(global_correlations)) if global_correlations else None,
                       n_anchors=n_anchors, n_valid=len(global_correlations),
                       n_undefined=n_anchors - len(global_correlations), seed=anchor_seed,
                       anchor_indices=anchors.tolist()),
                   provenance=dict(features_sha256=_array_hash(features),
                       coordinates_sha256=_array_hash(points), labels_sha256=_array_hash(group),
                       cell_ids_sha256=None if ids is None else _array_hash(ids)))
    return NeighborhoodEvaluation(points.copy(), group.copy(), summary, purity, recall,
                                  None if ids is None else ids.copy())


def evaluate_expression_neighborhoods(expression, coordinates, *, ks=(15, 30),
                                      geometry='poincare', block_size=64,
                                      per_gene=False, gene_names=None, cell_ids=None):
    """Input-expression preservation, not held-out prediction or biology.

    Predict each gene by the mean of k nonself embedding neighbors; average
    squared error divided by that gene's population variance (ddof=0), then
    average over cells and positive-variance genes. Zero numerical-variance
    genes are omitted and counted. Inputs must use identical cell order; no
    normalization, feature selection or preprocessing is inferred.

    Opt-in per_gene reports retain every original column index, including
    undefined zero-variance genes. Optional names are labels, not identifiers;
    duplicates are retained by index. These are reconstruction diagnostics,
    not marker validation, held-out predictions or gene significance tests.
    """
    if not isinstance(geometry, str) or geometry not in ('poincare', 'euclidean'):
        raise ValueError("geometry must be 'poincare' or 'euclidean'.")
    values, points = _matrix(expression, 'expression'), _matrix(coordinates, 'coordinates')
    n = len(values)
    if n < 2 or points.shape != (n, 2):
        raise ValueError('Expression and coordinates must share N >= 2 cells.')
    if geometry == 'poincare' and np.any(np.sum(points * points, axis=1) >= 1):
        raise ValueError('coordinates must lie strictly inside the unit disk.')
    try:
        sizes = sorted({_positive_integer(k, 'k') for k in ks})
    except TypeError:
        raise ValueError('ks must be an iterable of positive integers.') from None
    if not sizes or sizes[-1] >= n:
        raise ValueError('Each k must be smaller than the cell count.')
    block_size = _positive_integer(block_size, 'block_size')
    if not isinstance(per_gene, (bool, np.bool_)):
        raise ValueError('per_gene must be boolean.')
    if gene_names is not None and not per_gene:
        raise ValueError('gene_names requires per_gene=True.')
    names = None
    if gene_names is not None:
        names = np.asarray(gene_names, dtype=object)
        if names.shape != (values.shape[1],) or not all(isinstance(x, str) and x.strip() for x in names):
            raise ValueError('gene_names must contain one nonempty string per expression column.')
        names = names.astype(str)
    ids = None if cell_ids is None else _cell_ids(cell_ids, n)
    provenance = dict(expression_sha256=_array_hash(values), coordinates_sha256=_array_hash(points),
        gene_names_sha256=None if names is None else _array_hash(names),
        cell_ids_sha256=None if ids is None else _array_hash(ids)) if per_gene or ids is not None else None
    with np.errstate(over='ignore', invalid='ignore'):
        variance = values.var(axis=0)
    if not np.isfinite(variance).all():
        raise FloatingPointError('Nonfinite gene variances; review expression scale.')
    positive = variance > 0
    if not np.any(positive):
        raise ValueError('At least one positive-variance gene is required.')
    original_variance = variance.copy() if per_gene else None
    values, variance = values[:, positive], variance[positive]
    gene_totals = {k: np.zeros(len(variance)) for k in sizes} if per_gene else None
    totals = {k: 0. for k in sizes}
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        distance = (pairwise_disk_length(points[start:stop], points) if geometry == 'poincare'
                    else cdist(points[start:stop], points))
        if not np.isfinite(distance).all():
            raise FloatingPointError('Nonfinite expression-evaluation distances.')
        distance[np.arange(stop - start), np.arange(start, stop)] = np.inf
        neighbors = np.argsort(distance, axis=1, kind='stable')[:, :sizes[-1]]
        for k in sizes:
            prediction = values[neighbors[:, :k]].mean(axis=1)
            gene_error = (prediction - values[start:stop])**2 / variance
            error = gene_error.mean(axis=1)
            if not np.isfinite(error).all():
                raise FloatingPointError('Nonfinite normalized expression error.')
            totals[k] += float(error.sum())
            if per_gene:
                gene_totals[k] += gene_error.sum(axis=0)
    result = dict(geometry=geometry, n_cells=n, n_genes_total=int(len(positive)),
                  n_genes_evaluated=int(positive.sum()), n_zero_variance_genes=int((~positive).sum()),
                  variance_ddof=0, self_excluded=True, tie_policy='distance_then_row_index',
                  normalized_mse={str(k): totals[k] / n for k in sizes})
    if provenance is not None:
        result['provenance'] = provenance
    if per_gene:
        result['per_gene'] = []
        position = 0
        for index, included in enumerate(positive):
            result['per_gene'].append(dict(gene_index=index, gene_name=None if names is None else str(names[index]),
                variance=float(original_variance[index]), evaluated=bool(included),
                normalized_mse={str(k): float(gene_totals[k][position] / n) if included else None for k in sizes}))
            position += int(included)
        result['gene_names_unique'] = None if names is None else len(np.unique(names)) == len(names)
    return result


def evaluate_ordering(pseudotime, stages, *, stage_order, cell_ids=None):
    """Describe agreement with an explicitly predeclared stage order.

    This does not choose a root, infer stage order, refit an embedding or prove
    biological lineage. Unknown/missing stages are rejected, never dropped.
    Constant ranks/pseudotime are explicitly undefined. Declare roots, markers
    and stage order before viewing results to avoid post-hoc selection.
    """
    raw = np.asarray(pseudotime)
    if raw.ndim != 1 or len(raw) < 2 or raw.dtype.kind not in 'biuf':
        raise ValueError('pseudotime must be a real vector with at least two cells.')
    values = raw.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('pseudotime must be finite.')
    order = np.asarray(stage_order, dtype=object)
    if order.ndim != 1 or len(order) < 2:
        raise ValueError('stage_order must be a sequence of at least two categories.')
    order = _canonical_labels(order, len(order))
    if len(np.unique(order)) != len(order):
        raise ValueError('stage_order contains duplicate equivalent categories.')
    group = _canonical_labels(stages, len(values))
    if not set(group).issubset(set(order)):
        raise ValueError('Every observed stage must be in stage_order; no cells are dropped.')
    mapping = {label: i for i, label in enumerate(order)}
    ranks = np.array([mapping[x] for x in group])
    ids = None if cell_ids is None else _cell_ids(cell_ids, len(values))
    defined = len(np.unique(ranks)) > 1 and len(np.unique(values)) > 1
    rho = float(spearmanr(values, ranks).statistic) if defined else None
    if rho is not None and not np.isfinite(rho):
        rho = None
    populations = []
    for label in order:
        subset = values[group == label]
        populations.append(dict(stage=str(label), n_cells=len(subset),
            pseudotime_median=float(np.median(subset)) if len(subset) else None))
    return dict(n_cells=len(values), stage_order=order.tolist(), stages=populations,
        spearman=rho, status='defined' if rho is not None else 'undefined_constant_order_or_pseudotime',
        n_distinct_observed_stages=len(np.unique(ranks)), n_unique_pseudotimes=len(np.unique(values)),
        provenance=dict(pseudotime_sha256=_array_hash(values), stage_labels_sha256=_array_hash(group),
            stage_order_sha256=_array_hash(order), cell_ids_sha256=None if ids is None else _array_hash(ids)))


def compare_neighborhoods(before, after):
    """Return after-minus-before deltas only for matching evaluation inputs.

    Refuse missing IDs, reordering, changed features/labels, k, or anchor rules.
    IDs must honestly describe the row order of each embedding upstream. This
    function does not align, select checkpoints, rank winners or infer biology.
    Undefined global correlations produce a null delta, never a zero delta.
    """
    if before.cell_ids is None or after.cell_ids is None:
        raise ValueError("Both evaluations require cell_ids for a checked comparison.")
    for report in (before, after):
        _validate_snapshot(report)
    if not np.array_equal(before.cell_ids, after.cell_ids):
        raise ValueError("Cell IDs or cell order differ; align inputs explicitly before evaluation.")
    for key in ('features_sha256', 'labels_sha256', 'cell_ids_sha256'):
        if before.summary['provenance'][key] != after.summary['provenance'][key]:
            raise ValueError(f"Evaluation input mismatch: {key}.")
    for key in ('ks', 'input_metric', 'embedding_metric', 'self_excluded', 'tie_policy'):
        if before.summary[key] != after.summary[key]:
            raise ValueError(f"Evaluation policy mismatch: {key}.")
    left_global = before.summary['global_feature_rank']
    right_global = after.summary['global_feature_rank']
    for key in ('n_anchors', 'seed', 'anchor_indices'):
        if left_global[key] != right_global[key]:
            raise ValueError(f"Global evaluation policy mismatch: {key}.")
    metrics = {}
    for k in before.summary['ks']:
        left, right = before.summary['metrics'][str(k)], after.summary['metrics'][str(k)]
        delta = {key: right[key] - left[key]
                 for key in ('overall_purity', 'macro_purity', 'input_neighbor_recall')}
        delta['populations'] = [dict(label=a['label'], n_cells=a['n_cells'],
            purity=b['purity'] - a['purity'],
            input_neighbor_recall=b['input_neighbor_recall'] - a['input_neighbor_recall'])
            for a, b in zip(left['populations'], right['populations'])]
        metrics[str(k)] = delta
    # Different sets of defined anchors are not a like-for-like global mean.
    a, b = left_global['mean_spearman'], right_global['mean_spearman']
    global_delta = b - a if a is not None and b is not None and left_global['n_undefined'] == right_global['n_undefined'] == 0 else None
    return dict(direction='after_minus_before', cell_order_verified=True,
                coordinates_identical=bool(np.array_equal(before.coordinates, after.coordinates)),
                max_absolute_coordinate_change=float(np.max(np.abs(after.coordinates - before.coordinates))),
                metrics=metrics, global_feature_rank_delta=global_delta,
                global_feature_rank_before=left_global, global_feature_rank_after=right_global)


def plot_neighborhood_diagnostics(result, *, k=15, before_coordinates=None, baseline=None, title=None):
    """Plot labels, per-cell input recall and population purity with counts.

    Optionally prepend a before panel with the same labels, palette, axes and
    point sizes. The caller must guarantee matching cell order; no alignment,
    recentering or aesthetic transformation is applied. Scores always describe
    ``result.coordinates``, not the optional before panel. A ``baseline``
    evaluation instead enables ID/input-checked before/after population bars.
    Modified coordinate/label/ID snapshots are rejected before allocating a
    Figure. Returns a Figure; never calls show() or mutates coordinates.
    """
    _validate_snapshot(result)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if result.summary['embedding_metric'] != 'poincare':
        raise ValueError('Use native Euclidean plotting for Euclidean evaluations, not a disk plot.')
    k = _positive_integer(k, 'k')
    if k not in result.purity:
        raise ValueError("k must be one of the evaluated neighborhood sizes.")
    comparison = None
    if baseline is not None:
        if before_coordinates is not None:
            raise ValueError("Pass baseline or before_coordinates, not both.")
        comparison = compare_neighborhoods(baseline, result)
        before_coordinates = baseline.coordinates
    before = None
    if before_coordinates is not None:
        before = _matrix(before_coordinates, 'before_coordinates')
        if before.shape != result.coordinates.shape or np.any(np.sum(before * before, axis=1) >= 1):
            raise ValueError("before_coordinates must have matching shape and lie inside the disk.")
    labels = result.labels
    classes = np.unique(labels)
    palette = plt.get_cmap('tab20')
    colors = {label: palette(i % 20) for i, label in enumerate(classes)}
    panels = 3 if before is None else 4
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, max(5.8, .28 * len(classes) + 1.6)),
                             layout='constrained')
    label_panels = [(result.coordinates, 'Embedding: labels')]
    if before is not None:
        label_panels = [(before, 'Before: labels'), (result.coordinates, 'After: labels')]
    for ax, (xy, caption) in zip(axes, label_panels):
        ax.scatter(xy[:, 0], xy[:, 1], c=[colors[label] for label in labels], s=14, alpha=.8, linewidths=0)
        ax.set_title(caption)
    disk_axes = axes[:len(label_panels) + 1]
    for ax in disk_axes:
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='.6', linewidth=1))
        ax.set(xlim=(-1.06, 1.06), ylim=(-1.06, 1.06), aspect='equal', xlabel='Disk x', ylabel='Disk y')
    ax = disk_axes[-1]
    xy = result.coordinates
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=result.input_recall[k], s=14, linewidths=0,
                         cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'Input-neighbor recall, K={k}')
    fig.colorbar(scatter, ax=ax, fraction=.046, pad=.04, label='Recall (0–1)')
    handles = [Line2D([], [], marker='o', linestyle='', color=colors[label], label=label, markersize=4)
               for label in classes]
    for ax in axes[:len(label_panels)]:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.14),
                  fontsize=7, ncol=4, frameon=False)
    metric = result.summary['metrics'][str(k)]
    populations = metric['populations']
    ax = axes[-1]
    y = np.arange(len(populations))
    bar_colors = [colors[p['label']] for p in populations]
    if baseline is None:
        ax.barh(y, [p['purity'] for p in populations], color=bar_colors)
    else:
        old_populations = baseline.summary['metrics'][str(k)]['populations']
        ax.barh(y - .2, [p['purity'] for p in old_populations], height=.4,
                color=bar_colors, alpha=.35, label='Before')
        ax.barh(y + .2, [p['purity'] for p in populations], height=.4,
                color=bar_colors, label='After')
    ax.scatter([p['purity_ceiling'] for p in populations], y, marker='|', color='black', label='Size-based ceiling')
    ax.axvline(metric['overall_purity'], color='black', linestyle='--', label='Overall')
    ax.axvline(metric['macro_purity'], color='purple', linestyle=':', label='Macro')
    ax.set_yticks(y, [f"{p['label']} (n={p['n_cells']})" for p in populations])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.02), xlabel=f'Neighbor purity, K={k}', title='Population-level evaluation')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.12))
    caption = (f"K={k} | overall purity={metric['overall_purity']:.3f} | "
               f"macro purity={metric['macro_purity']:.3f} | input recall={metric['input_neighbor_recall']:.3f}")
    global_rank = result.summary['global_feature_rank']['mean_spearman']
    caption += f' | feature global rank={global_rank:.3f}' if global_rank is not None else ' | feature global rank=undefined'
    if comparison is not None:
        caption += f" | macro delta={comparison['metrics'][str(k)]['macro_purity']:+.3f}"
    fig.suptitle(f'{title or "Post-fit neighborhood diagnostics"}\n{caption}', fontsize=13)
    return fig
