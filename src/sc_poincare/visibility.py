"""Fixed-canvas display diagnostics, distinct from intrinsic distance/biology."""
from __future__ import annotations

import numbers

import numpy as np
from scipy.spatial.distance import cdist

from sc_poincare._disk import pairwise_disk_length
from sc_poincare.evaluation import _positive_integer, _validate_snapshot


def evaluate_visibility(result, *, canvas_pixels=600, marker_diameter=4., block_size=256):
    """Describe an evaluated raw embedding on a normalized fixed display.

    The open unit disk has diameter ``canvas_pixels``. Population width is
    twice the 90th percentile Euclidean distance from the population's arithmetic
    display center. Macro width weights populations equally, including singleton
    widths of zero. No point is omitted from plots or changed by this function.

    Marker-center overlap means nearest center distance < marker_diameter,
    not raster occlusion, human readability, or biological mixing. Cross-label
    overlap and nearest-other intrinsic distances are undefined for one-label
    inputs, represented by None. Occupied 10-pixel bins are grid/orientation
    dependent and must not be compared across different inputs or cell counts.

    Intrinsic nearest-other distance uses hyperbolic geometry; it can decrease
    even when displayed populations become wider. Label granularity and expected
    continuous transitions matter for interpretation. These observations never
    tune or select the learned embedding. Inputs are copied by the underlying
    NeighborhoodEvaluation; modified snapshots are rejected here.
    """
    _validate_snapshot(result)
    if result.summary['embedding_metric'] != 'poincare':
        raise ValueError('Fixed-disk visibility requires a Poincare evaluation; do not project Euclidean controls.')
    canvas_pixels = _positive_integer(canvas_pixels, 'canvas_pixels')
    block_size = _positive_integer(block_size, 'block_size')
    if (isinstance(marker_diameter, (bool, np.bool_)) or not isinstance(marker_diameter, numbers.Real)
            or not np.isfinite(marker_diameter) or marker_diameter <= 0):
        raise ValueError('marker_diameter must be finite and positive.')
    points, labels = result.coordinates, result.labels
    n = len(points)
    scale = canvas_pixels / 2.
    nearest = np.empty(n)
    other_screen = np.full(n, np.inf)
    other_intrinsic = np.full(n, np.inf)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        screen = cdist(points[start:stop], points) * scale
        screen[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest[start:stop] = screen.min(axis=1)
        same = labels[start:stop, None] == labels[None, :]
        screen[same] = np.inf
        other_screen[start:stop] = screen.min(axis=1)
        intrinsic = pairwise_disk_length(points[start:stop], points)
        intrinsic[same] = np.inf
        other_intrinsic[start:stop] = intrinsic.min(axis=1)
    populations = []
    multiple_labels = len(np.unique(labels)) > 1
    for label in np.unique(labels):
        mask = labels == label
        cloud = points[mask] * scale
        width = float(2 * np.quantile(np.linalg.norm(cloud - cloud.mean(axis=0), axis=1), .9))
        populations.append(dict(label=str(label), n_cells=int(mask.sum()), width_q90_px=width,
            cross_label_overlap_fraction=float(np.mean(other_screen[mask] < marker_diameter)) if multiple_labels else None,
            nearest_other_center_median_px=float(np.median(other_screen[mask])) if multiple_labels else None,
            intrinsic_nearest_other_median=float(np.median(other_intrinsic[mask])) if multiple_labels else None))
    bins = np.floor(((points + 1) * scale) / 10).astype(np.int64)
    return dict(n_cells=n, policy=dict(canvas_pixels=canvas_pixels,
            marker_diameter=float(marker_diameter), bin_width_px=10, population_width_quantile=.9,
            display='raw_disk_no_recenter_or_alignment', overlap='strict_center_distance_less_than_diameter'),
        provenance=dict(result.summary['provenance']),
        macro_width_q90_px=float(np.mean([p['width_q90_px'] for p in populations])),
        nearest_center_median_px=float(np.median(nearest)),
        marker_overlap_fraction=float(np.mean(nearest < marker_diameter)),
        cross_label_overlap_fraction=float(np.mean(other_screen < marker_diameter)) if multiple_labels else None,
        intrinsic_nearest_other_median=float(np.median(other_intrinsic)) if multiple_labels else None,
        occupied_10px_bins=int(len(np.unique(bins, axis=0))), populations=populations)


def compare_visibility(before, after):
    """Return descriptive union/after minus mutual/before display deltas.

    Requires matching features, labels, unique IDs/order and display policies.
    A positive width delta or negative overlap delta is not an accuracy verdict.
    """
    if before['policy'] != after['policy']:
        raise ValueError('Visibility policies must match.')
    for key in ('features_sha256', 'labels_sha256', 'cell_ids_sha256'):
        if before['provenance'][key] is None or before['provenance'][key] != after['provenance'][key]:
            raise ValueError(f'Visibility input identity mismatch: {key}.')
    def delta(a, b):
        return None if a is None or b is None else b - a
    keys = ('macro_width_q90_px', 'nearest_center_median_px', 'marker_overlap_fraction',
            'cross_label_overlap_fraction', 'intrinsic_nearest_other_median', 'occupied_10px_bins')
    result = {key: delta(before[key], after[key]) for key in keys}
    result['direction'] = 'after_minus_before'
    result['populations'] = [dict(label=a['label'], n_cells=a['n_cells'],
        **{key: delta(a[key], b[key]) for key in ('width_q90_px', 'cross_label_overlap_fraction',
                      'nearest_other_center_median_px', 'intrinsic_nearest_other_median')})
        for a, b in zip(before['populations'], after['populations'])]
    return result


def plot_visibility_pair(before_result, after_result, *, title='Fixed-canvas visibility',
                         canvas_pixels=600, marker_diameter=4.):
    """Draw exact-size disk axes/markers; no alignment, root selection or forces.

    Returns a Figure at 100 dpi. Save it at its native dpi (without tight bbox)
    to retain the canvas/marker dimensions. Population colors match across modes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sc_poincare.evaluation import compare_neighborhoods
    compare_neighborhoods(before_result, after_result)
    reports = [evaluate_visibility(r, canvas_pixels=canvas_pixels, marker_diameter=marker_diameter)
               for r in (before_result, after_result)]
    diameter = reports[0]['policy']['canvas_pixels']
    dpi = 100
    width, height = 2 * diameter + 300, diameter + 220
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    classes = np.unique(before_result.labels)
    palette = plt.get_cmap('tab20')
    colors = {label: palette(i % 20) for i, label in enumerate(classes)}
    fig.suptitle(title, fontsize=14, y=.98)
    for index, (embedding, info, caption) in enumerate(zip((before_result, after_result), reports, ('Mutual', 'Union'))):
        ax = fig.add_axes([(50 + index * (diameter + 150)) / width, 130 / height,
                           diameter / width, diameter / height])
        xy = embedding.coordinates
        ax.scatter(xy[:, 0], xy[:, 1], c=[colors[label] for label in embedding.labels],
                   s=(marker_diameter * 72 / dpi)**2, alpha=.8, linewidths=0, marker='o')
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='.6', linewidth=.7))
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal', xlabel='Disk x', ylabel='Disk y')
        overlap = info['cross_label_overlap_fraction']
        overlap_text = 'undefined' if overlap is None else f'{100 * overlap:.1f}%'
        ax.set_title(f"{caption} | macro width {info['macro_width_q90_px']:.1f}px\n"
                     f"cross-label center overlap {overlap_text}", fontsize=10)
        handles = [Line2D([], [], marker='o', linestyle='', color=colors[label], label=label, markersize=3)
                   for label in classes]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.12), ncol=4, fontsize=7, frameon=False)
    return fig
