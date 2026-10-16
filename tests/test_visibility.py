"""Display spread is not the same as intrinsic separation or accuracy."""
import json

import numpy as np
import pytest

from sc_poincare.evaluation import evaluate_neighborhoods
from sc_poincare.visibility import evaluate_visibility, compare_visibility, plot_visibility_pair


def report(points, labels=('A', 'A', 'B', 'B')):
    n = len(points)
    return evaluate_neighborhoods(np.arange(n, dtype=float)[:, None], np.asarray(points, dtype=float), labels,
                                  ks=[1], cell_ids=[f'cell{i}' for i in range(n)])


@pytest.mark.parametrize('block_size', [1, 3, 256])
def test_known_diameter_geometry_and_width(block_size):
    points = np.array([[0., 0.], [.2, 0.], [.6, 0.], [.8, 0.]])
    evaluation = report(points)
    result = evaluate_visibility(evaluation, block_size=block_size)
    assert result['macro_width_q90_px'] == pytest.approx(60.)
    assert result['nearest_center_median_px'] == pytest.approx(60.)
    assert result['marker_overlap_fraction'] == 0
    assert result['cross_label_overlap_fraction'] == 0
    assert result['occupied_10px_bins'] == 4
    distance = 2 * np.arctanh(points[:, 0])
    nearest_other = [distance[2] - distance[0], distance[2] - distance[1],
                     distance[2] - distance[1], distance[3] - distance[1]]
    assert result['intrinsic_nearest_other_median'] == pytest.approx(np.median(nearest_other))
    for p in result['populations']:
        assert p['n_cells'] == 2 and p['width_q90_px'] == pytest.approx(60.)
    np.testing.assert_array_equal(evaluation.coordinates, points)
    json.dumps(result, allow_nan=False)


def test_wider_display_can_have_lower_intrinsic_separation():
    a = evaluate_visibility(report([[-.9, 0], [-.89, 0], [.89, 0], [.9, 0]]))
    b = evaluate_visibility(report([[-.2, 0], [.2, 0], [.3, 0], [.4, 0]]))
    delta = compare_visibility(a, b)
    assert delta['macro_width_q90_px'] > 0
    assert delta['intrinsic_nearest_other_median'] < 0


def test_marker_center_overlap_and_duplicate_points():
    r = evaluate_visibility(report([[0, 0], [.001, 0], [.2, 0], [.201, 0]], ['A', 'B', 'A', 'B']))
    assert r['marker_overlap_fraction'] == r['cross_label_overlap_fraction'] == 1
    duplicate = evaluate_visibility(report([[0, 0]] * 4))
    assert duplicate['macro_width_q90_px'] == 0
    assert duplicate['cross_label_overlap_fraction'] == 1
    assert duplicate['intrinsic_nearest_other_median'] == 0


def test_one_label_has_undefined_other_population_metrics():
    r = evaluate_visibility(report([[0, 0], [.1, 0], [.2, 0], [.3, 0]], ['A'] * 4))
    assert r['cross_label_overlap_fraction'] is None
    assert r['intrinsic_nearest_other_median'] is None
    assert r['populations'][0]['nearest_other_center_median_px'] is None
    assert compare_visibility(r, r)['intrinsic_nearest_other_median'] is None
    json.dumps(r, allow_nan=False)


def test_canvas_scaling_and_rotation_separate_display_from_intrinsic():
    evaluation = report([[0, 0], [.2, 0], [.6, 0], [.8, 0]])
    a = evaluate_visibility(evaluation)
    b = evaluate_visibility(evaluation, canvas_pixels=1200)
    assert b['macro_width_q90_px'] == pytest.approx(2 * a['macro_width_q90_px'])
    assert b['intrinsic_nearest_other_median'] == a['intrinsic_nearest_other_median']
    rotated = report([[0, 0], [0, .2], [0, .6], [0, .8]])
    assert evaluate_visibility(rotated)['macro_width_q90_px'] == pytest.approx(a['macro_width_q90_px'])
    with pytest.raises(ValueError, match='policies'):
        compare_visibility(a, b)


@pytest.mark.parametrize('kwargs', [dict(canvas_pixels=0), dict(canvas_pixels=True), dict(canvas_pixels=2.5),
                                   dict(marker_diameter=0), dict(marker_diameter=True),
                                   dict(marker_diameter=np.inf), dict(marker_diameter='4'), dict(block_size=0)])
def test_invalid_display_policy(kwargs):
    with pytest.raises(ValueError):
        evaluate_visibility(report([[0, 0], [.1, 0], [.2, 0], [.3, 0]]), **kwargs)


def test_mutated_snapshot_and_unmatched_comparison_rejected():
    r = report([[0, 0], [.1, 0], [.2, 0], [.3, 0]])
    visibility = evaluate_visibility(r)
    r.coordinates[0] = [.5, .5]
    with pytest.raises(ValueError, match='changed'):
        evaluate_visibility(r)
    other = evaluate_visibility(report([[0, 0], [.1, 0], [.2, 0], [.3, 0]], ['A', 'B', 'B', 'B']))
    with pytest.raises(ValueError, match='identity'):
        compare_visibility(visibility, other)
    with pytest.raises(TypeError):
        evaluate_visibility(np.zeros((4, 2)))


def test_plot_has_exact_canvas_and_marker_dimensions():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points = [[0, 0], [.1, 0], [.2, 0], [.3, 0]]
    a = report(points)
    b = report([[0, 0], [.2, 0], [.4, 0], [.6, 0]])
    fig = plot_visibility_pair(a, b)
    try:
        fig.canvas.draw()
        for ax, expected in zip(fig.axes, (a.coordinates, b.coordinates)):
            bounds = ax.get_window_extent()
            assert bounds.width == pytest.approx(600)
            assert bounds.height == pytest.approx(600)
            np.testing.assert_array_equal(ax.collections[0].get_offsets(), expected)
            assert np.sqrt(ax.collections[0].get_sizes()[0]) * fig.dpi / 72 == pytest.approx(4.)
    finally:
        plt.close(fig)
