"""Identity-checked ablations, global safeguards and compact provenance."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from sc_poincare.evaluation import compare_neighborhoods, evaluate_neighborhoods, plot_neighborhood_diagnostics


@pytest.fixture
def inputs():
    features = np.array([[0.], [.1], [1.], [2.], [3.]])
    coordinates = np.array([[0., 0.], [.05, 0.], [.15, 0.], [.3, 0.], [.5, 0.]])
    labels = np.array(['A', 'A', 'A', 'B', 'C'])
    ids = np.array(['c0', 'c1', 'c2', 'c3', 'c4'])
    return features, coordinates, labels, ids


def evaluate(inputs, **kwargs):
    features, points, labels, ids = inputs
    return evaluate_neighborhoods(features, points, labels, cell_ids=ids, ks=[1, 2], **kwargs)


def test_global_rank_matches_independent_rowwise_reference(inputs):
    features, points, _, _ = inputs
    result = evaluate(inputs, global_n_anchors=3, global_random_state=7, block_size=2)
    info = result.summary['global_feature_rank']
    expected_anchors = np.sort(np.random.default_rng(7).choice(5, 3, replace=False))
    np.testing.assert_array_equal(info['anchor_indices'], expected_anchors)
    correlations = []
    for i in expected_anchors:
        j = np.arange(5) != i
        feature_distances = np.abs(features[i, 0] - features[j, 0])
        # Points lie on a diameter, so intrinsic distance is difference in
        # signed radial coordinates, independently of pairwise_disk_length.
        disk_distances = np.abs(2 * np.arctanh(points[i, 0]) - 2 * np.arctanh(points[j, 0]))
        correlations.append(float(spearmanr(feature_distances, disk_distances).statistic))
    assert info['mean_spearman'] == pytest.approx(np.mean(correlations))
    assert info['n_valid'] == 3
    assert evaluate(inputs, global_n_anchors=3, global_random_state=7, block_size=1).summary == result.summary


def test_undefined_global_rank_is_null_not_zero():
    features, points, labels = np.zeros((4, 2)), np.zeros((4, 2)), ['A'] * 4
    a = evaluate_neighborhoods(features, points, labels, ks=[1], cell_ids=['a', 'b', 'c', 'd'])
    info = a.summary['global_feature_rank']
    assert info['mean_spearman'] is None
    assert info['n_valid'] == 0 and info['n_undefined'] == 4
    assert compare_neighborhoods(a, a)['global_feature_rank_delta'] is None
    json.dumps(a.summary, allow_nan=False)


@pytest.mark.parametrize('kwargs', [dict(global_n_anchors=0), dict(global_n_anchors=True),
                                   dict(global_random_state=-1), dict(global_random_state=True),
                                   dict(global_random_state=1.5)])
def test_invalid_global_policy(inputs, kwargs):
    with pytest.raises(ValueError):
        evaluate(inputs, **kwargs)


@pytest.mark.parametrize('ids', [['a'] * 5, ['a', 'b', 'c', 'd', ''],
                                ['a', 'b', 'c', 'd', None], [0, 1, 2, 3, 4], ['a'] * 4])
def test_invalid_ids(inputs, ids):
    with pytest.raises(ValueError):
        evaluate((*inputs[:3], ids))


def test_comparison_reports_real_after_minus_before_deltas(inputs):
    baseline = evaluate(inputs)
    changed = inputs[1][[4, 3, 0, 1, 2]]
    candidate = evaluate((inputs[0], changed, inputs[2], inputs[3]))
    comparison = compare_neighborhoods(baseline, candidate)
    assert comparison['direction'] == 'after_minus_before'
    assert not comparison['coordinates_identical']
    assert comparison['max_absolute_coordinate_change'] > 0
    for k in ['1', '2']:
        for key in ['overall_purity', 'macro_purity', 'input_neighbor_recall']:
            assert comparison['metrics'][k][key] == pytest.approx(
                candidate.summary['metrics'][k][key] - baseline.summary['metrics'][k][key])
        for row, a, b in zip(comparison['metrics'][k]['populations'],
                             baseline.summary['metrics'][k]['populations'],
                             candidate.summary['metrics'][k]['populations']):
            assert row['purity'] == pytest.approx(b['purity'] - a['purity'])
            assert row['n_cells'] == a['n_cells'] == b['n_cells']
    assert compare_neighborhoods(baseline, baseline)['coordinates_identical']


@pytest.mark.parametrize('change', ['ids', 'order', 'features', 'labels', 'ks', 'anchors', 'seed', 'mutated'])
def test_mismatched_or_modified_evaluations_refused(inputs, change):
    baseline = evaluate(inputs)
    features, points, labels, ids = [a.copy() for a in inputs]
    kwargs = dict(ks=[1, 2], cell_ids=ids)
    if change == 'ids': kwargs['cell_ids'] = None
    elif change == 'order': kwargs['cell_ids'] = ids[::-1]
    elif change == 'features': features *= 2  # Ranks unchanged, but representation is not identical.
    elif change == 'labels': labels[-1] = 'A'
    elif change == 'ks': kwargs['ks'] = [1]
    elif change == 'anchors': kwargs['global_n_anchors'] = 2
    elif change == 'seed': kwargs['global_random_state'] = 0
    candidate = evaluate_neighborhoods(features, points, labels, **kwargs)
    if change == 'mutated': candidate.coordinates[:] = 0
    with pytest.raises(ValueError):
        compare_neighborhoods(baseline, candidate)


def cli():
    spec = importlib.util.spec_from_file_location('evaluation_comparison_cli',
        Path(__file__).resolve().parents[1] / 'examples/evaluate_neighborhoods.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_matched_baseline_raw_coordinate_priority_and_manifest(inputs, tmp_path):
    features, points, labels, ids = inputs
    baseline, candidate = tmp_path / 'baseline.npz', tmp_path / 'candidate.npz'
    np.savez(baseline, features=features, coordinates=points, labels=labels, cell_ids=ids)
    np.savez(candidate, features=features, coordinates=np.ones_like(points),
             raw_coordinates=points, labels=labels, cell_ids=ids)
    out = tmp_path / 'out'
    cli().main([str(candidate), '--baseline', str(baseline), '--output', str(out), '--ks', '1'])
    comparison = json.loads((out / 'comparison.json').read_text())
    assert comparison['coordinates_identical']
    assert comparison['metrics']['1']['macro_purity'] == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['cell_order_verified']
    assert len(manifest['input_sha256']) == len(manifest['baseline_sha256']) == 64
    assert len(manifest['evaluator_sha256']) == 64
    assert 'python' in manifest['versions']
    assert (out / 'diagnostics_k1.png').stat().st_size > 0


@pytest.mark.parametrize('change', ['missing_ids', 'reordered', 'features'])
def test_cli_rejects_unmatched_baseline_before_writing(inputs, tmp_path, change):
    features, points, labels, ids = inputs
    baseline, candidate = tmp_path / 'baseline.npz', tmp_path / 'candidate.npz'
    np.savez(baseline, features=features, coordinates=points, labels=labels, cell_ids=ids)
    arrays = dict(features=features, coordinates=points, labels=labels, cell_ids=ids)
    if change == 'missing_ids': arrays.pop('cell_ids')
    elif change == 'reordered': arrays['cell_ids'] = ids[::-1]
    elif change == 'features': arrays['features'] = features * 2
    np.savez(candidate, **arrays)
    out = tmp_path / 'out'
    with pytest.raises(ValueError):
        cli().main([str(candidate), '--baseline', str(baseline), '--output', str(out), '--ks', '1'])
    assert not out.exists()


def test_checked_plot_shows_both_populations_without_moving_points(inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline = evaluate(inputs)
    figure = plot_neighborhood_diagnostics(baseline, k=1, baseline=baseline)
    try:
        assert len(figure.axes[3].patches) == 6  # 3 populations x before/after
        np.testing.assert_array_equal(figure.axes[0].collections[0].get_offsets(), inputs[1])
        np.testing.assert_array_equal(figure.axes[1].collections[0].get_offsets(), inputs[1])
    finally:
        plt.close(figure)
