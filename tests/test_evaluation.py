"""Independent neighborhood references and post-fit non-mutation contracts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from sc_poincare.evaluation import evaluate_neighborhoods, plot_neighborhood_diagnostics


@pytest.fixture
def example():
    return (np.array([[0.], [.1], [1.], [2.], [3.]]),
            np.array([[0., 0.], [.05, 0.], [.15, 0.], [.3, 0.], [.5, 0.]]),
            np.array(['A', 'A', 'A', 'B', 'C']))


def dense_reference(features, coordinates, labels, k):
    n = len(features)
    feature_distance = cdist(features, features)
    separation = ((coordinates[:, None] - coordinates[None, :])**2).sum(axis=-1)
    radius = (coordinates**2).sum(axis=1)
    distance = np.arccosh(1 + 2 * separation / ((1 - radius[:, None]) * (1 - radius[None, :])))
    np.fill_diagonal(feature_distance, np.inf)
    np.fill_diagonal(distance, np.inf)
    input_neighbors = np.argsort(feature_distance, axis=1, kind='stable')[:, :k]
    neighbors = np.argsort(distance, axis=1, kind='stable')[:, :k]
    purity = np.array([sum(labels[j] == labels[i] for j in neighbors[i]) / k for i in range(n)])
    recall = np.array([len(set(a) & set(b)) / k for a, b in zip(input_neighbors, neighbors)])
    return purity, recall


@pytest.mark.parametrize('block_size', [1, 3, 100])
def test_matches_independent_dense_reference(example, block_size):
    features, coordinates, labels = example
    report = evaluate_neighborhoods(features, coordinates, labels, ks=[2, 1, 2], block_size=block_size)
    assert report.summary['ks'] == [1, 2]
    for k in [1, 2]:
        purity, recall = dense_reference(*example, k)
        np.testing.assert_array_equal(report.purity[k], purity)
        np.testing.assert_array_equal(report.input_recall[k], recall)
        metric = report.summary['metrics'][str(k)]
        assert metric['overall_purity'] == pytest.approx(purity.mean())
        assert metric['macro_purity'] == pytest.approx(np.mean([purity[labels == x].mean() for x in set(labels)]))
        assert metric['input_neighbor_recall'] == pytest.approx(recall.mean())
        for population in metric['populations']:
            assert population['n_cells'] == sum(labels == population['label'])
    # A has 3 cells; singleton B and C count equally in the macro mean.
    k1 = report.summary['metrics']['1']
    assert k1['overall_purity'] == pytest.approx(.6)
    assert k1['macro_purity'] == pytest.approx(1 / 3)
    assert [p['purity_ceiling'] for p in k1['populations']] == [1, 0, 0]
    json.dumps(report.summary, allow_nan=False)


@pytest.mark.parametrize('block_size', [1, 4, 20])
def test_duplicates_exclude_self_and_break_ties_by_row_index(block_size):
    features = np.zeros((5, 2))
    coordinates = np.zeros((5, 2))
    labels = np.array(['A', 'B', 'A', 'C', 'C'])
    result = evaluate_neighborhoods(features, coordinates, labels, ks=[1, 4], block_size=block_size)
    np.testing.assert_array_equal(result.purity[1], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(result.purity[4], [.25, 0, .25, .25, .25])
    for k in [1, 4]:
        np.testing.assert_array_equal(result.input_recall[k], np.ones(5))


def test_labels_cannot_change_neighbor_retrieval_and_inputs_are_unchanged(example):
    copies = [a.copy() for a in example]
    for a in example:
        a.setflags(write=False)
    report = evaluate_neighborhoods(*example, ks=[1, 2])
    other = evaluate_neighborhoods(example[0], example[1], np.repeat('all', 5), ks=[1, 2])
    for k in [1, 2]:
        np.testing.assert_array_equal(report.input_recall[k], other.input_recall[k])
        np.testing.assert_array_equal(other.purity[k], np.ones(5))
    report.coordinates[:] = .8
    report.labels[:] = 'Z'
    for original, copy in zip(example, copies):
        np.testing.assert_array_equal(original, copy)


@pytest.mark.parametrize('labels', [[0, 0, 0, 1, 2], ['A', 'A', 'A', 'B', 'C']])
def test_numeric_and_object_string_labels(example, labels):
    dtype = object if isinstance(labels[0], str) else int
    result = evaluate_neighborhoods(example[0], example[1], np.array(labels, dtype=dtype), ks=[1])
    assert result.summary['metrics']['1']['macro_purity'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('labels', [
    [0, 0., 1, 1.],
    [-0., False, True, np.float32(1.)],
    [2**53, float(2**53), 2**53 + 1, 2**53 + 1],
])
def test_equal_numeric_labels_share_categories_without_large_integer_rounding(labels):
    features = np.array([[0.], [.01], [1.], [1.01]])
    points = np.array([[0., 0.], [.01, 0.], [.5, 0.], [.51, 0.]])
    result = evaluate_neighborhoods(features, points, labels, ks=[1])
    metric = result.summary['metrics']['1']
    assert metric['overall_purity'] == metric['macro_purity'] == 1.
    assert [p['n_cells'] for p in metric['populations']] == [2, 2]
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]


def test_distinct_float_precisions_and_rational_labels_are_not_merged():
    from fractions import Fraction
    features = np.array([[0.], [.01], [1.], [1.01]])
    points = np.array([[0., 0.], [.01, 0.], [.5, 0.], [.51, 0.]])
    for a, b in [(np.float32(.1), .1), (Fraction(1, 3), 1 / 3)]:
        result = evaluate_neighborhoods(features, points, [a, a, b, b], ks=[1])
        assert len(np.unique(result.labels)) == 2
        assert result.summary['metrics']['1']['macro_purity'] == 1.


def test_equivalent_numeric_representations_can_be_compared():
    from sc_poincare.evaluation import compare_neighborhoods
    features = np.array([[0.], [.01], [1.], [1.01]])
    points = np.array([[0., 0.], [.01, 0.], [.5, 0.], [.51, 0.]])
    kwargs = dict(ks=[1], cell_ids=['a', 'b', 'c', 'd'])
    before = evaluate_neighborhoods(features, points, [0, 0, 1, 1], **kwargs)
    after = evaluate_neighborhoods(features, points, [0., False, 1., True], **kwargs)
    assert compare_neighborhoods(before, after)['metrics']['1']['macro_purity'] == 0.


@pytest.mark.parametrize('ks', [[], [0], [-1], [True], [1.5], [5], [6], 1, None])
def test_invalid_neighborhood_sizes(example, ks):
    with pytest.raises(ValueError):
        evaluate_neighborhoods(*example, ks=ks)


@pytest.mark.parametrize('size', [0, -1, True, 1.2])
def test_invalid_block_size(example, size):
    with pytest.raises(ValueError):
        evaluate_neighborhoods(*example, ks=[1], block_size=size)


@pytest.mark.parametrize('labels', [[None]*5, [np.nan]*5, ['A', np.nan, 'B', 'C', 'D'],
                                    ['A', None, 'B', 'C', 'D'], [1, 1, np.inf, 2, 3],
                                    ['A', '', 'B', 'C', 'D'],
                                    ['A', ' ', 'B', 'C', 'D'], ['A']*4, [['A']]*5])
def test_invalid_labels(example, labels):
    with pytest.raises(ValueError):
        evaluate_neighborhoods(example[0], example[1], labels, ks=[1])


@pytest.mark.parametrize('features', [np.ones((5, 0)), np.ones(5), np.full((5, 2), np.nan),
                                      np.ones((5, 2), complex), np.full((5, 2), 'label')])
def test_invalid_features(example, features):
    with pytest.raises(ValueError):
        evaluate_neighborhoods(features, example[1], example[2], ks=[1])


@pytest.mark.parametrize('points', [np.ones((5, 2)), np.full((5, 2), np.nan), np.zeros((4, 2)),
                                    np.zeros((5, 3)), np.ones((5, 2), complex)])
def test_invalid_coordinates(example, points):
    with pytest.raises(ValueError):
        evaluate_neighborhoods(example[0], points, example[2], ks=[1])


def test_nonfinite_distances_raise(example):
    features = np.array([[1e308], [-1e308], [0.], [1.], [2.]])
    with pytest.raises(FloatingPointError, match='distances'):
        evaluate_neighborhoods(features, example[1], example[2], ks=[1])


def test_plot_preserves_positions_and_fixes_recall_color_scale(example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = evaluate_neighborhoods(*example, ks=[1])
    before = example[1] * .9
    figure = plot_neighborhood_diagnostics(result, k=1, before_coordinates=before)
    try:
        np.testing.assert_array_equal(figure.axes[0].collections[0].get_offsets(), before)
        np.testing.assert_array_equal(figure.axes[1].collections[0].get_offsets(), example[1])
        np.testing.assert_array_equal(figure.axes[2].collections[0].get_offsets(), example[1])
        assert figure.axes[2].collections[0].get_clim() == (0, 1)
        assert any('n=3' in tick.get_text() for tick in figure.axes[3].get_yticklabels())
        np.testing.assert_array_equal(result.coordinates, example[1])
    finally:
        plt.close(figure)
    with pytest.raises(ValueError):
        plot_neighborhood_diagnostics(result, k=2)
    with pytest.raises(ValueError):
        plot_neighborhood_diagnostics(result, k=1, before_coordinates=np.ones((5, 2)))


@pytest.mark.parametrize('field', ['coordinates', 'labels', 'cell_ids'])
@pytest.mark.parametrize('with_baseline', [False, True])
def test_plot_rejects_modified_snapshot_before_allocating_figures(example, field, with_baseline):
    import matplotlib.pyplot as plt
    kwargs = dict(ks=[1], cell_ids=[f'c{i}' for i in range(len(example[0]))])
    result = evaluate_neighborhoods(*example, **kwargs)
    baseline = evaluate_neighborhoods(*example, **kwargs) if with_baseline else None
    if field == 'coordinates':
        result.coordinates[[1, 3]] = result.coordinates[[3, 1]]
    else:
        getattr(result, field)[0] = 'zz'
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match=f'Evaluated {field} changed'):
        plot_neighborhood_diagnostics(result, k=1, baseline=baseline)
    assert plt.get_fignums() == figures


def load_cli():
    spec = importlib.util.spec_from_file_location('evaluate_cli',
        Path(__file__).resolve().parents[1] / 'examples/evaluate_neighborhoods.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_evaluation_without_mutating_input_or_overwriting(example, tmp_path):
    cli = load_cli()
    path = tmp_path / 'input.npz'
    np.savez(path, features=example[0], coordinates=example[1], labels=example[2], before_coordinates=example[1])
    original = path.read_bytes()
    output = tmp_path / 'evaluation'
    summary = cli.main([str(path), '--output', str(output), '--ks', '1', '2'])
    assert json.loads((output / 'summary.json').read_text()) == summary
    assert path.read_bytes() == original
    assert (output / 'diagnostics_k1.png').stat().st_size > 0
    assert (output / 'diagnostics_k2.png').stat().st_size > 0
    with np.load(output / 'cell_scores.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['purity_k1'], [1, 1, 1, 0, 0])
    saved = (output / 'summary.json').read_bytes()
    with pytest.raises(SystemExit):
        cli.main([str(path), '--output', str(output)])
    assert (output / 'summary.json').read_bytes() == saved


def test_benchmark_records_actual_k_and_matches_standalone_evaluator(tmp_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    features = np.arange(6, dtype=float)[:, None] / 10
    labels = np.array(['A', 'A', 'A', 'B', 'B', 'C'])
    path = tmp_path / 'tiny.csv'
    pd.DataFrame({'feature': features[:, 0], 'label': labels}).to_csv(path, index=False)
    output = tmp_path / 'benchmark.json'
    spec = importlib.util.spec_from_file_location('small_benchmark',
        Path(__file__).resolve().parents[1] / 'examples/benchmark_small.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr('sys.argv', ['benchmark_small.py', str(path), '--output', str(output),
        '--epochs', '1', '--seeds', '0', '--n-neighbors', '2', '--sigma', '1', '--initialization', 'random'])
    module.main()
    summary = json.loads(output.read_text())['records'][0]['neighborhood_evaluation']
    assert summary['ks'] == [5]  # Both requested sizes are capped to N-1 here.
    with np.load(tmp_path / 'benchmark_seed0.npz', allow_pickle=False) as data:
        expected = evaluate_neighborhoods(features, data['raw_coordinates'], labels, ks=[5]).summary
    assert summary == expected


def test_benchmark_rejects_missing_labels_before_fitting(tmp_path, monkeypatch):
    path = tmp_path / 'missing.csv'
    path.write_text('feature,label\n0,A\n1,A\n2,\n3,B\n')
    output = tmp_path / 'benchmark.json'
    spec = importlib.util.spec_from_file_location('missing_label_benchmark',
        Path(__file__).resolve().parents[1] / 'examples/benchmark_small.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'fit_disk_embedding', lambda *a, **kw: pytest.fail('Fitting must not run'))
    monkeypatch.setattr('sys.argv', ['benchmark_small.py', str(path), '--output', str(output)])
    with pytest.raises(ValueError, match='missing'):
        module.main()
    assert not output.exists()


def test_cli_invalid_before_does_not_create_output(example, tmp_path):
    path = tmp_path / 'invalid.npz'
    np.savez(path, features=example[0], coordinates=example[1], labels=example[2], before_coordinates=np.ones((5, 2)))
    output = tmp_path / 'output'
    with pytest.raises(ValueError):
        load_cli().main([str(path), '--output', str(output), '--ks', '1'])
    assert not output.exists()
