"""Native controls are evaluated in their own metric, never projected to disk."""
import numpy as np
import pytest

from sc_poincare.evaluation import evaluate_neighborhoods, compare_neighborhoods, plot_neighborhood_diagnostics
from sc_poincare.visibility import evaluate_visibility


def test_native_euclidean_neighbors_and_global_rank():
    points = np.array([[0., 0.], [2., 0.], [7., 0.], [11., 0.]])
    original = points.copy()
    report = evaluate_neighborhoods(points, points, ['A','A','B','B'], ks=[1,2],
                                   geometry='euclidean', cell_ids=['a','b','c','d'])
    assert report.summary['embedding_metric'] == 'euclidean'
    assert report.summary['global_feature_rank']['mean_spearman'] == pytest.approx(1.)
    np.testing.assert_array_equal(report.input_recall[1], np.ones(4))
    np.testing.assert_array_equal(report.purity[1], np.ones(4))
    np.testing.assert_array_equal(report.coordinates, original)
    np.testing.assert_array_equal(points, original)
    with pytest.raises(ValueError, match='inside'):
        evaluate_neighborhoods(points, points, ['A','A','B','B'], ks=[1])
    with pytest.raises(ValueError, match='Euclidean'):
        plot_neighborhood_diagnostics(report, k=1)
    with pytest.raises(ValueError, match='Poincare'):
        evaluate_visibility(report)


@pytest.mark.parametrize('geometry', ['lorentz', None, True, ['euclidean']])
def test_invalid_geometry(geometry):
    with pytest.raises(ValueError, match='geometry'):
        evaluate_neighborhoods([[0],[1]], [[0,0],[.1,0]], ['A','B'],ks=[1],geometry=geometry)


def test_mixed_metric_comparison_is_not_presented_as_checked_delta():
    features = np.arange(4.)[:,None]
    points = np.array([[0,0],[.1,0],[.2,0],[.3,0]])
    kwargs = dict(ks=[1],cell_ids=['a','b','c','d'])
    a = evaluate_neighborhoods(features,points,['A','A','B','B'],**kwargs)
    b = evaluate_neighborhoods(features,points,['A','A','B','B'],geometry='euclidean',**kwargs)
    with pytest.raises(ValueError,match='policy'):
        compare_neighborhoods(a,b)
