"""The research extension must preserve its declared objective and chunk semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

pytest.importorskip('torch')

from sc_poincare._disk import pairwise_disk_length
from sc_poincare._fitting import evaluate_alignment
from sc_poincare._proximity import estimate_proximity, local_affinity

spec = importlib.util.spec_from_file_location('experimental_geodesic_example',
    Path(__file__).resolve().parents[1] / 'examples' / 'experimental_geodesic_fit.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
fit = module.fit_geodesic_regularized


def test_returned_checkpoint_has_the_reported_regularized_objective():
    features = np.column_stack([np.linspace(0., 1., 8), np.linspace(0., 1., 8)**2])
    result = fit(features, n_neighbors=3, sigma=1., stages=((4, 2.), (3, 5.)), random_state=2)
    target = estimate_proximity(features, 3, 1.).accessibility
    _, lengths = local_affinity(features, 3, 10.)
    geo = dijkstra(lengths, directed=False)
    n = len(features)
    scaled = geo / (geo.sum() / (n * (n - 1))) * 3.
    distances = pairwise_disk_length(result.coordinates)
    expected = evaluate_alignment(result.coordinates, target, .5) / n + 5 * np.sum(((distances - scaled) / 3.)**2) / (n * (n - 1))
    assert result.stages[-1]['best_loss'] == pytest.approx(expected, rel=1e-10)
    assert [len(stage['loss_trace']) for stage in result.stages] == [4, 3]
    assert result.parameters['optimizer'] == 'spatial_coordinate_adam'
    assert np.all(np.linalg.norm(result.coordinates, axis=1) < 1.)


def test_research_fit_reproducibility_and_row_chunk_invariance():
    features = np.random.default_rng(7).normal(size=(9, 3))
    original = features.copy()
    a = fit(features, n_neighbors=4, sigma=2., stages=((5, 3.),), chunk_size=1, random_state=4)
    b = fit(features, n_neighbors=4, sigma=2., stages=((5, 3.),), chunk_size=9, random_state=4)
    np.testing.assert_array_equal(features, original)
    np.testing.assert_allclose(a.coordinates, b.coordinates, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(a.stages[0]['loss_trace'], b.stages[0]['loss_trace'], atol=1e-10)


@pytest.mark.parametrize('kwargs', [
    {'gamma': 0.}, {'learning_rate': float('nan')}, {'distance_scale': -1.},
    {'chunk_size': 0}, {'stages': ()}, {'stages': ((2, 0.),)}, {'stages': ((1.5, 2.),)},
])
def test_research_fit_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        fit(np.arange(8.)[:, None], **kwargs)


def test_research_fit_rejects_a_zero_distance_manifold():
    with pytest.raises(ValueError, match='positive off-diagonal mean'):
        fit(np.zeros((8, 2)), n_neighbors=7, sigma=1., stages=((1, 1.),))
