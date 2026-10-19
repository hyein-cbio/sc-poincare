"""Observational progress snapshots and explicit warm starts for long runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sc_poincare import _fitting


def _target():
    t = np.random.default_rng(8).uniform(.1, 1., (6, 6))
    return t / t.sum(axis=1, keepdims=True)


@pytest.mark.parametrize('backend', ['torch', 'numpy'])
def test_callbacks_are_observational_and_report_evaluated_checkpoints(monkeypatch, backend):
    if backend == 'torch':
        pytest.importorskip('torch')
    else:
        monkeypatch.setattr(_fitting, 'torch', None)
    target = _target()
    expected = _fitting.fit_disk_embedding(target, n_epochs=4, random_state=3)
    seen = []
    def callback(progress):
        assert progress.loss == pytest.approx(_fitting.evaluate_alignment(progress.coordinates, target, 2.), rel=1e-9)
        assert progress.best_loss == pytest.approx(_fitting.evaluate_alignment(progress.best_coordinates, target, 2.), rel=1e-9)
        seen.append(progress.epoch)
        # A callback may reuse or edit its snapshots without changing the fit.
        progress.coordinates[:] = 0.
        progress.best_coordinates[:] = 0.
        return 'ignored'
    actual = _fitting.fit_disk_embedding(target, n_epochs=4, random_state=3, progress_callback=callback)
    assert seen == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(actual.coordinates, expected.coordinates)
    np.testing.assert_array_equal(actual.loss_trace, expected.loss_trace)
    assert actual.best_loss == expected.best_loss
    assert len(actual.gradient_norm_trace) == len(actual.loss_trace) == 4
    assert all(np.isfinite(actual.gradient_norm_trace)) and min(actual.gradient_norm_trace) > 0
    assert actual.coordinates is not actual.loss_trace
    best_epoch = int(np.argmin(actual.loss_trace))
    assert _fitting.evaluate_alignment(actual.coordinates, target, 2.) == pytest.approx(actual.loss_trace[best_epoch], rel=1e-8)


@pytest.mark.parametrize('backend', ['torch', 'numpy'])
def test_explicit_warm_start_matches_an_uninterrupted_stateless_schedule(monkeypatch, backend):
    if backend == 'torch':
        pytest.importorskip('torch')
    else:
        monkeypatch.setattr(_fitting, 'torch', None)
    target = _target()
    initial = np.random.default_rng(5).normal(scale=.05, size=(6, 2))
    original = initial.copy()
    full = _fitting.fit_disk_embedding(target, initial_coordinates=initial, burn_in_epochs=0, n_epochs=4)
    first_snapshots = []
    _fitting.fit_disk_embedding(target, initial_coordinates=initial, burn_in_epochs=0,
                               n_epochs=2, progress_callback=first_snapshots.append)
    resumed = _fitting.fit_disk_embedding(target, initial_coordinates=first_snapshots[-1].coordinates,
                                         burn_in_epochs=0, n_epochs=2)
    # This checks a constant-step case; warm starts do not restore patience state.
    np.testing.assert_allclose(full.coordinates, resumed.coordinates, atol=1e-12)
    np.testing.assert_array_equal(initial, original)
    assert full.initialization == resumed.initialization == 'provided'


@pytest.mark.parametrize('bad', [np.zeros((5, 2)), np.zeros((6, 3)), np.full((6, 2), np.nan), np.full((6, 2), 1.)])
def test_invalid_initial_coordinates_are_rejected(bad):
    with pytest.raises(ValueError, match='initial_coordinates'):
        _fitting.fit_disk_embedding(_target(), initial_coordinates=bad)


def test_callback_failure_is_not_hidden():
    def callback(progress):
        raise RuntimeError('checkpoint storage failed')
    with pytest.raises(RuntimeError, match='checkpoint storage failed'):
        _fitting.fit_disk_embedding(_target(), n_epochs=2, progress_callback=callback)
    with pytest.raises(ValueError, match='progress_callback'):
        _fitting.fit_disk_embedding(_target(), progress_callback=3)


def test_training_projection_roundoff_is_accepted_for_warm_start():
    initial = np.zeros((6, 2))
    initial[0] = (1 - 1e-5) * np.array([np.cos(.3), np.sin(.3)])
    report = _fitting.fit_disk_embedding(_target(), initial_coordinates=initial, n_epochs=1)
    assert report.initialization == 'provided'
    assert np.isfinite(report.best_loss)


def test_distance_rank_prefers_a_feature_aligned_layout():
    import importlib.util
    file = Path(__file__).resolve().parents[1] / 'examples' / 'compare_spectral_init.py'
    spec = importlib.util.spec_from_file_location('compare_spectral_init', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = np.random.default_rng(0).normal(size=(12, 2))
    aligned = features / (np.linalg.norm(features, axis=1).max() * 4)
    random = np.random.default_rng(1).normal(scale=0.05, size=(12, 2))
    assert module.distance_rank(aligned, features) > module.distance_rank(random, features)


def test_neighbor_purity_is_one_for_a_single_label():
    import importlib.util
    file = Path(__file__).resolve().parents[1] / 'examples' / 'inspect_rare_populations.py'
    spec = importlib.util.spec_from_file_location('inspect_rare_populations', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    coordinates = np.random.default_rng(2).normal(size=(8, 2))
    labels = np.array(['a'] * 8)
    scores = module.neighbor_purity(coordinates, labels, k=3)
    assert scores['a'] == 1
    assert scores['overall'] == 1
