"""Research-driver checks without real datasets or optimizer runs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_driver(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'examples' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('residual', [0., .1])
def test_paul_reference_keeps_pinned_features_even_if_pca_reconstruction_matches(residual, tmp_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    driver = load_driver('run_paul_landmark_reference')
    monkeypatch.setattr(driver, '__file__', str(tmp_path / 'examples' / 'driver.py'))
    folder = tmp_path / 'data' / 'paul'
    folder.mkdir(parents=True)
    (tmp_path / 'docs').mkdir()
    features = np.random.default_rng(17).normal(size=(19, 4))
    np.save(folder / 'features.npy', features)
    np.savez(folder / 'pca.npz', components=np.eye(4), mean=np.zeros(4))
    frame = pd.DataFrame(features + residual)
    frame['label'] = 'synthetic'
    frame.to_csv(folder / 'Paul.csv', index=False, float_format='%.17g')
    # Pin a read-back matrix so equality does not depend on CSV rounding.
    matrix = pd.read_csv(folder / 'Paul.csv').iloc[:, :-1].to_numpy(dtype=np.float64)
    if residual == 0.:
        features = matrix.copy()
        np.save(folder / 'features.npy', features)
    monkeypatch.setattr(driver, 'SOURCE_SHA256', driver._sha256(folder / 'Paul.csv'))
    monkeypatch.setattr(driver, 'FEATURE_FILE_SHA256', driver._sha256(folder / 'features.npy'))
    monkeypatch.setattr(driver, 'FEATURE_SHA256', driver._array_sha256(features))
    protocol = SimpleNamespace(feature_sha256=driver.FEATURE_SHA256, max_epochs=0)
    monkeypatch.setattr(driver, 'freeze_comparison', lambda *a, **k: protocol)

    def compare(actual, frozen, config):
        np.testing.assert_array_equal(actual, features)
        return SimpleNamespace(report={'primary_value': 0., 'metrics': {'column_objective_gradient_relative_l2': .5}})

    monkeypatch.setattr(driver, 'compare_landmark_reference', compare)
    monkeypatch.setattr(driver, 'write_study_tree', lambda *a: None)
    monkeypatch.setattr(driver, 'reevaluate_study_tree', lambda *a: {'primary_abs_difference': 0.})
    driver.main()
    report = json.loads((tmp_path / 'docs' / 'paul-landmark-reference.json').read_text())
    assert report['transform_pin']['recomputed_scores_used'] is False
    if residual == 0.:
        assert report['transform_pin']['recomputed_array_sha256'] == driver.FEATURE_SHA256
        assert report['transform_pin']['max_abs_residual_versus_recorded_features'] == 0.


def test_saved_optimizer_screen_deltas_and_guardrails_match_the_driver():
    pytest.importorskip('pandas')
    driver = load_driver('run_adam_rmsprop_screen')
    record = json.loads((ROOT / 'docs' / 'adam-rmsprop-screen.json').read_text())
    for run in record['runs']:
        change = driver.deltas(run['adam']['scores'], run['rmsprop']['scores'], run['dataset'])
        assert change == run['rmsprop_minus_adam']
        for arm in ('adam', 'rmsprop'):
            assert driver.compact(run[arm]['scores']) == run[arm]['scores']
        checks = driver.guardrail_passes(change, run['dataset'])
        assert checks == run['guardrails']
        assert all(checks.values()) == run['guardrails_pass']
    assert record['seed0_guardrails_pass'] == all(run['guardrails_pass'] for run in record['runs'])


def test_optimizer_screen_writes_population_scores_without_running_fits(tmp_path, monkeypatch):
    pytest.importorskip('pandas')
    driver = load_driver('run_adam_rmsprop_screen')
    archived = json.loads((ROOT / 'docs' / 'adam-rmsprop-screen.json').read_text())
    scores = {run['dataset']: run['adam']['scores'] for run in archived['runs']}
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'docs').mkdir()
    coordinates = np.zeros((19, 2))
    monkeypatch.setattr(driver, 'load_dataset', lambda name: (np.zeros((19, 4)), np.repeat('synthetic', 19)))
    monkeypatch.setattr(driver, 'prepare_graph_stress', lambda *a, **k: object())
    monkeypatch.setattr(driver, 'classical_jitter', lambda *a: coordinates)
    monkeypatch.setattr(driver, 'fit_arm', lambda *a: (dict(
        best=coordinates, current=coordinates, best_epoch=0, best_loss=1., loss_trace=[1., 1.]), 0.))
    monkeypatch.setattr(driver, 'score_coordinates', lambda features, labels, points, dataset: scores[dataset])
    driver.main()
    written = json.loads((tmp_path / 'docs' / 'adam-rmsprop-screen.json').read_text())
    assert written['fits_executed'] == 4
    assert written['confirmation_seeds_executed'] is False
    for run in written['runs']:
        for arm in ('adam', 'rmsprop'):
            assert run[arm]['scores'] == scores[run['dataset']]
            assert set(run[arm]) == set(archived['runs'][0][arm])
    with pytest.raises(SystemExit, match='Refusing to overwrite'):
        driver.main()
