"""Audit existing pairs; keep failures and never refit for display."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sc_poincare.evaluation import evaluate_neighborhoods, compare_neighborhoods


def cli():
    spec = importlib.util.spec_from_file_location('visibility_cli',
        Path(__file__).resolve().parents[1] / 'examples/evaluate_graph_visibility.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def study(tmp_path, *, seed=0):
    root = tmp_path / 'study'
    folder = root / 'toy'; folder.mkdir(parents=True)
    features = np.arange(4.)[:, None]
    labels = np.array(['A', 'A', 'B', 'B'])
    ids = np.array(['a', 'b', 'c', 'd'])
    before = np.array([[0., 0.], [.1, 0.], [.3, 0.], [.5, 0.]])
    after = before * 1.2
    np.savez(folder / 'input.npz', features=features, labels=labels, cell_ids=ids)
    np.savez(folder / f'seed{seed}.npz', mutual=before, union=after)
    a = evaluate_neighborhoods(features, before, labels, cell_ids=ids, ks=[1])
    b = evaluate_neighborhoods(features, after, labels, cell_ids=ids, ks=[1])
    summary = dict(status='completed', planned_fits=2, runs=[{}, {}],
                   comparisons=[dict(dataset='toy', seed=seed, mutual=a.summary, union=b.summary,
                                     comparison=compare_neighborhoods(a, b))])
    (root / 'summary.json').write_text(json.dumps(summary))
    return root


def test_cli_audits_pair_and_writes_fixed_size_figure(tmp_path):
    from PIL import Image
    source = study(tmp_path)
    out = tmp_path / 'report'
    result = cli().main([str(source), '--output', str(out)])
    assert result['cases']['toy'][0]['delta']['macro_width_q90_px'] > 0
    assert result['cases']['toy'][0]['structural_delta']['cell_order_verified']
    assert len(result['source_sha256']['visibility.py']) == 64
    with Image.open(out / 'toy_visibility.png') as image:
        assert image.size == (1500, 820)
    assert (out / 'visibility_overview.png').exists()
    with pytest.raises(SystemExit):
        cli().main([str(source), '--output', str(out)])


def test_cli_rejects_modified_coordinates_before_writing(tmp_path):
    source = study(tmp_path)
    np.savez(source / 'toy/seed0.npz', mutual=np.zeros((4, 2)), union=np.zeros((4, 2)))
    out = tmp_path / 'report'
    with pytest.raises(ValueError, match='hash mismatch'):
        cli().main([str(source), '--output', str(out)])
    assert not out.exists()


def test_cli_refuses_duplicate_pairs_and_nonzero_representative(tmp_path):
    source = study(tmp_path)
    with pytest.raises(ValueError, match='Duplicate'):
        cli().main([str(source), str(source), '--output', str(tmp_path / 'duplicate')])
    other = tmp_path / 'other'; other.mkdir()
    no_seed0 = study(other, seed=1)
    with pytest.raises(ValueError, match='Seed 0'):
        cli().main([str(no_seed0), '--output', str(tmp_path / 'no-seed0')])


@pytest.mark.parametrize('name,seed', [('../escape', 0), ('/tmp/escape', 0), ('..\\escape', 0),
                                      ('toy', '../1'), ('toy', -1)])
def test_cli_rejects_unsafe_case_paths_and_seeds_before_writing(tmp_path, name, seed):
    source = study(tmp_path)
    path = source / 'summary.json'
    data = json.loads(path.read_text())
    data['comparisons'][0].update(dataset=name, seed=seed)
    path.write_text(json.dumps(data))
    out = tmp_path / 'output'
    with pytest.raises(ValueError, match='safe path'):
        cli().main([str(source), '--output', str(out)])
    assert not out.exists()


@pytest.mark.parametrize('structured', [False, True])
def test_cli_preserves_failure_without_inventing_coordinates(tmp_path, structured):
    source = tmp_path / 'failure'; source.mkdir()
    summary = dict(status='failed', planned_fits=6, runs=[], comparisons=[],
                   active=dict(dataset='stress', stage='preparation'),
                   error=dict(type='ValueError', message='Gaussian underflow'))
    if structured:
        summary = dict(status='partial', planned_fits=6, runs=[], comparisons=[], active=None,
                       failures=[dict(active=summary['active'], error=summary['error'], status='failed')])
    (source / 'summary.json').write_text(json.dumps(summary))
    out = tmp_path / 'report'
    result = cli().main([str(source), '--output', str(out)])
    assert result['cases'] == {}
    assert result['failures'][0]['active']['dataset'] == 'stress'
    assert (out / 'visibility_overview.png').exists()
    assert not (out / 'stress_visibility.png').exists()
