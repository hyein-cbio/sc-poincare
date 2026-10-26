"""Smoke checks for the public, data-independent experiment driver."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def _module():
    file=Path(__file__).resolve().parents[1]/'examples'/'benchmark_objectives.py'
    spec=importlib.util.spec_from_file_location('objective_benchmark',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _research_module(name, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'scanpy', SimpleNamespace())
    file = Path(__file__).resolve().parents[1] / 'examples' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('script,helper', [
    ('benchmark_objectives', '_neighbor_purity'),
    ('inspect_pbmc3k_frozen', 'purity'),
    ('inspect_rare_populations', 'neighbor_purity'),
])
def test_purity_excludes_self_with_coincident_coordinates(script, helper, monkeypatch):
    module = _research_module(script, monkeypatch)
    scores = getattr(module, helper)(np.zeros((6, 2)), np.arange(6), k=2)
    assert scores['overall'] == 0.
    assert all(value == 0. for value in scores.values())


@pytest.mark.parametrize('script,helper', [
    ('compare_spectral_init', 'distance_rank'),
    ('inspect_pbmc3k_frozen', 'feature_rank'),
])
def test_feature_rank_uses_cdist_and_accepts_reusable_distances(script, helper, monkeypatch):
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    from sc_poincare._disk import pairwise_disk_length
    module = _research_module(script, monkeypatch)
    rng = np.random.default_rng(8)
    features = rng.normal(size=(6, 4))
    coordinates = rng.normal(scale=.1, size=(6, 2))
    reference = np.linalg.norm(features[:, None] - features[None, :], axis=-1)
    mask = ~np.eye(len(features), dtype=bool)
    expected = spearmanr(reference[mask], pairwise_disk_length(coordinates)[mask]).statistic
    calls = []

    def tracked_cdist(left, right):
        calls.append((left.shape, right.shape))
        return cdist(left, right)

    monkeypatch.setattr(module, 'cdist', tracked_cdist)
    rank = getattr(module, helper)
    assert rank(coordinates, features) == pytest.approx(expected)
    assert rank(coordinates, features, feature_distance=reference) == pytest.approx(expected)
    assert calls == [((6, 4), (6, 4))]


def test_spectral_comparison_reuses_feature_distances_across_seeds(monkeypatch):
    from types import SimpleNamespace
    from scipy.spatial.distance import cdist
    module = _research_module('compare_spectral_init', monkeypatch)
    calls = []

    def tracked_cdist(left, right):
        calls.append(left.shape)
        return cdist(left, right)

    def fit(target, *, initial_coordinates, **kwargs):
        return SimpleNamespace(coordinates=initial_coordinates, best_loss=1.,
                               stop_reason='epoch_limit', n_epochs=1)

    monkeypatch.setattr(module, 'cdist', tracked_cdist)
    monkeypatch.setattr(module, 'fit_disk_embedding', fit)
    result = module.compare(np.random.default_rng(8).normal(size=(8, 3)),
                            n_neighbors=3, sigma=1., epochs=1, seeds=[0, 1])
    assert len(result['runs']) == 4
    assert calls == [(8, 3)]


@pytest.mark.parametrize('script', ['inspect_pbmc3k_frozen', 'inspect_pbmc3k_graph_stress'])
def test_pbmc_drivers_reuse_feature_distances_without_downloading(script, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scipy.spatial.distance import cdist
    frozen = _research_module('inspect_pbmc3k_frozen', monkeypatch)
    module = frozen if script.endswith('_frozen') else _research_module(script, monkeypatch)
    features = np.random.default_rng(8).normal(size=(20, 3))
    coordinates = np.random.default_rng(9).normal(scale=.05, size=(20, 2))
    names = np.array([str(i) for i in range(20)])
    labels = pd.Series(['a'] * 10 + ['b'] * 10, index=names)
    monkeypatch.setattr(frozen, 'prepare', lambda: (names, features, labels))
    report = SimpleNamespace(coordinates=coordinates, best_loss=1., best_epoch=1,
                             current_loss=1., stop_reason='epoch_limit', n_epochs=1)
    if module is frozen:
        monkeypatch.setattr(frozen, 'fit_disk_embedding', lambda *a, **kw: report)
    else:
        monkeypatch.setattr(module, '_frozen', lambda: frozen)
        monkeypatch.setattr(module, 'prepare_graph_stress', lambda *a, **kw: object())
        monkeypatch.setattr(module, 'fit_graph_stress', lambda *a, **kw: report)
    calls = []

    def tracked_cdist(left, right):
        calls.append(left.shape)
        return cdist(left, right)

    monkeypatch.setattr(frozen, 'cdist', tracked_cdist)
    monkeypatch.setattr(module, 'cdist', tracked_cdist)
    output = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', [script, '--seeds', '0', '1', '--output', str(output)])
    module.main()
    assert len(json.loads(output.read_text())['runs']) == 2
    assert calls == [(20, 3)]


def test_benchmark_uses_common_initialization_and_never_fits_labels(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    data=pd.DataFrame(np.random.default_rng(4).normal(size=(12,3)),columns=['a','b','c'])
    data['label']=['one']*12
    source=tmp_path/'input.csv';data.to_csv(source,index=False)
    module=_module();modes=['legacy_floor_1e-12','symmetric_kl','conditional_symmetric_kl']
    outputs=[]
    for i in [0,1]:
        if i:data['label']=['different']*12;data.to_csv(source,index=False)
        out=tmp_path/f'run{i}.json';outputs.append(out)
        monkeypatch.setattr(sys,'argv',['benchmark',str(source),'--output',str(out),
            '--seeds','0','--epochs','2','--n-neighbors','4','--evaluation-k','4','5',
            '--threads','1','--include-legacy-floor'])
        module.main()
        report=json.loads(out.read_text());assert len(report['records'])==3
        assert [r['objective'] for r in report['records']]==modes
        for record in report['records']:
            feature_rank=record['evaluation']['feature']['global_distance_rank']
            assert np.isfinite(feature_rank)
            assert set(record['evaluation'])=={'4','5','feature'}
            if record['objective']=='legacy_floor_1e-12':
                assert record['gradient_norm_trace'] is None
            else:
                assert len(record['gradient_norm_trace'])==record['epochs_run']==2
                assert min(record['gradient_norm_trace'])>0
            assert record['label_neighbor_purity_k']==4
            assert record['label_neighbor_purity']['overall']==1
        label='one' if i==0 else 'different'
        assert report['label_counts']=={label:12}
        provenance=report['provenance']
        assert provenance['import_file'].endswith('sc_poincare/__init__.py')
        assert len(provenance['git_revision'])==40
        assert isinstance(provenance['git_dirty'], bool)
        assert len(provenance['git_diff_sha256'])==64
        assert provenance['versions']['numpy']
        assert all(r['label_neighbor_purity'][label]==1 for r in report['records'])
        assert all(r['epochs_run']==2 for r in report['records'])
        arrays=[np.load(out.with_name(f'{out.stem}_{mode}_seed0.npz')) for mode in modes]
        for z in arrays:
            np.testing.assert_array_equal(z['initial_coordinates'],arrays[0]['initial_coordinates'])
            assert np.isfinite(z['coordinates']).all()
            assert z['current_coordinates'].shape==z['coordinates'].shape
            assert np.isfinite(z['current_coordinates']).all()
    for mode in modes:
        a=np.load(outputs[0].with_name(f'run0_{mode}_seed0.npz'))
        b=np.load(outputs[1].with_name(f'run1_{mode}_seed0.npz'))
        np.testing.assert_array_equal(a['coordinates'],b['coordinates'])
        np.testing.assert_array_equal(a['current_coordinates'],b['current_coordinates'])


def test_provenance_uses_the_driver_repository(tmp_path, monkeypatch):
    module=_module()
    monkeypatch.chdir(tmp_path)
    provenance=module._run_provenance()
    repo=Path(__file__).resolve().parents[1]
    expected=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert provenance['git_revision']==expected
    assert provenance['git_diff_sha256']


def test_legacy_reference_accumulates_global_gradients_across_chunks():
    pytest.importorskip('torch')
    from types import SimpleNamespace
    module=_module();p=np.random.default_rng(3).uniform(.1,1.,size=(7,7));p/=p.sum(1,keepdims=True)
    xy=np.random.default_rng(9).normal(scale=.05,size=(7,2))
    results=[]
    for chunk in [1,7]:
        args=SimpleNamespace(batch_size=chunk,epochs=2,gamma=.5,learning_rate=.01,burn_in=0)
        results.append(module._legacy_reference(p,xy,args))
    np.testing.assert_allclose(results[0][0],results[1][0],atol=1e-10)
    np.testing.assert_allclose(results[0][1],results[1][1],atol=1e-10)
