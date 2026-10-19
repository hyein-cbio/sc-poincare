"""Smoke checks for the public, data-independent experiment driver."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def _module():
    file=Path(__file__).resolve().parents[1]/'examples'/'benchmark_objectives.py'
    spec=importlib.util.spec_from_file_location('objective_benchmark',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


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
