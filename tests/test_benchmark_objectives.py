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
        assert all(r['epochs_run']==2 for r in report['records'])
        arrays=[np.load(out.with_name(f'{out.stem}_{mode}_seed0.npz')) for mode in modes]
        for z in arrays:
            np.testing.assert_array_equal(z['initial_coordinates'],arrays[0]['initial_coordinates'])
            assert np.isfinite(z['coordinates']).all()
    for mode in modes:
        a=np.load(outputs[0].with_name(f'run0_{mode}_seed0.npz'))['coordinates']
        b=np.load(outputs[1].with_name(f'run1_{mode}_seed0.npz'))['coordinates']
        np.testing.assert_array_equal(a,b)


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
