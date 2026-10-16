"""Demo routing must not pass its historical RFA recipe into the MU default."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('method',['mu','rfa'])
def test_demo_routes_method_specific_options_without_downloads(method,monkeypatch):
    scanpy=pytest.importorskip('scanpy')
    import anndata as ad
    spec=importlib.util.spec_from_file_location('demo_methods',
        Path(__file__).resolve().parents[1]/'examples/run_scanpy_demo.py')
    demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    x=ad.AnnData(np.random.default_rng(1).normal(size=(20,4)))
    monkeypatch.setattr(demo,'load_dataset',lambda name:(x,None,dict(use_rep='X',n_neighbors=15,sigma=2.,gamma=2.)))
    recorded={}
    class Stop(Exception):pass
    def capture(adata,**kwargs):
        recorded.update(kwargs);raise Stop()
    monkeypatch.setattr(demo.dr.tl,'poincare_map',capture)
    with pytest.raises(Stop):demo.main(['--method',method])
    assert recorded['method']==method
    if method=='mu':
        assert not set(recorded)&{'sigma','gamma','burn_in_epochs','patience','learning_rate','n_epochs'}
    else:
        assert recorded['n_epochs']==recorded['patience']==200
        assert recorded['sigma']==2. and recorded['gamma']==2.
