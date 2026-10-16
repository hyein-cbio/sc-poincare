"""Provisional MU default, explicit exact legacy path and safe AnnData resume."""
from copy import deepcopy
from dataclasses import replace
import inspect

import numpy as np
import pytest
from scipy import sparse

ad = pytest.importorskip('anndata')
from sc_poincare import tl, experimental as mu
from sc_poincare._anndata import layout_cells
from sc_poincare._fit_progress import SCHEMA


@pytest.fixture
def torch():
    module=pytest.importorskip('torch')
    old=module.get_num_threads();module.set_num_threads(1)
    yield module
    module.set_num_threads(old)


def data(n=18,d=4):
    return ad.AnnData(np.random.default_rng(74).normal(size=(n,d)))


def test_method_default_sentinel_survives_copy_and_pickle():
    import pickle
    from sc_poincare._mu_anndata import METHOD_DEFAULT
    assert deepcopy(METHOD_DEFAULT) is METHOD_DEFAULT
    assert pickle.loads(pickle.dumps(METHOD_DEFAULT)) is METHOD_DEFAULT


def test_explicit_rfa_restores_every_old_default(monkeypatch):
    from sc_poincare import _anndata
    captured={}
    def legacy(adata,**kwargs):captured.update(kwargs)
    monkeypatch.setattr(_anndata,'_layout_rfa',legacy)
    tl.poincare_map(data(),method='rfa')
    expected=dict(n_neighbors=15,sigma=2.,gamma=2.,n_epochs=250,learning_rate=.01,
        burn_in_epochs=30,min_delta=1e-4,patience=40,max_grad_norm=2.5,
        objective='symmetric_kl',graph_mode='mutual',random_state=0)
    assert all(captured[key]==value for key,value in expected.items())


def test_default_resolves_tested_mu_recipe_and_does_not_write_on_abort(torch,monkeypatch):
    x=data(8)
    captured={}
    class Stop(Exception):pass
    def stop(setup,**kwargs):
        captured.update(config=setup.config,arguments=kwargs)
        raise Stop()
    monkeypatch.setattr(mu,'fit_graph_stress',stop)
    with pytest.raises(Stop):tl.poincare_map(x)
    assert inspect.signature(tl.poincare_map).parameters['method'].default=='mu'
    assert captured['arguments']['epochs']==1500
    assert captured['config'].n_neighbors==7
    assert captured['config'].gamma==mu.GraphStressConfig().gamma
    assert captured['config'].learning_rate==mu.GraphStressConfig().learning_rate
    assert captured['config'].local_bandwidth=='global'
    assert not x.obsm and 'poincare' not in x.uns


def test_mu_default_writes_old_slots_and_separate_teacher_and_roundtrips(torch,tmp_path):
    x=data();original=x.X.copy()
    tl.poincare_map(x,n_epochs=3,n_neighbors=4)
    saved=x.uns['poincare'];p=saved['params']
    assert p['method']=='mu' and p['n_epochs_run']==3
    assert saved['best_epoch']<=3 and len(saved['loss'])==3
    assert saved['loss_includes_epoch_zero'] is False
    assert saved['initial_loss']>=saved['best_loss']
    assert saved['coordinate_space']=='raw_training'
    assert np.all(np.linalg.norm(x.obsm['X_poincare'],axis=1)<1)
    graph=x.obsp['poincare_connectivities'];teacher=x.obsp['poincare_local_teacher']
    assert sparse.issparse(graph) and sparse.issparse(teacher)
    np.testing.assert_allclose(graph.toarray(),graph.toarray().T)
    np.testing.assert_allclose(np.asarray(teacher.sum(axis=1)).ravel(),1)
    assert 'self_mass_median' not in saved['proximity_diagnostics']
    topology=saved['proximity_diagnostics']['topology_reconstruction']
    assert topology['scope']=='unweighted_pre_stitch_reconstruction'
    assert topology['mutual']['required_bridges_for_connection']>=0
    assert saved['proximity_diagnostics']['weighted_local_teacher_components']>=1
    assert 'state' not in saved  # Torch tensors/optimizer never enter H5AD uns.
    np.testing.assert_array_equal(x.X,original)
    path=tmp_path/'mu.h5ad';x.write_h5ad(path)
    loaded=ad.read_h5ad(path)
    assert loaded.uns['poincare']['params']['method']=='mu'
    np.testing.assert_array_equal(loaded.obsm['X_poincare'],x.obsm['X_poincare'])


def test_mu_adapter_matches_the_unchanged_low_level_recipe(torch):
    x=data();tl.poincare_map(x,n_neighbors=4,n_epochs=3)
    setup=mu.prepare_graph_stress(x.X,config=mu.GraphStressConfig(n_neighbors=4))
    expected=mu.fit_graph_stress(setup,epochs=3,random_state=0,chunk_size=256)
    np.testing.assert_array_equal(x.obsm['X_poincare'],expected.coordinates)
    np.testing.assert_array_equal(x.uns['poincare']['loss'],expected.loss_trace[1:])
    assert x.uns['poincare']['initial_loss']==expected.loss_trace[0]


def test_copy_custom_basis_and_existing_scanpy_helpers(torch):
    x=data();x.obsm['X_custom']=x.X.copy()
    y=tl.poincare_map(x,copy=True,key_added='mu2',use_rep='X_custom',n_neighbors=4,n_epochs=3)
    assert 'X_mu2' in y.obsm and 'X_mu2' not in x.obsm
    assert y.uns['mu2']['params']['use_rep']=='X_custom'
    tl.poincare_recenter(y,0,basis='X_mu2')
    tl.poincare_pseudotime(y,basis='X_mu2')
    tl.poincare_lineages(y,n_lineages=2,basis='X_mu2')
    assert y.obs['poincare_pseudotime'].iloc[0]<1e-8
    assert y.obs['poincare_lineage'].nunique()==2


def test_local_radius_and_shared_overrides_are_explicit(torch):
    x=data()
    config=mu.GraphStressConfig(n_neighbors=5,local_bandwidth='local_radius')
    tl.poincare_map(x,mu_config=config,n_epochs=2,n_neighbors=3,gamma=.08,learning_rate=.02)
    p=x.uns['poincare']['params']
    assert p['local_bandwidth']=='local_radius' and p['n_neighbors']==3
    assert p['gamma']==.08 and p['learning_rate']==.02
    assert config.n_neighbors==5  # Caller config is not mutated.


def test_adam_resume_restores_recipe_seed_rep_and_matches_uninterrupted(torch,tmp_path):
    x=data(24,120);x.obsm['X_pca']=np.random.default_rng(2).normal(size=(24,8))
    checkpoints=[]
    tl.poincare_map(x,n_epochs=3,n_neighbors=4,n_pcs=5,random_state=7,
        local_bandwidth='local_radius',checkpoint_every=1,
        progress_callback=lambda p:checkpoints.append(p.checkpoint))
    checkpoint=checkpoints[-1]
    assert checkpoint['schema']==SCHEMA
    path=tmp_path/'state.pt';torch.save(checkpoint,path)
    checkpoint=torch.load(path,weights_only=True)
    continued=data(24,120);continued.obsm['X_pca']=x.obsm['X_pca'].copy()
    tl.poincare_map(continued,resume=checkpoint,n_epochs=6)
    uninterrupted=data(24,120);uninterrupted.obsm['X_pca']=x.obsm['X_pca'].copy()
    tl.poincare_map(uninterrupted,n_epochs=6,n_neighbors=4,n_pcs=5,random_state=7,local_bandwidth='local_radius')
    np.testing.assert_array_equal(continued.obsm['X_poincare'],uninterrupted.obsm['X_poincare'])
    np.testing.assert_array_equal(continued.uns['poincare']['loss'],uninterrupted.uns['poincare']['loss'])
    p=continued.uns['poincare']['params']
    assert p['resume_from_epoch']==3 and p['random_state']==7
    assert p['local_bandwidth']=='local_radius' and p['n_neighbors']==4


@pytest.fixture
def checkpoint(torch):
    x=data();saved=[]
    tl.poincare_map(x,n_epochs=1,n_neighbors=4,checkpoint_every=1,
                    progress_callback=lambda p:saved.append(p.checkpoint))
    return saved[-1]


@pytest.mark.parametrize('change',['row_order','cell_names','feature_values','config','config_object','seed','chunks','corrupt_state','corrupt_header','bare'])
def test_unsafe_resume_rejected_before_graph_build(checkpoint,change,monkeypatch):
    x=data();state=deepcopy(checkpoint);kwargs={}
    if change=='row_order':x=x[::-1].copy()
    elif change=='cell_names':x.obs_names=[f'other{i}' for i in range(x.n_obs)]
    elif change=='feature_values':x.X[0,0]+=.1
    elif change=='config':kwargs['gamma']=.5
    elif change=='config_object':kwargs['mu_config']=mu.GraphStressConfig(n_neighbors=3)
    elif change=='seed':kwargs['random_state']=3
    elif change=='chunks':kwargs['batch_size']=3
    elif change=='corrupt_state':state['state']['u'][0,0]+=.1
    elif change=='corrupt_header':state['provenance']['resolved_rep']='other'
    elif change=='bare':state=state['state']
    monkeypatch.setattr(mu,'prepare_graph_stress',lambda *a,**k:pytest.fail('Invalid resume must fail before graph preparation'))
    with pytest.raises(ValueError):tl.poincare_map(x,resume=state,n_epochs=3,**kwargs)
    assert 'poincare' not in x.uns


def test_resume_requires_explicit_total_epoch_and_rejects_completed_target(checkpoint):
    with pytest.raises(ValueError,match='explicitly'):tl.poincare_map(data(),resume=checkpoint)
    with pytest.raises(ValueError,match='exceed'):tl.poincare_map(data(),resume=checkpoint,n_epochs=1)


def test_duplicate_ids_fit_but_are_not_verified_resume_identity(torch):
    x=data();x.obs_names=['same']*x.n_obs;states=[]
    tl.poincare_map(x,n_epochs=1,n_neighbors=4,progress_callback=lambda p:states.append(p.checkpoint))
    assert not x.uns['poincare']['provenance']['unique_nonempty_obs_names']
    with pytest.raises(ValueError,match='unique'):
        tl.poincare_map(x,resume=states[-1],n_epochs=2)


def test_callbacks_are_isolated_and_abort_does_not_overwrite_existing_results(torch):
    a=data();b=data()
    def mutate(p):
        assert isinstance(p,tl.PoincareProgress)
        p.current[:]=99;p.best[:]=99
        p.checkpoint['state']['u'].zero_();p.checkpoint['provenance'].clear()
    tl.poincare_map(a,n_neighbors=4,n_epochs=3,progress_callback=mutate,checkpoint_every=1)
    tl.poincare_map(b,n_neighbors=4,n_epochs=3)
    np.testing.assert_array_equal(a.obsm['X_poincare'],b.obsm['X_poincare'])
    original=a.obsm['X_poincare'].copy();saved=deepcopy(a.uns['poincare'])
    def stop(p):raise RuntimeError('stop requested')
    with pytest.raises(RuntimeError):tl.poincare_map(a,n_epochs=5,progress_callback=stop)
    np.testing.assert_array_equal(a.obsm['X_poincare'],original)
    assert a.uns['poincare']['last_checkpoint_sha256']==saved['last_checkpoint_sha256']


@pytest.mark.parametrize('method',['mu','rfa'])
@pytest.mark.parametrize('change',['features','ids'])
def test_external_input_mutation_in_callback_is_detected(torch,method,change):
    x=data()
    def mutate(p):
        if change=='features':x.X[0,0]+=.1
        else:x.obs_names=[f'changed{i}' for i in range(x.n_obs)]
    with pytest.raises(ValueError,match='changed during fitting'):
        tl.poincare_map(x,method=method,n_epochs=2,n_neighbors=4,progress_callback=mutate)
    assert 'X_poincare' not in x.obsm and 'poincare' not in x.uns


def test_root_display_is_not_an_optimizer_checkpoint(torch):
    a=data();b=data();states=[]
    tl.poincare_map(a,n_epochs=2,n_neighbors=4,root=0,progress_callback=lambda p:states.append(p.checkpoint))
    tl.poincare_map(b,n_epochs=2,n_neighbors=4)
    assert np.linalg.norm(a.obsm['X_poincare'][0])<1e-10
    assert a.uns['poincare']['coordinate_space']=='root_isometry'
    state=states[-1]['state']
    np.testing.assert_array_equal(state['best'].numpy(),b.obsm['X_poincare'])
    assert a.uns['poincare']['raw_best_sha256']==b.uns['poincare']['raw_best_sha256']


@pytest.mark.parametrize('kwargs',[dict(sigma=2.),dict(patience=40),dict(burn_in_epochs=0),
    dict(min_delta=0.),dict(max_grad_norm=None),dict(objective='symmetric_kl'),dict(graph_mode='union'),
    dict(initial_coordinates=np.zeros((18,2)))])
def test_rfa_only_arguments_never_silently_apply_to_mu(kwargs):
    with pytest.raises(ValueError,match="method='rfa'"):tl.poincare_map(data(),n_epochs=1,**kwargs)


@pytest.mark.parametrize('kwargs',[dict(method='other'),dict(method=None),dict(metric='cosine'),dict(n_components=3)])
def test_invalid_default_method_settings(kwargs):
    with pytest.raises(ValueError):tl.poincare_map(data(),n_epochs=1,**kwargs)


def test_default_needs_torch_but_rfa_numpy_path_is_explicit(monkeypatch):
    def absent():raise ImportError('missing Torch')
    monkeypatch.setattr(mu,'_torch',absent)
    with pytest.raises(ImportError,match="method='rfa'"):tl.poincare_map(data(),n_epochs=1)


def test_complex_mu_features_are_not_silently_cast(torch):
    x=ad.AnnData(np.ones((18,3),complex))
    with pytest.raises(ValueError,match='complex'):tl.poincare_map(x,n_epochs=1)


def test_method_switch_cleans_only_its_owned_teacher(torch):
    x=data();tl.poincare_map(x,n_epochs=1,n_neighbors=4)
    assert 'poincare_local_teacher' in x.obsp
    tl.poincare_map(x,method='rfa',n_epochs=1,n_neighbors=4)
    assert 'poincare_local_teacher' not in x.obsp
    assert x.uns['poincare']['method']=='rfa'


def test_progress_display_provenance_and_annotation_freshness(torch):
    from sc_poincare.diagnostics import result_status
    x=data();tl.poincare_map(x,n_epochs=2,n_neighbors=4)
    raw_sha=x.uns['poincare']['raw_best_sha256']
    tl.poincare_pseudotime(x)
    assert result_status(x)['annotations']['poincare_pseudotime']=='current'
    tl.poincare_recenter(x,0)
    status=result_status(x)
    assert status['coordinate_space']=='root_isometry' and status['coordinates_match_record']
    assert status['annotations']['poincare_pseudotime']=='stale'
    assert x.uns['poincare']['raw_best_sha256']==raw_sha
    tl.poincare_pseudotime(x)
    assert result_status(x)['annotations']['poincare_pseudotime']=='current'
    x.obs.loc[x.obs_names[2],'poincare_pseudotime']+=1.
    assert result_status(x)['annotations']['poincare_pseudotime']=='stale'
    tl.poincare_pseudotime(x)
    assert result_status(x)['annotations']['poincare_pseudotime']=='current'
    x.obsm['X_poincare'][1,0]+=.0001
    assert not result_status(x)['coordinates_match_record']


def test_derived_provenance_survives_h5ad_and_detects_edited_values(torch,tmp_path):
    from sc_poincare.diagnostics import result_status
    x=data();tl.poincare_map(x,n_epochs=2,n_neighbors=4)
    tl.poincare_pseudotime(x,root=0);tl.poincare_lineages(x,n_lineages=2)
    path=tmp_path/'derived.h5ad';x.write_h5ad(path);loaded=ad.read_h5ad(path)
    status=result_status(loaded)
    assert status['annotations']['poincare_pseudotime']=='current'
    assert status['annotations']['poincare_lineage']=='current'
    label=loaded.obs['poincare_lineage'].iloc[0]
    loaded.obs.loc[loaded.obs_names[0],'poincare_lineage']='1' if label=='0' else '0'
    assert result_status(loaded)['annotations']['poincare_lineage']=='stale'


def test_old_embeddings_remain_readable_without_fabricated_provenance():
    from sc_poincare.diagnostics import result_status
    x=data();x.obsm['X_poincare']=np.zeros((x.n_obs,2))
    x.uns['poincare']={'params':{'objective':'symmetric_kl'}}
    x.obs['poincare_pseudotime']=0.
    status=result_status(x)
    assert status['method']=='rfa' and status['coordinates_match_record'] is None
    assert status['annotations']['poincare_pseudotime']=='unknown'
    tl.poincare_pseudotime(x)
    assert x.obs['poincare_pseudotime'].sum()==0


def test_seed_generator_is_supported_and_effective_seed_is_recorded(torch):
    x=data();tl.poincare_map(x,n_neighbors=4,n_epochs=1,random_state=np.random.default_rng(7))
    seed=x.uns['poincare']['params']['random_state']
    assert isinstance(seed,int)
    other=data();tl.poincare_map(other,n_neighbors=4,n_epochs=1,random_state=seed)
    np.testing.assert_array_equal(x.obsm['X_poincare'],other.obsm['X_poincare'])


def test_rfa_progress_and_coordinate_warm_start_are_new_runs(torch):
    x=data();events=[]
    initial=np.random.default_rng(8).normal(scale=.03,size=(x.n_obs,2))
    tl.poincare_map(x,method='rfa',n_epochs=2,n_neighbors=4,initial_coordinates=initial,
                   progress_callback=lambda p:events.append(p))
    assert [p.epoch for p in events]==[0,1,2]
    assert all(p.checkpoint is None and p.method=='rfa' for p in events)
    np.testing.assert_array_equal(events[0].current,initial)
    assert x.uns['poincare']['params']['initialization']=='provided'
