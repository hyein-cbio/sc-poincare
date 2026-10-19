"""Numerical, restart and explicit-input contracts for the opt-in API."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp

from sc_poincare import experimental as method


@pytest.fixture
def features():
    return np.random.default_rng(421).normal(size=(19, 4))


@pytest.fixture
def setup(features):
    return method.prepare_graph_stress(features, config=method.GraphStressConfig(n_neighbors=4))


@pytest.fixture
def torch():
    module = pytest.importorskip("torch")
    old = module.get_num_threads()
    module.set_num_threads(1)
    yield module
    module.set_num_threads(old)


@pytest.mark.parametrize("bandwidth", ["global", "local_radius"])
@pytest.mark.parametrize("geometry", ["poincare", "euclidean"])
def test_exact_objective_and_chunk_gradient(features, bandwidth, geometry, torch):
    setup = method.prepare_graph_stress(features, config=method.GraphStressConfig(n_neighbors=4, local_bandwidth=bandwidth))
    u = torch.tensor(method._tangent_from_disk(setup.base_initial), requires_grad=True)
    t, p = torch.tensor(setup.target), torch.tensor(setup.teacher)
    chunked = method._objective(u, t, p, setup.config, geometry, 3, backward=True)
    gradient = u.grad.clone()
    full = method._objective(u, t, p, setup.config, geometry, len(u), backward=True)
    np.testing.assert_allclose(chunked, full, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(gradient, u.grad, rtol=1e-12, atol=1e-12)
    x = setup.base_initial if geometry == "poincare" else u.detach().numpy()
    distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    if geometry == "poincare":
        r2 = (x*x).sum(1)
        distance = 2*np.arcsinh(distance / np.sqrt((1-r2[:,None])*(1-r2[None,:])))
    np.fill_diagonal(distance, 0)
    c = setup.config
    global_loss = (((distance-setup.target)/c.distance_scale)**2 / np.maximum(setup.target/c.distance_scale,c.relative_floor)**2).sum()/(len(x)*(len(x)-1))
    logits = -distance/c.gamma
    np.fill_diagonal(logits, -np.inf)
    logq = logits-logsumexp(logits,axis=1)[:,None]
    np.fill_diagonal(logq,0)
    local_loss = -(setup.teacher*logq).sum()/len(x)
    np.testing.assert_allclose(full, [global_loss+c.local_weight*local_loss,global_loss,local_loss], rtol=1e-12)


def test_sgd_matches_adam_at_the_same_start_and_is_not_the_default(setup, torch):
    adam = method.fit_graph_stress(setup, epochs=5, chunk_size=4, random_state=0)
    sgd = method.fit_graph_stress(setup, epochs=5, chunk_size=4, random_state=0, optimizer_name="sgd")
    euclidean = method.fit_graph_stress(setup, epochs=0, chunk_size=4, random_state=0, geometry="euclidean")
    np.testing.assert_array_equal(adam.initial_tangent, sgd.initial_tangent)
    np.testing.assert_array_equal(adam.initial_tangent, euclidean.initial_tangent)
    assert adam.loss_trace[0] == sgd.loss_trace[0]
    assert adam.loss_trace[0] != euclidean.loss_trace[0]
    assert not np.allclose(adam.loss_trace, sgd.loss_trace)
    assert adam.best_epoch == int(adam.loss_trace.argmin())
    assert sgd.best_epoch == int(sgd.loss_trace.argmin())
    assert any("exp_avg" in slot for slot in adam.state["optimizer"]["state"].values())
    assert all("exp_avg" not in slot for slot in sgd.state["optimizer"]["state"].values())
    assert "optimizer" not in adam.state["fit_configuration"]
    assert sgd.state["fit_configuration"]["optimizer"] == "sgd"
    with pytest.raises(ValueError, match="mismatch"):
        method.fit_graph_stress(setup, epochs=6, chunk_size=4, resume=adam.state, optimizer_name="sgd")


def test_lorentz_starts_at_the_same_disk_distances_and_stays_opt_in(setup):
    disk = method.fit_graph_stress(setup, epochs=5, chunk_size=4, random_state=1)
    lorentz = method.fit_graph_stress(setup, epochs=5, chunk_size=4, random_state=1, geometry="lorentz")
    origin = method.fit_graph_stress(setup, epochs=0, chunk_size=4, random_state=1, geometry="lorentz")
    assert disk.geometry == "poincare"
    assert method.fit_graph_stress(setup, epochs=0).geometry == "poincare"
    np.testing.assert_array_equal(disk.initial, lorentz.initial)
    np.testing.assert_allclose(origin.coordinates, origin.initial, atol=1e-12)
    np.testing.assert_allclose(disk.loss_trace[0], lorentz.loss_trace[0], rtol=1e-9, atol=1e-9)
    assert np.max(np.linalg.norm(lorentz.coordinates, axis=1)) < 1
    assert not np.allclose(disk.loss_trace, lorentz.loss_trace)
    assert disk.best_epoch == int(disk.loss_trace.argmin())
    assert lorentz.best_epoch == int(lorentz.loss_trace.argmin())
    with pytest.raises(ValueError, match="mismatch"):
        method.fit_graph_stress(setup, epochs=6, chunk_size=4, random_state=1, geometry="lorentz", resume=disk.state)


@pytest.mark.parametrize("geometry", ["poincare", "euclidean"])
def test_resume_preserves_adam_and_own_best(setup, torch, tmp_path, geometry):
    full = method.fit_graph_stress(setup, epochs=20, geometry=geometry, chunk_size=5)
    first = method.fit_graph_stress(setup, epochs=10, geometry=geometry, chunk_size=5)
    path=tmp_path/'state.pt'
    torch.save(first.state,path)
    state=torch.load(path,weights_only=True)
    resumed=method.fit_graph_stress(setup,epochs=20,geometry=geometry,chunk_size=5,resume=state)
    np.testing.assert_array_equal(full.loss_trace,resumed.loss_trace)
    np.testing.assert_array_equal(full.current,resumed.current)
    np.testing.assert_array_equal(full.coordinates,resumed.coordinates)
    assert full.best_epoch == full.loss_trace.argmin()
    assert full.best_loss == full.loss_trace.min()
    assert resumed.state['checkpoint_sha256'] == full.state['checkpoint_sha256']
    zero=method.fit_graph_stress(setup,epochs=0,geometry=geometry)
    assert zero.best_epoch == 0 and len(zero.loss_trace)==1
    np.testing.assert_array_equal(zero.coordinates,zero.current)


def test_callback_isolation_and_chunk_trajectory(setup, torch):
    epochs=[]
    def mutate(progress):
        epochs.append(progress.epoch)
        progress.current[:]=999
        progress.best[:]=999
        progress.state['u'].zero_()
        progress.state['metadata'].clear()
    a=method.fit_graph_stress(setup,epochs=6,chunk_size=3,callback=mutate,checkpoint_every=2)
    b=method.fit_graph_stress(setup,epochs=6,chunk_size=19)
    assert epochs==[0,2,4,6]
    np.testing.assert_allclose(a.current,b.current,atol=1e-13,rtol=1e-12)
    assert a.state['metadata'] and np.max(a.current)<1


def test_tampered_setup_and_resume_rejected(setup, torch):
    first=method.fit_graph_stress(setup,epochs=2)
    damaged=deepcopy(first.state);damaged['u'][0,0]+=1
    with pytest.raises(ValueError,match='integrity'):
        method.fit_graph_stress(setup,epochs=3,resume=damaged)
    for kwargs in [dict(geometry='euclidean'),dict(random_state=2),dict(chunk_size=3)]:
        with pytest.raises(ValueError,match='mismatch'):
            method.fit_graph_stress(setup,epochs=3,resume=first.state,**kwargs)
    with pytest.raises(ValueError,match='exceed'):
        method.fit_graph_stress(setup,epochs=2,resume=first.state)
    with pytest.raises(ValueError,match='configuration'):
        method.fit_graph_stress(replace(setup,config=replace(setup.config,distance_scale=3)),epochs=1)
    for name in ['target','teacher','base_initial']:
        array=getattr(setup,name).copy();array.flat[0]+=1
        with pytest.raises(ValueError,match='changed'):
            method.fit_graph_stress(replace(setup,**{name:array}),epochs=1)
    metadata=deepcopy(setup.metadata);metadata['source_hashes']['experimental.py']='changed'
    with pytest.raises(ValueError,match='source hashes'):
        method.fit_graph_stress(replace(setup,metadata=metadata),epochs=1)


@pytest.mark.parametrize('change', [dict(n_neighbors=True),dict(n_neighbors=1.5),dict(n_neighbors=0),dict(distance_scale=0),dict(gamma=np.nan),dict(local_weight=-1),dict(relative_floor=0),dict(initial_radius=1),dict(jitter=-1),dict(learning_rate=np.inf),dict(local_bandwidth='adaptive')])
def test_config_validation(change):
    with pytest.raises(ValueError):method.GraphStressConfig(**change)


@pytest.mark.parametrize('array', [np.zeros((4,2)),np.zeros((20,0)),np.zeros(20),np.full((20,2),np.nan),np.ones((20,2),complex),np.full((20,2),'label')])
def test_feature_validation(array):
    with pytest.raises(ValueError):method.prepare_graph_stress(array)


def test_budget_checked_before_allocation_and_again_at_fit(features,setup,monkeypatch):
    def should_not_run(*args):raise AssertionError('graph allocation preceded budget')
    monkeypatch.setattr(method,'_graph_lengths',should_not_run)
    with pytest.raises(MemoryError,match='graph stress'):
        method.prepare_graph_stress(features,config=setup.config,max_dense_bytes=1)
    with pytest.raises(MemoryError):method.fit_graph_stress(setup,max_dense_bytes=1)


def test_duplicate_radii_and_coincident_gradients(torch):
    f=np.vstack([np.zeros((6,2)),np.arange(1,9)[:,None]*np.array([[1.,.3]])])
    s=method.prepare_graph_stress(f,config=method.GraphStressConfig(n_neighbors=3,local_bandwidth='local_radius'))
    assert s.metadata['zero_radius_count']==6
    assert s.metadata['radius_floor']>0
    np.testing.assert_allclose(s.teacher.sum(1),1)
    u=torch.zeros((len(f),2),dtype=torch.float64,requires_grad=True)
    loss=method._objective(u,torch.tensor(s.target),torch.tensor(s.teacher),s.config,'poincare',3,backward=True)
    assert np.isfinite(loss).all() and torch.isfinite(u.grad).all()
    with pytest.raises(ValueError):
        method.prepare_graph_stress(np.zeros((20,2)))


def test_initial_boundary_and_lazy_torch(setup,monkeypatch):
    with pytest.raises(ValueError,match='open unit disk'):
        method._tangent_from_disk(np.array([[1.,0.]]))
    original=method.importlib.import_module
    def blocked(name):
        if name=='torch':raise ImportError('blocked')
        return original(name)
    monkeypatch.setattr(method.importlib,'import_module',blocked)
    with pytest.raises(ImportError,match=r'sc-poincare\[torch\]'):
        method.fit_graph_stress(setup,epochs=0)


def load_cli():
    spec=importlib.util.spec_from_file_location('graph_cli',Path(__file__).parents[1]/'examples/run_graph_stress.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_cli_excludes_label_before_pca_and_preserves_output(tmp_path):
    pytest.importorskip('pandas')
    from sklearn.decomposition import PCA
    cli=load_cli()
    path=tmp_path/'input.csv'
    path.write_text('a,b,label\n1,5,alpha\n2,3,beta\n3,8,gamma\n4,9,delta\n')
    got=cli.load_features(path,csv_label_column=-1,pca=1)
    expected=PCA(1,svd_solver='full').fit_transform([[1,5],[2,3],[3,8],[4,9]])
    np.testing.assert_array_equal(got,expected)
    with pytest.raises(ValueError):cli.load_features(path,pca=1)
    with pytest.raises(ValueError):cli.load_features(path,csv_label_column=10)
    np.save(tmp_path/'input.npy',np.ones((4,2)))
    with pytest.raises(ValueError):cli.load_features(tmp_path/'input.npy',csv_label_column=-1)
    out=tmp_path/'existing';out.mkdir();(out/'checkpoint.pt').write_bytes(b'preserve')
    with pytest.raises(SystemExit):cli.main(['--input',str(path),'--output',str(out)])
    assert (out/'checkpoint.pt').read_bytes()==b'preserve'


@pytest.mark.parametrize('geometry',['poincare','euclidean'])
def test_gradient_matches_finite_difference(setup,torch,geometry):
    u=torch.tensor(method._tangent_from_disk(setup.base_initial),requires_grad=True)
    target,teacher=torch.tensor(setup.target),torch.tensor(setup.teacher)
    method._objective(u,target,teacher,setup.config,geometry,4,backward=True)
    analytic=u.grad[3,1].item()
    delta=1e-6
    with torch.no_grad():u[3,1]+=delta
    plus=method._objective(u,target,teacher,setup.config,geometry,4,backward=False)[0]
    with torch.no_grad():u[3,1]-=2*delta
    minus=method._objective(u,target,teacher,setup.config,geometry,4,backward=False)[0]
    assert analytic==pytest.approx((plus-minus)/(2*delta),abs=2e-8,rel=2e-7)


@pytest.mark.parametrize('kwargs',[dict(epochs=True),dict(epochs=-1),dict(chunk_size=0),dict(checkpoint_every=0),dict(random_state=-1),dict(geometry='disk'),dict(optimizer_name='lorentz')])
def test_fit_option_validation(setup,kwargs):
    with pytest.raises(ValueError):method.fit_graph_stress(setup,**kwargs)


def test_unrepresentable_disk_point_rejected(setup,torch):
    u=torch.full((len(setup.target),2),100.,dtype=torch.float64,requires_grad=True)
    with pytest.raises(FloatingPointError,match='open disk'):
        method._objective(u,torch.tensor(setup.target),torch.tensor(setup.teacher),setup.config,'poincare',4,backward=True)


def test_empty_local_teacher_is_rejected(features,monkeypatch):
    # Simulate complete Gaussian underflow after valid graph construction.
    # The global teacher stays finite; only the radius teacher loses its mass.
    graphs={name:method._graph_lengths(features,4,name) for name in ['mutual','union']}
    monkeypatch.setattr(method,'_graph_lengths',lambda f,k,name:graphs[name].copy())
    original=np.exp
    calls=[]
    def underflow(exponent):
        calls.append(1)
        return original(exponent) if len(calls)==1 else np.zeros_like(exponent)
    monkeypatch.setattr(method.np,'exp',underflow)
    with pytest.raises(FloatingPointError,match='empty or nonfinite row'):
        method.prepare_graph_stress(features,config=method.GraphStressConfig(n_neighbors=4,local_bandwidth='local_radius'))


def test_old_geometry_source_checkpoint_is_rejected_without_hash_migration(setup,torch):
    state=deepcopy(method.fit_graph_stress(setup,epochs=1).state)
    state['metadata']['source_hashes']['_disk.py']='0'*64
    state['checkpoint_sha256']=method._checkpoint_hash(state)
    with pytest.raises(ValueError,match='source.*mismatch'):
        method.fit_graph_stress(setup,epochs=2,resume=state)


def test_initializer_jitter_is_unchanged_and_not_a_deprecated_display_api(features,torch):
    config=method.GraphStressConfig(n_neighbors=4)
    assert config.jitter==.0002
    setup=method.prepare_graph_stress(features,config=config)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always',FutureWarning)
        result=method.fit_graph_stress(setup,epochs=0,random_state=2)
    assert not any(name in str(w.message) for w in recorded for name in ('poincare_jitter','poincare_fan','poincare_repel'))
    expected=setup.base_initial+np.random.default_rng(2).normal(scale=.0002,size=setup.base_initial.shape)
    np.testing.assert_array_equal(result.initial,expected)


def test_preparation_does_not_require_torch(features,monkeypatch):
    def unavailable():raise ImportError('Torch unavailable')
    monkeypatch.setattr(method,'_torch',unavailable)
    setup=method.prepare_graph_stress(features,config=method.GraphStressConfig(n_neighbors=4))
    assert setup.target.shape==(19,19)


def test_cli_atomic_checkpoint_and_actual_resume(features,tmp_path,torch):
    pytest.importorskip('threadpoolctl')
    cli=load_cli()
    path=tmp_path/'features.npy';np.save(path,features)
    shared=['--input',str(path),'--neighbors','4','--threads','1','--checkpoint-every','2']
    cli.main(shared+['--output',str(tmp_path/'first'),'--epochs','3'])
    checkpoint=tmp_path/'first/checkpoint.pt'
    state=torch.load(checkpoint,weights_only=True)
    assert state['state']['epoch']==3
    np.testing.assert_array_equal(state['current'].numpy(),np.load(tmp_path/'first/result.npz')['current'])
    assert not list((tmp_path/'first').glob('*.tmp'))
    cli.main(shared+['--output',str(tmp_path/'resumed'),'--epochs','6','--resume',str(checkpoint)])
    cli.main(shared+['--output',str(tmp_path/'full'),'--epochs','6'])
    for key in ['current','coordinates','loss_trace']:
        np.testing.assert_array_equal(np.load(tmp_path/'resumed/result.npz')[key],np.load(tmp_path/'full/result.npz')[key])


@pytest.mark.parametrize('values',[np.full((4,2),'label'),np.ones((4,2),complex)])
def test_cli_rejects_nonnumeric_npy_cleanly(tmp_path,values):
    path=tmp_path/'input.npy';np.save(path,values)
    with pytest.raises(ValueError,match='real numeric'):
        load_cli().load_features(path)


@pytest.mark.parametrize("bandwidth", ["global", "local_radius"])
def test_teacher_metadata_uniform_triangle(bandwidth):
    f = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3.) / 2]])
    setup = method.prepare_graph_stress(f, config=method.GraphStressConfig(
        n_neighbors=2, local_bandwidth=bandwidth))
    diagnostics = setup.metadata
    np.testing.assert_allclose(diagnostics['teacher_entropy_mean'], np.log(2.), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(diagnostics['teacher_entropy_quantiles'], np.log(2.), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(diagnostics['teacher_incoming_mass_quantiles'], 1., atol=1e-14, rtol=0.)
    assert diagnostics['teacher_diagnostic_quantiles'] == [0., .1, .5, .9, 1.]
    assert diagnostics['teacher_max_asymmetry'] < 1e-14
    assert diagnostics['teacher_max_row_sum_error'] < 1e-14
    assert not setup.teacher.flags.writeable


def test_teacher_metadata_asymmetric_line():
    # Closed-form Gaussian weights: center row has odds exp(1), endpoints
    # each have only the center. Unequal incoming mass is valid normalization.
    setup = method.prepare_graph_stress(np.array([[0.], [1.], [4.]]),
        config=method.GraphStressConfig(n_neighbors=1, local_bandwidth='local_radius'))
    q = 1. / (1. + np.exp(-1.))
    np.testing.assert_allclose(setup.teacher,
        [[0., 1., 0.], [q, 0., 1. - q], [0., 1., 0.]], atol=1e-14, rtol=0.)
    diagnostics = setup.metadata
    entropy = -q * np.log(q) - (1. - q) * np.log(1. - q)
    levels = diagnostics['teacher_diagnostic_quantiles']
    np.testing.assert_allclose(diagnostics['teacher_entropy_mean'], entropy / 3, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(diagnostics['teacher_entropy_quantiles'],
        np.quantile([0., entropy, 0.], levels), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(diagnostics['teacher_incoming_mass_quantiles'],
        np.quantile([q, 2., 1. - q], levels), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(diagnostics['teacher_max_asymmetry'], q, atol=1e-14, rtol=0.)
    assert diagnostics['teacher_max_row_sum_error'] < 1e-14
