import numpy as np
import pytest
from scipy.special import logsumexp

from sc_poincare import _fitting
from sc_poincare._objectives import CONDITIONAL_OBJECTIVE as CONDITIONAL
from sc_poincare._objectives import _numpy_alignment_loss, _torch_alignment_loss


def _target(n=7):
    p=np.random.default_rng(8).uniform(.1,1.,size=(n,n))
    return p/p.sum(1,keepdims=True)


def test_conditional_matches_explicit_nonself_distribution_and_zero_diagonal_gradient():
    torch=pytest.importorskip('torch')
    p=_target(4);d=np.random.default_rng(3).uniform(0,2.,size=(4,4))
    expected=0.
    for i in range(4):
        columns=np.arange(4)!=i
        pp=p[i,columns]/p[i,columns].sum()
        logits=-d[i,columns]/.5;logq=logits-logsumexp(logits);q=np.exp(logq)
        expected+=np.sum(pp*(np.log(pp)-logq)+q*(logq-np.log(pp)))
    actual=_numpy_alignment_loss(p,d,.5,objective=CONDITIONAL)
    assert actual==pytest.approx(expected,rel=1e-12)
    distance=torch.tensor(d,requires_grad=True)
    value=_torch_alignment_loss(torch.tensor(p),distance,.5,objective=CONDITIONAL)
    assert float(value.detach())==pytest.approx(expected,rel=1e-12)
    value.backward();assert torch.isfinite(distance.grad).all()
    np.testing.assert_array_equal(distance.grad.diagonal().numpy(),0.)
    other=p.copy();np.fill_diagonal(other,1e50)
    changed=d.copy();np.fill_diagonal(changed,1e100)
    assert _numpy_alignment_loss(other,changed,.5,objective=CONDITIONAL)==pytest.approx(expected,rel=1e-12)


@pytest.mark.parametrize('chunk',[1,3,20])
def test_conditional_chunks_accumulate_all_column_gradients(chunk):
    torch=pytest.importorskip('torch')
    xy=torch.tensor(np.random.default_rng(10).normal(scale=.1,size=(7,2)),requires_grad=True)
    p=torch.tensor(_target())
    full=_torch_alignment_loss(p,_fitting._torch_disk_length(xy),.5,objective=CONDITIONAL)
    full.backward();expected=xy.grad.clone();xy.grad=None;total=0.
    for start in range(0,7,chunk):
        stop=min(start+chunk,7);cols=torch.arange(start,stop)
        dist=_fitting._torch_disk_length(xy[start:stop],xy,self_columns=cols)
        loss=_torch_alignment_loss(p[start:stop],dist,.5,objective=CONDITIONAL,self_columns=cols)
        total+=float(loss.detach());loss.backward()
    assert total==pytest.approx(float(full.detach()),rel=1e-12)
    np.testing.assert_allclose(xy.grad.numpy(),expected.numpy(),atol=1e-10)
    assert _fitting.evaluate_alignment(xy.detach().numpy(),p.numpy(),.5,
                                      objective=CONDITIONAL,block_size=chunk)==pytest.approx(total,rel=1e-12)


def test_conditional_numeric_gradient_matches_autograd():
    torch=pytest.importorskip('torch')
    points=np.random.default_rng(10).normal(scale=.1,size=(7,2));p=_target()
    xy=torch.tensor(points,requires_grad=True)
    _torch_alignment_loss(torch.tensor(p),_fitting._torch_disk_length(xy),.5,
                          objective=CONDITIONAL).backward()
    grad=_fitting._numeric_row_grad(points,p,.5,np.arange(7),eps=1e-6,objective=CONDITIONAL)
    np.testing.assert_allclose(grad,xy.grad.numpy(),atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('numpy_backend',[False,True])
def test_conditional_fit_trajectory_preserved_across_chunks(monkeypatch,numpy_backend):
    if numpy_backend:monkeypatch.setattr(_fitting,'torch',None)
    else:pytest.importorskip('torch')
    reports=[_fitting.fit_disk_embedding(_target(4),n_epochs=2,batch_size=chunk,
              burn_in_epochs=0,objective=CONDITIONAL) for chunk in [1,4]]
    np.testing.assert_allclose(reports[0].coordinates,reports[1].coordinates,atol=1e-10)
    assert reports[0].objective==CONDITIONAL
    assert reports[0].best_loss==pytest.approx(_fitting.evaluate_alignment(
        reports[0].coordinates,_target(4),2.,objective=CONDITIONAL),rel=1e-10)


def test_zero_offdiagonal_mass_rejected_and_tiny_mass_not_cancelled():
    with pytest.raises(ValueError,match='off-diagonal mass'):
        _fitting.fit_disk_embedding(np.eye(3),n_epochs=1,objective=CONDITIONAL)
    p=np.full((3,3),1e-100);np.fill_diagonal(p,1.)
    report=_fitting.fit_disk_embedding(p,n_epochs=1,objective=CONDITIONAL)
    assert np.isfinite(report.best_loss)


def test_conditional_does_not_warn_about_ignored_self_zeros():
    import warnings
    p=np.ones((3,3))-np.eye(3)
    with warnings.catch_warnings(record=True) as caught:
        report=_fitting.fit_disk_embedding(p,n_epochs=1,objective=CONDITIONAL)
    assert not caught and np.isfinite(report.best_loss)


@pytest.mark.parametrize('bad',['offdiag','auto',None,True,[]])
def test_invalid_objective_rejected(bad):
    with pytest.raises(ValueError,match='objective'):
        _fitting.fit_disk_embedding(_target(),objective=bad,n_epochs=1)


def test_default_self_inclusive_objective_is_unchanged():
    a=_fitting.fit_disk_embedding(_target(),n_epochs=2)
    b=_fitting.fit_disk_embedding(_target(),n_epochs=2,objective='symmetric_kl')
    np.testing.assert_array_equal(a.coordinates,b.coordinates)
    assert a.objective=='symmetric_kl'


@pytest.mark.parametrize('objective',['symmetric_kl',CONDITIONAL])
def test_scanpy_records_method_and_keeps_proximity_diagnostics(tmp_path,objective):
    ad=pytest.importorskip('anndata')
    from sc_poincare.tl import poincare_map
    data=ad.AnnData(np.random.default_rng(2).normal(size=(10,3)))
    poincare_map(data,n_neighbors=4,n_epochs=2,objective=objective)
    saved=data.uns['poincare'];params=saved['params']
    assert params['objective']==objective
    assert params['objective_experimental']==(objective==CONDITIONAL)
    assert params['probability_computation']=='log_softmax'
    assert params['zero_target_policy']=='machine_tiny_logs_only'
    assert 'self_mass_median' in saved['proximity_diagnostics']
    file=tmp_path/'method.h5ad';data.write_h5ad(file)
    assert ad.read_h5ad(file).uns['poincare']['params']['objective']==objective
