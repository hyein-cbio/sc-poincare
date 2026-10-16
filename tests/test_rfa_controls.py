"""Experimental propagation/initializer knobs never silently replace defaults."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence

from sc_poincare import _fitting as fitting, _proximity as proximity, tl


def graph(n=25):
    values = np.ones(n - 1)
    return sparse.diags([values, values], [-1, 1], shape=(n, n), format='csr')


@pytest.mark.parametrize('tau', [.1, 1., 8.])
def test_forest_tau_matches_independent_dense_inverse_without_changing_graph(tau):
    weights = graph(7); original = weights.copy()
    actual = proximity.forest_accessibility(weights, forest_tau=tau, block_size=2)
    a = weights.toarray();laplacian = np.diag(a.sum(1)) - a
    expected = np.linalg.inv(np.eye(7) + tau * laplacian)
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    np.testing.assert_allclose(actual.sum(1), 1., atol=2e-15)
    np.testing.assert_array_equal(weights.data, original.data)


def test_default_tau_is_exact_and_does_not_change_kernel_or_stitching():
    x = np.random.default_rng(1).normal(size=(19, 3))
    base = proximity.estimate_proximity(x, 4, 1.)
    explicit = proximity.estimate_proximity(x, 4, 1., forest_tau=1.)
    longer = proximity.estimate_proximity(x, 4, 1., forest_tau=4.)
    np.testing.assert_array_equal(base.accessibility, explicit.accessibility)
    for name in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(base.weights, name), getattr(longer.weights, name))
    assert not np.array_equal(base.accessibility, longer.accessibility)
    assert longer.forest_tau == longer.diagnostics['forest_tau'] == 4.


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf, True, '1', [], 1j, np.complex64(1)])
def test_invalid_tau_rejected_before_graph_construction(value, monkeypatch):
    monkeypatch.setattr(proximity, 'local_affinity', lambda *a, **kw: pytest.fail('Must reject before graph'))
    with pytest.raises(ValueError, match='forest_tau'):
        proximity.estimate_proximity(np.ones((5, 2)), 2, 1., forest_tau=value)


def test_explicit_spectral_controls_use_arpack_and_record_residuals(monkeypatch):
    calls = []
    def solve(a, **kw):
        calls.append(kw)
        values, vectors = np.linalg.eigh(a.toarray())
        return values[-kw['k']:], vectors[:, -kw['k']:]
    monkeypatch.setattr(fitting, 'eigsh', solve)
    status = {};a = graph()
    out = fitting._seed_from_connectivities(a, 2, np.random.default_rng(3), status=status,
                                           spectral_tol=1e-4, spectral_maxiter=40)
    assert calls[0]['which'] == 'LA' and calls[0]['tol'] == 1e-4 and calls[0]['maxiter'] == 40
    d = status['initialization_diagnostics']
    assert d['solver'] == 'arpack_eigsh' and d['converged']
    assert d['checked_eigenpairs'] == 2 and d['max_eigen_residual'] < 1e-12
    assert np.all(np.linalg.norm(out, axis=1) < 1)


def test_default_spectral_controls_keep_dense_policy(monkeypatch):
    monkeypatch.setattr(fitting, 'eigsh', lambda *a, **kw: pytest.fail('Legacy dense policy must stay'))
    status = {}
    a = fitting._seed_from_connectivities(graph(), 2, np.random.default_rng(2), status=status)
    b = fitting._seed_from_connectivities(graph(), 2, np.random.default_rng(2), spectral_tol=0., spectral_maxiter=None)
    np.testing.assert_array_equal(a, b)
    assert status['initialization_diagnostics']['solver'] == 'dense_eigh'


def test_nonconvergence_warns_records_partial_pairs_and_is_seeded(monkeypatch):
    def fail(*a, **kw):
        raise ArpackNoConvergence('budget', np.array([1.]), np.ones((25, 1)))
    monkeypatch.setattr(fitting, 'eigsh', fail)
    outputs = []
    for _ in range(2):
        status = {}
        with pytest.warns(UserWarning, match='did not converge'):
            outputs.append(fitting._seed_from_connectivities(graph(), 2, np.random.default_rng(4),
                status=status, spectral_maxiter=1))
        assert status['initialization'] == 'random_fallback'
        assert status['initialization_reason'] == 'eigensolver_failure'
        d = status['initialization_diagnostics']
        assert not d['converged'] and d['partial_eigenpairs'] == 1
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize('tol,maxiter', [(True,None),(-1,None),(np.nan,None),('1',None),(10**400,None),(0,0),(0,True),(0,1.5)])
def test_invalid_spectral_controls_rejected(tol,maxiter):
    with pytest.raises(ValueError, match='spectral_'):
        fitting.fit_disk_embedding(np.ones((5,5)), n_epochs=1, connectivities=graph(5),
                                   spectral_tol=tol, spectral_maxiter=maxiter)


def test_controls_cannot_be_ignored_for_provided_or_random_starts():
    for kwargs in ({}, dict(connectivities=graph(5),initial_coordinates=np.zeros((5,2)))):
        with pytest.raises(ValueError, match='Spectral controls require'):
            fitting.fit_disk_embedding(np.ones((5,5)), n_epochs=1, spectral_maxiter=1, **kwargs)


def test_rfa_api_records_controls_and_isolates_callback_diagnostics(tmp_path):
    ad = pytest.importorskip('anndata')
    x = ad.AnnData(np.random.default_rng(3).normal(size=(8,3)))
    def mutate(p):
        p.initialization_diagnostics.clear()
    tl.poincare_map(x, method='rfa', n_epochs=1, n_neighbors=3, forest_tau=2.,
                    spectral_tol=1e-4, spectral_maxiter=100, progress_callback=mutate)
    meta = x.uns['poincare']
    assert meta['params']['forest_tau'] == 2. and meta['params']['forest_experimental']
    assert meta['initialization_diagnostics']['solver'] == 'arpack_eigsh'
    path = tmp_path/'controls.h5ad';x.write_h5ad(path)
    assert ad.read_h5ad(path).uns['poincare']['params']['spectral_maxiter'] == 100


@pytest.mark.parametrize('kwargs', [dict(forest_tau=1),dict(spectral_tol=0),dict(spectral_maxiter=None)])
def test_new_rfa_knobs_never_silently_enter_mu(kwargs):
    ad = pytest.importorskip('anndata')
    with pytest.raises(ValueError, match="method='rfa'"):
        tl.poincare_map(ad.AnnData(np.ones((8,3))), **kwargs)
