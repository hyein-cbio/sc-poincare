"""Teacher observability is read-only, not a promotion of new target rules."""
import numpy as np
import pytest

from sc_poincare import experimental as mu


def test_blocked_diagnostics_match_independent_reference_without_mutation():
    n = 131  # Cross the helper's 128-row workspace boundary.
    rng = np.random.default_rng(31)
    weights = rng.uniform(.1, 1., size=(n, n))
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0.)
    teacher = weights / weights.sum(1, keepdims=True)
    original = teacher.copy(); teacher.setflags(write=False)
    report = mu._teacher_diagnostics(teacher)
    logs = np.zeros_like(original)
    np.log(original, out=logs, where=original > 0)
    entropy = -(original * logs).sum(1)
    levels = [0., .1, .5, .9, 1.]
    assert report['teacher_entropy_mean'] == pytest.approx(entropy.mean(), abs=1e-14)
    np.testing.assert_allclose(report['teacher_entropy_quantiles'], np.quantile(entropy, levels), atol=1e-14)
    np.testing.assert_allclose(report['teacher_incoming_mass_quantiles'], np.quantile(original.sum(0), levels))
    assert report['teacher_max_asymmetry'] == pytest.approx(np.abs(original-original.T).max())
    assert report['teacher_max_row_sum_error'] == pytest.approx(np.abs(original.sum(1)-1).max())
    np.testing.assert_array_equal(teacher, original)
    # Row-normalized symmetric weights need not be symmetric, but are reversible.
    pi = weights.sum(1) / weights.sum()
    np.testing.assert_allclose(pi[:, None] * teacher, pi[None, :] * teacher.T, atol=1e-16)
    assert report['teacher_max_asymmetry'] > 0


@pytest.mark.parametrize('damage', ['negative', 'nonfinite', 'self_edge', 'unnormalized'])
def test_invalid_probability_teacher_is_not_silently_described(damage):
    teacher = np.array([[0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])
    if damage == 'negative': teacher[0] = [0., -.2, 1.2]
    elif damage == 'nonfinite': teacher[0, 1] = np.nan
    elif damage == 'self_edge': teacher[0] = [.1, .4, .5]
    else: teacher[0] *= .5
    with pytest.raises(ValueError): mu._teacher_diagnostics(teacher)


@pytest.mark.parametrize('geometry', ['poincare', 'euclidean'])
def test_local_cross_entropy_respects_teacher_entropy_lower_bound(geometry):
    torch = pytest.importorskip('torch')
    features = np.random.default_rng(2).normal(size=(19, 4))
    setup = mu.prepare_graph_stress(features, config=mu.GraphStressConfig(n_neighbors=4))
    u = torch.tensor(mu._tangent_from_disk(setup.base_initial), dtype=torch.float64, requires_grad=True)
    _, _, ce = mu._objective(u, torch.tensor(setup.target), torch.tensor(setup.teacher),
                              setup.config, geometry, 4, backward=False)
    assert ce >= setup.metadata['teacher_entropy_mean'] - 1e-12


def test_local_pair_derivative_counts_incoming_as_well_as_outgoing_mass():
    torch = pytest.importorskip('torch')
    # A reversible but asymmetric row teacher, not a label-balanced rule.
    teacher = torch.tensor([[0., 1., 0.], [.75, 0., .25], [0., 1., 0.]], dtype=torch.float64)
    pairs = torch.tensor([.2, .7, .4], dtype=torch.float64, requires_grad=True)
    index = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 2, 0]])
    distance = pairs[index]
    mask = torch.eye(3, dtype=torch.bool)
    logits = (-distance / .3).masked_fill(mask, -torch.inf)
    logq = torch.log_softmax(logits, dim=1).masked_fill(mask, 0.)
    loss = -(teacher * logq).sum() / 3
    loss.backward()
    q = logq.exp().masked_fill(mask, 0.)
    expected = [(teacher[i,j]+teacher[j,i]-q[i,j]-q[j,i])/(3*.3)
                for i,j in [(0,1),(0,2),(1,2)]]
    torch.testing.assert_close(pairs.grad, torch.stack(expected))


def test_teacher_diagnostics_roundtrip_and_callback_metadata_isolation(tmp_path):
    pytest.importorskip('torch')
    ad = pytest.importorskip('anndata')
    from sc_poincare import tl
    x = ad.AnnData(np.random.default_rng(74).normal(size=(19, 4)))
    keys = ['teacher_entropy_mean', 'teacher_entropy_quantiles', 'teacher_incoming_mass_quantiles',
            'teacher_max_asymmetry', 'teacher_max_row_sum_error', 'teacher_diagnostic_quantiles']
    observed = []
    def mutate(progress):
        metadata = progress.checkpoint['state']['metadata']
        observed.append(metadata['teacher_entropy_mean'])
        metadata['teacher_entropy_quantiles'][0] = -999.
        metadata['teacher_entropy_mean'] = -999.
    tl.poincare_map(x, n_neighbors=4, n_epochs=2, checkpoint_every=1, progress_callback=mutate)
    d = x.uns['poincare']['proximity_diagnostics']
    assert len(observed) == 3 and all(value >= 0 for value in observed)
    assert d['teacher_entropy_mean'] >= 0 and d['teacher_entropy_quantiles'][0] >= 0
    path = tmp_path/'teacher.h5ad'; x.write_h5ad(path)
    loaded = ad.read_h5ad(path).uns['poincare']['proximity_diagnostics']
    for key in keys: np.testing.assert_array_equal(loaded[key], d[key])
