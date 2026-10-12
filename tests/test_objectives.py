"""Distribution/gradient references independent of the optimizer loop."""
import numpy as np
import pytest
from scipy.special import logsumexp

from sc_poincare import _fitting
from sc_poincare._objectives import _numpy_alignment_loss, _torch_alignment_loss


def test_default_loss_matches_direct_positive_distribution_formula():
    p = np.array([[.6, .3, .1], [.2, .5, .3]])
    d = np.array([[0., 1., 2.], [1., 0., 3.]])
    logits = -d / .5
    q = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    expected = np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)))
    assert _numpy_alignment_loss(p, d, .5) == pytest.approx(expected, rel=1e-13)
    torch = pytest.importorskip('torch')
    distance = torch.tensor(d, dtype=torch.float64, requires_grad=True)
    actual = _torch_alignment_loss(torch.tensor(p), distance, .5)
    predicted = torch.softmax(-distance / .5, dim=1)
    reference = (torch.tensor(p) * (torch.tensor(p).log() - predicted.log())
                 + predicted * (predicted.log() - torch.tensor(p).log())).sum()
    a = torch.autograd.grad(actual, distance, retain_graph=True)[0]
    b = torch.autograd.grad(reference, distance)[0]
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('tiny', [1e-60, 1e-320])
def test_positive_target_tails_including_subnormals_are_preserved(tiny):
    p = np.array([[.5, .5, tiny]])
    d = np.zeros_like(p)
    expected = np.sum(p * (np.log(p) + np.log(3.))
                      + (1 / 3) * (-np.log(3.) - np.log(p)))
    assert _numpy_alignment_loss(p, d, .5) == pytest.approx(expected, rel=1e-13)
    torch = pytest.importorskip('torch')
    distance = torch.tensor(d, dtype=torch.float64, requires_grad=True)
    value = _torch_alignment_loss(torch.tensor(p), distance, .5)
    assert float(value.detach()) == pytest.approx(expected, rel=1e-13)
    value.backward()
    assert torch.isfinite(distance.grad).all()


def test_log_softmax_keeps_underflowed_prediction_logs_and_gradients():
    p = np.array([[.5, .5]])
    d = np.array([[0., 1000.]])
    # q2 underflows, but p2 * -log(q2) remains finite and informative.
    assert _numpy_alignment_loss(p, d, 1.) == pytest.approx(500., abs=1e-10)
    torch = pytest.importorskip('torch')
    distance = torch.tensor(d, requires_grad=True)
    loss = _torch_alignment_loss(torch.tensor(p), distance, 1.)
    loss.backward()
    np.testing.assert_allclose(distance.grad.numpy(), [[-.5, .5]], atol=1e-12)


def test_zero_target_uses_tiny_logs_only_not_added_probability_mass():
    p = np.array([[1., 0.]])
    d = np.zeros_like(p)
    expected = -.5 * np.log(np.finfo(np.float64).tiny)
    assert _numpy_alignment_loss(p, d, .5) == pytest.approx(expected, rel=1e-13)
    torch = pytest.importorskip('torch')
    distance = torch.tensor(d, requires_grad=True)
    value = _torch_alignment_loss(torch.tensor(p), distance, .5)
    assert float(value.detach()) == pytest.approx(expected, rel=1e-13)
    value.backward()
    assert torch.isfinite(distance.grad).all()


def test_fit_warns_about_zero_target_regularization():
    with pytest.warns(UserWarning, match='Zero target probabilities'):
        report = _fitting.fit_disk_embedding(np.eye(3), n_epochs=1)
    assert np.isfinite(report.best_loss)


def test_overflowing_target_row_mass_is_rejected():
    with pytest.raises(ValueError, match='finite positive mass'):
        _fitting.fit_disk_embedding(np.full((3, 3), 1e308), n_epochs=1)


def test_numpy_and_torch_fit_same_stable_objective(monkeypatch):
    pytest.importorskip('torch')
    p = np.array([[.7, .3-1e-50, 1e-50], [.2, .5, .3], [.1, .2, .7]])
    xy = np.array([[0., 0.], [.07, .02], [-.03, .06]])
    kwargs = dict(initial_coordinates=xy, n_epochs=2, burn_in_epochs=0, learning_rate=.001)
    torch_report = _fitting.fit_disk_embedding(p, **kwargs)
    monkeypatch.setattr(_fitting, 'torch', None)
    numpy_report = _fitting.fit_disk_embedding(p, **kwargs)
    np.testing.assert_allclose(numpy_report.coordinates, torch_report.coordinates, atol=1e-8)
    np.testing.assert_allclose(numpy_report.loss_trace, torch_report.loss_trace, rtol=1e-8)
