"""Initialization and degree-aware probes stay off the default path."""
import numpy as np
import pytest

from sc_poincare import experimental
from sc_poincare.controlled_probe import (
    classical_ball,
    classical_jitter,
    degree_aware_teacher,
    fit_from_initial,
    promotion_allowed,
    random_disk,
)


@pytest.fixture
def features():
    return np.random.default_rng(11).normal(size=(19, 4))


def test_random_disk_is_inside_and_not_the_classical_layout(features):
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    classical = classical_jitter(setup, 0)
    alternate = random_disk(len(features), config.initial_radius, 0)
    other = random_disk(len(features), config.initial_radius, 1)
    assert np.max(np.linalg.norm(alternate, axis=1)) < config.initial_radius
    assert not np.allclose(alternate, classical)
    assert not np.allclose(alternate, other)
    with pytest.raises(ValueError):
        random_disk(len(features), 1., 0)


def test_classical_start_matches_exact_fit(features):
    pytest.importorskip("torch")
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    initial = classical_jitter(setup, 0)
    probed = fit_from_initial(setup, initial, epochs=3, chunk_size=5, max_dense_bytes=1 << 20)
    exact = experimental.fit_graph_stress(setup, epochs=3, random_state=0, chunk_size=5)
    np.testing.assert_allclose(probed["loss_trace"], exact.loss_trace, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(probed["current"], exact.current, rtol=1e-8, atol=1e-8)


def test_degree_aware_teacher_is_stochastic_and_not_the_default(features):
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    teacher, meta = degree_aware_teacher(features, config)
    dense = teacher.toarray()
    np.testing.assert_allclose(dense.sum(1), 1., atol=1e-10)
    assert meta["rule"] == "sigma_sqrt_degree_product"
    assert not np.allclose(dense, setup.teacher)
    again = experimental.prepare_graph_stress(features, config=config)
    np.testing.assert_array_equal(again.teacher, setup.teacher)


def test_sgd_is_a_separate_optimizer(features):
    pytest.importorskip("torch")
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    initial = classical_jitter(setup, 0)
    adam = fit_from_initial(setup, initial, epochs=3, chunk_size=5, max_dense_bytes=1 << 20, optimizer_name="adam")
    sgd = fit_from_initial(setup, initial, epochs=3, chunk_size=5, max_dense_bytes=1 << 20, optimizer_name="sgd")
    assert adam["optimizer_name"] == "adam"
    assert sgd["loss_trace"][0] == pytest.approx(adam["loss_trace"][0])
    assert sgd["loss_trace"][-1] != pytest.approx(adam["loss_trace"][-1])


def test_multiscale_mixture_is_stochastic_and_not_tau_one(features):
    from sc_poincare._proximity import forest_accessibility, local_affinity
    from sc_poincare.controlled_probe import fixed_multiscale_accessibility
    weights, _ = local_affinity(features, 4, 1.)
    base = forest_accessibility(weights, forest_tau=1.)
    mixed = fixed_multiscale_accessibility(weights, max_dense_bytes=1 << 20)
    np.testing.assert_allclose(mixed.sum(1), 1., atol=1e-8)
    assert not np.allclose(mixed, base)
    again = forest_accessibility(weights, forest_tau=1.)
    np.testing.assert_allclose(again, base)


def test_three_dimensional_ball_is_explicit_and_finite(features):
    pytest.importorskip("torch")
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    assert setup.base_initial.shape[1] == 2
    ball = classical_ball(setup, 3, 0)
    assert ball.shape == (len(features), 3)
    assert np.max(np.linalg.norm(ball, axis=1)) < 1.
    fitted = fit_from_initial(setup, ball, epochs=1, chunk_size=5, max_dense_bytes=1 << 20)
    assert len(fitted["loss_trace"]) == 2
    assert np.isfinite(fitted["loss_trace"]).all()
    assert fitted["current"].shape == (len(features), 3)


def test_unit_curvature_matches_the_default_disk(features):
    pytest.importorskip("torch")
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    initial = classical_jitter(setup, 0)
    plain = fit_from_initial(setup, initial, epochs=2, chunk_size=5, max_dense_bytes=1 << 20)
    curved = fit_from_initial(setup, initial, epochs=2, chunk_size=5, max_dense_bytes=1 << 20, curvature=1.)
    np.testing.assert_allclose(curved["loss_trace"], plain["loss_trace"], rtol=1e-10, atol=1e-10)
    milder = fit_from_initial(setup, initial, epochs=1, chunk_size=5, max_dense_bytes=1 << 20, curvature=.25)
    assert milder["curvature"] == .25
    assert np.isfinite(milder["loss_trace"]).all()
    assert np.max(np.sum(milder["current"] ** 2, axis=1) * .25) < 1
    with pytest.raises(ValueError):
        fit_from_initial(setup, initial, epochs=1, chunk_size=5, max_dense_bytes=1 << 20, curvature=0)


def test_lorentz_chart_matches_disk_distance_and_stays_on_the_hyperboloid(features):
    pytest.importorskip("torch")
    from sc_poincare._disk import pairwise_disk_length
    from sc_poincare.controlled_probe import disk_from_lorentz, fit_lorentz, lorentz_from_disk
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    initial = classical_jitter(setup, 0)
    hyperboloid = lorentz_from_disk(initial)
    assert hyperboloid.shape == (len(features), 3)
    minkowski = -hyperboloid[:, 0] ** 2 + np.sum(hyperboloid[:, 1:] ** 2, axis=1)
    np.testing.assert_allclose(minkowski, -1., atol=1e-8)
    disk = pairwise_disk_length(initial)
    left, right = hyperboloid, hyperboloid
    inner = -left[:, :1] * right[:, 0] + left[:, 1:] @ right[:, 1:].T
    lorentz = np.arccosh(np.clip(-inner, 1., None))
    np.fill_diagonal(lorentz, 0.)
    np.testing.assert_allclose(lorentz, disk, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(disk_from_lorentz(hyperboloid), initial, rtol=1e-8, atol=1e-8)
    disk_fit = fit_from_initial(setup, initial, epochs=0, chunk_size=5, max_dense_bytes=1 << 20)
    lorentz_fit = fit_lorentz(setup, initial, epochs=2, chunk_size=5, max_dense_bytes=1 << 20)
    assert lorentz_fit["loss_trace"][0] == pytest.approx(disk_fit["loss_trace"][0], rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(lorentz_fit["minkowski_diag"], -1., atol=1e-6)


def test_intrinsic_rmsprop_matches_the_initial_loss_and_stays_in_the_disk(features):
    pytest.importorskip("torch")
    from sc_poincare.controlled_probe import fit_intrinsic_rmsprop
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = experimental.prepare_graph_stress(features, config=config)
    initial = classical_jitter(setup, 0)
    adam = fit_from_initial(setup, initial, epochs=2, chunk_size=5, max_dense_bytes=1 << 20)
    intrinsic = fit_intrinsic_rmsprop(setup, initial, epochs=2, chunk_size=5, max_dense_bytes=1 << 20)
    assert intrinsic["loss_trace"][0] == pytest.approx(adam["loss_trace"][0], rel=1e-6, abs=1e-6)
    assert intrinsic["loss_trace"][-1] != pytest.approx(adam["loss_trace"][-1])
    assert np.max(np.sum(intrinsic["current"] ** 2, axis=1)) < 1


def test_development_input_blocks_promotion():
    assert promotion_allowed(development_input=True, joint_improvements=[True, True]) is False
    assert promotion_allowed(development_input=False, joint_improvements=[True, False]) is False
    assert promotion_allowed(development_input=False, joint_improvements=[True, True]) is True


def test_full_planaria_stays_over_the_default_dense_guard():
    from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES, check_dense_budget
    with pytest.raises(MemoryError):
        check_dense_budget(21612, DEFAULT_MAX_DENSE_BYTES)


def test_lorentz_gradient_is_the_metric_dual_and_a_descent_direction(features):
    torch = pytest.importorskip('torch')
    from sc_poincare.controlled_probe import _lorentz_tangent_gradient, _minkowski, lorentz_from_disk, fit_lorentz
    point = torch.tensor(lorentz_from_disk(np.array([[.2, -.1], [.3, .2]])), dtype=torch.float64)
    gradient = torch.tensor([[2., -1., .5], [-3., .5, 1.]], dtype=torch.float64)
    original = gradient.clone()
    direction = torch.tensor([[.5, 1., -.5], [.2, -.4, 1.]], dtype=torch.float64)
    direction = direction + _minkowski(point, direction)[:, None] * point
    projected = _lorentz_tangent_gradient(point, gradient)
    torch.testing.assert_close(_minkowski(point, projected), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0.)
    torch.testing.assert_close(_minkowski(projected, direction), (gradient * direction).sum(-1))
    torch.testing.assert_close(gradient, original)

    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig(n_neighbors=4))
    fitted = fit_lorentz(setup, classical_jitter(setup, 0), epochs=1,
                         learning_rate=1e-4, max_dense_bytes=1 << 20)
    assert fitted['loss_trace'][1] < fitted['loss_trace'][0]


@pytest.mark.parametrize('coincident', [False, True])
def test_lorentz_probe_has_finite_self_and_coincident_gradients(features, coincident):
    torch = pytest.importorskip('torch')
    from sc_poincare.controlled_probe import _probe_objective, lorentz_from_disk, fit_lorentz
    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig(n_neighbors=4))
    initial = classical_jitter(setup, 0)
    initial[0] = 0.
    if coincident:
        initial[1] = initial[0]
    point = torch.tensor(lorentz_from_disk(initial), dtype=torch.float64, requires_grad=True)
    target = torch.tensor(setup.target, dtype=torch.float64)
    teacher = torch.tensor(setup.teacher, dtype=torch.float64)
    value = _probe_objective(point, target, teacher, setup.config, 4, lorentz=True, backward=True)
    assert np.isfinite(value)
    assert torch.isfinite(point.grad).all()
    fitted = fit_lorentz(setup, initial, epochs=2, chunk_size=4, max_dense_bytes=1 << 20)
    assert np.isfinite(fitted['loss_trace']).all()
    assert np.isfinite(fitted['current']).all()
    np.testing.assert_allclose(fitted['minkowski_diag'], -1., atol=1e-8)


@pytest.mark.parametrize('name', ['fit_lorentz', 'fit_intrinsic_rmsprop'])
def test_probe_backpropagates_each_chunk_and_matches_full_objective(features, name, monkeypatch):
    torch = pytest.importorskip('torch')
    from sc_poincare import controlled_probe
    fit = getattr(controlled_probe, name)
    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig(n_neighbors=4))
    initial = classical_jitter(setup, 0)
    whole = fit(setup, initial, epochs=2, chunk_size=len(features), max_dense_bytes=1 << 20)
    backward = torch.Tensor.backward
    calls = []

    def track_backward(self, *args, **kwargs):
        calls.append(self.numel())
        return backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'backward', track_backward)
    chunked = fit(setup, initial, epochs=2, chunk_size=4, max_dense_bytes=1 << 20)
    assert calls == [1] * (2 * ((len(features) + 3) // 4))
    np.testing.assert_allclose(chunked['loss_trace'], whole['loss_trace'], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(chunked['current'], whole['current'], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('invalid', ['negative', 'diagonal'])
@pytest.mark.parametrize('as_sparse', [False, True])
def test_replacement_teacher_rejects_invalid_probabilities(features, invalid, as_sparse):
    pytest.importorskip('torch')
    from scipy import sparse
    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig(n_neighbors=4))
    teacher = np.array(setup.teacher, copy=True)
    if invalid == 'diagonal':
        teacher = np.eye(len(features))
        message = 'diagonal must be zero'
    else:
        teacher[0, 2] += teacher[0, 1] + .1
        teacher[0, 1] = -.1
        message = 'must be nonnegative'
    np.testing.assert_allclose(teacher.sum(axis=1), 1.)
    if as_sparse:
        teacher = sparse.csr_matrix(teacher)
    with pytest.raises(ValueError, match=message):
        fit_from_initial(setup, classical_jitter(setup, 0), teacher=teacher, epochs=0, max_dense_bytes=1 << 20)


def test_curvature_script_evaluates_in_the_matching_unit_ball(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples/run_moignard_curvature_probe.py'
    spec = importlib.util.spec_from_file_location('curvature_probe_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = np.zeros((19, 4))
    monkeypatch.setattr(module, 'load_moignard', lambda *_: (features, np.repeat('synthetic', len(features))))
    monkeypatch.setattr(module, 'prepare_graph_stress', lambda *a, **k: object())
    monkeypatch.setattr(module, 'classical_jitter', lambda *a: np.zeros((len(features), 2)))
    monkeypatch.setattr(module, 'provenance', lambda *a, **k: {})
    unit_points = np.tile([.7, .2], (len(features), 1))
    fitted_points = {}

    def fake_fit(*args, curvature, **kwargs):
        fitted_points[curvature] = unit_points / np.sqrt(curvature)
        return dict(current=fitted_points[curvature], loss_trace=[1.] * 21)

    evaluated = []

    def check_evaluation(features, points, labels):
        np.testing.assert_allclose(points, unit_points)
        evaluated.append(points.copy())
        return dict(macro_purity=1., input_neighbor_recall=1., populations=[])

    monkeypatch.setattr(module, 'fit_from_initial', fake_fit)
    monkeypatch.setattr(module, 'metrics_for', check_evaluation)
    monkeypatch.chdir(tmp_path)
    module.main(['--output', str(tmp_path / 'result.json')])
    # c=1 is evaluated twice, c=.25 once. The invalid legacy unit-ball
    # diagnostic is skipped, while the matching-geometry evaluation succeeds.
    assert len(evaluated) == 3
    assert np.all(np.linalg.norm(fitted_points[.25], axis=1) > 1.)
    np.testing.assert_array_equal(fitted_points[1.], unit_points)
    import json
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['results'][1]['legacy_unit_curvature_evaluation'] is None
    assert result['results'][1]['legacy_evaluation_defined'] is False
