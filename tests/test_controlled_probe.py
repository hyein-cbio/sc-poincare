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


def test_development_input_blocks_promotion():
    assert promotion_allowed(development_input=True, joint_improvements=[True, True]) is False
    assert promotion_allowed(development_input=False, joint_improvements=[True, False]) is False
    assert promotion_allowed(development_input=False, joint_improvements=[True, True]) is True


def test_full_planaria_stays_over_the_default_dense_guard():
    from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES, check_dense_budget
    with pytest.raises(MemoryError):
        check_dense_budget(21612, DEFAULT_MAX_DENSE_BYTES)
