"""Exact-recovery and protocol gates for the landmark reference check."""
import json
from pathlib import Path

import numpy as np
import pytest

from sc_poincare import experimental
from sc_poincare._research_policy import CLOSED_RESEARCH_MARKER
from sc_poincare.landmark import (
    LandmarkColumnSetup,
    compare_landmark_reference,
    fit_landmark_columns,
    freeze_comparison,
    prepare_landmark_columns,
    reevaluate_study_tree,
    verify_reference_tree,
    write_reference_tree,
    write_study_tree,
    _parameters_json,
    _sparse_teacher,
)


@pytest.fixture
def features():
    return np.random.default_rng(421).normal(size=(19, 4))


def _protocol(features, **overrides):
    config = overrides.pop("config", experimental.GraphStressConfig(n_neighbors=4))
    values = dict(
        input_id="synthetic-reference",
        role="reference_check",
        n_landmarks=len(features),
        max_dense_bytes=1 << 20,
        max_epochs=0,
        seeds=(0, 1),
        primary_metric="landmark_column_max_abs",
        sensitivity_metrics=("triangle_excess_max", "init_procrustes_rmse", "completion_mean_relative_error"),
        labels_excluded=True,
    )
    values.update(overrides)
    return freeze_comparison(features, config, **values), config


def test_checkpoint_sources_do_not_include_the_reference_module():
    assert set(experimental._source_hashes()) == {
        "experimental.py", "_proximity.py", "_fitting.py", "_disk.py",
    }


def test_all_landmarks_recover_exact_paths_scale_and_initialization(features):
    protocol, config = _protocol(features)
    reference = compare_landmark_reference(features, protocol, config)
    metrics = reference.report["metrics"]
    assert reference.report["fits_executed"] == 0
    assert reference.report["biological_validation"] is False
    assert reference.report["primary_metric"] == "landmark_column_max_abs"
    assert metrics["landmark_column_max_abs"] == pytest.approx(0., abs=1e-8)
    assert metrics["triangle_excess_max"] == pytest.approx(0., abs=1e-8)
    assert metrics["completion_mean_relative_error"] == pytest.approx(0., abs=1e-8)
    assert metrics["init_procrustes_rmse"] == pytest.approx(0., abs=1e-6)
    assert reference.report["declared_seeds"] == [0, 1]
    assert reference.report["executed_landmark_seeds"] == [0]
    assert set(reference.report["sensitivity"]) == set(protocol.sensitivity_metrics)


def test_fewer_landmarks_stay_above_the_exact_geodesic_and_do_not_reproduce_it(features):
    protocol, config = _protocol(features, n_landmarks=5)
    reference = compare_landmark_reference(features, protocol, config)
    metrics = reference.report["metrics"]
    assert metrics["landmark_column_max_abs"] == pytest.approx(0., abs=1e-8)
    assert metrics["triangle_excess_max"] > 1e-4
    assert metrics["triangle_excess_max"] >= -1e-8
    assert metrics["init_procrustes_rmse"] > 0.
    assert reference.arrays["landmark_index"].shape == (5,)


def test_endpoint_gradients_match_when_every_cell_is_a_landmark(features):
    pytest.importorskip("torch")
    protocol, config = _protocol(
        features,
        primary_metric="column_objective_gradient_relative_l2",
        sensitivity_metrics=("completion_gradient_relative_l2", "landmark_column_max_abs"),
    )
    reference = compare_landmark_reference(features, protocol, config)
    metrics = reference.report["metrics"]
    assert metrics["column_objective_gradient_relative_l2"] == pytest.approx(0., abs=1e-8)
    assert metrics["completion_gradient_relative_l2"] == pytest.approx(0., abs=1e-8)


def test_fewer_landmarks_change_the_endpoint_gradient(features):
    pytest.importorskip("torch")
    protocol, config = _protocol(
        features,
        n_landmarks=5,
        primary_metric="column_objective_gradient_relative_l2",
        sensitivity_metrics=("completion_gradient_relative_l2",),
    )
    metrics = compare_landmark_reference(features, protocol, config).report["metrics"]
    assert metrics["column_objective_gradient_relative_l2"] > 1e-3
    assert np.isfinite(metrics["completion_gradient_relative_l2"])


@pytest.mark.parametrize("input_id", ["Paul", "moignard2015", "pancreas", "Nestorowa"])
def test_known_development_collections_cannot_be_declared_unseen(features, input_id):
    with pytest.raises(ValueError, match="development collection"):
        _protocol(features, input_id=input_id, role="unseen_biological")


def test_unseen_declaration_is_not_executed(features):
    protocol, config = _protocol(features, input_id="larry", role="unseen_biological")
    with pytest.raises(ValueError, match="unseen"):
        compare_landmark_reference(features, protocol, config)


def test_protocol_rejects_labels_fitting_disabled_guard_and_hash_mismatch(features):
    with pytest.raises(ValueError, match="labels_excluded"):
        _protocol(features, labels_excluded=False)
    with pytest.raises(ValueError, match="max_epochs"):
        _protocol(features, max_epochs=10)
    with pytest.raises(ValueError, match="dense guard"):
        _protocol(features, max_dense_bytes=None)
    with pytest.raises(ValueError, match="primary_metric"):
        _protocol(features, primary_metric="visual_spread")
    protocol, config = _protocol(features)
    shifted = features + 1.
    with pytest.raises(ValueError, match="frozen input hash"):
        compare_landmark_reference(shifted, protocol, config)
    with pytest.raises(MemoryError):
        tiny, _ = _protocol(features, max_dense_bytes=128)
        compare_landmark_reference(features, tiny, config)


def test_manifest_round_trip_detects_tampering_and_closed_trees(features, tmp_path):
    protocol, config = _protocol(features)
    reference = compare_landmark_reference(features, protocol, config)
    directory = write_reference_tree(tmp_path / "reference", reference)
    loaded = verify_reference_tree(directory)
    assert loaded["primary_value"] == reference.report["primary_value"]
    report = directory / "report.json"
    report.write_bytes(report.read_bytes().replace(b"synthetic-reference", b"synthetic-referencX"))
    with pytest.raises(ValueError, match="checksum"):
        verify_reference_tree(directory)
    closed = tmp_path / "closed"
    closed.mkdir()
    (closed / CLOSED_RESEARCH_MARKER).write_text(json.dumps({"status": "closed"}))
    with pytest.raises(ValueError, match="Closed research"):
        write_reference_tree(closed / "child", reference)
    assert not (closed / "child").exists()


def test_landmark_column_fit_stores_no_dense_target_and_keeps_the_exact_guard(features):
    config = experimental.GraphStressConfig(n_neighbors=4)
    setup = prepare_landmark_columns(
        features, config, n_landmarks=5, seed=0, max_dense_bytes=1 << 20,
    )
    assert setup.distances.shape == (5, len(features))
    assert setup.metadata["stores_dense_target"] is False
    assert setup.metadata["exact_guard_allows"] is True
    assert setup.teacher.shape == (len(features), len(features))
    assert setup.teacher.nnz < len(features) * len(features)
    partial_budget = setup.metadata["workspace_bytes"] + 8
    narrow = prepare_landmark_columns(
        features, config, n_landmarks=5, seed=0, max_dense_bytes=partial_budget,
    )
    assert narrow.metadata["exact_guard_allows"] is False
    with pytest.raises(MemoryError, match="was not disabled"):
        prepare_landmark_columns(
            features, config, n_landmarks=5, seed=0, max_dense_bytes=setup.metadata["workspace_bytes"] - 1,
        )
    with pytest.raises(ValueError, match="dense guard"):
        prepare_landmark_columns(features, config, n_landmarks=5, seed=0, max_dense_bytes=None)


def test_sparse_teacher_matches_exact_and_full_landmark_fit_matches_exact_mu(features):
    pytest.importorskip("torch")
    config = experimental.GraphStressConfig(n_neighbors=4)
    exact = experimental.prepare_graph_stress(features, config=config)
    teacher, _ = _sparse_teacher(np.ascontiguousarray(features, dtype=np.float64), config)
    np.testing.assert_allclose(teacher.toarray(), exact.teacher, rtol=0., atol=1e-12)
    n = len(features)
    geodesic = np.array(exact.target, dtype=np.float64, copy=True)
    geodesic *= exact.metadata["global_mean"] / config.distance_scale
    column = LandmarkColumnSetup(
        distances=np.ascontiguousarray(geodesic),
        landmark_index=np.arange(n),
        teacher=teacher,
        base_initial=np.array(exact.base_initial, dtype=np.float64, copy=True),
        config=config,
        metadata=dict(
            n_cells=n,
            n_landmarks=n,
            global_mean=exact.metadata["global_mean"],
            stores_dense_target=False,
            workspace_bytes=4 * 8 * n * n,
            parameters_json=_parameters_json(config),
        ),
    )
    fitted = fit_landmark_columns(
        column, epochs=3, random_state=0, chunk_size=5, max_dense_bytes=1 << 20,
    )
    reference = experimental.fit_graph_stress(exact, epochs=3, random_state=0, chunk_size=5)
    np.testing.assert_allclose(fitted.loss_trace, reference.loss_trace, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fitted.current, reference.current, rtol=1e-8, atol=1e-8)
    assert fitted.metadata["fits_executed"] == 1
    assert fitted.metadata["biological_validation"] is False
    assert fitted.metadata["exact_guard_disabled"] is False
    assert fitted.metadata["evaluation_loss"] == pytest.approx(fitted.current_loss, rel=1e-12, abs=1e-12)
    assert fitted.metadata["checkpoint_bytes"] > 0
    for key in ("fitting_rss_bytes", "evaluation_rss_bytes", "checkpoint_rss_bytes"):
        assert fitted.metadata[key] > 0
    dense = LandmarkColumnSetup(
        distances=column.distances,
        landmark_index=column.landmark_index,
        teacher=np.array(exact.teacher, copy=True),
        base_initial=column.base_initial,
        config=config,
        metadata=column.metadata,
    )
    with pytest.raises(ValueError, match="sparse teacher"):
        fit_landmark_columns(dense, epochs=1, max_dense_bytes=1 << 20)


def test_study_tree_restore_recomputes_the_primary_metric(features, tmp_path):
    protocol, config = _protocol(features, n_landmarks=6)
    reference = compare_landmark_reference(features, protocol, config)
    directory = write_study_tree(tmp_path / "study", features, reference)
    restored = reevaluate_study_tree(directory)
    assert restored["fits_executed"] == 0
    assert restored["biological_validation"] is False
    assert restored["primary_abs_difference"] == pytest.approx(0., abs=1e-12)
    assert restored["feature_sha256"] == protocol.feature_sha256
    features_path = directory / "features.npy"
    features_path.write_bytes(features_path.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ValueError, match="checksum"):
        reevaluate_study_tree(directory)


def test_reference_tree_parent_is_created_only_as_the_new_directory(features, tmp_path):
    protocol, config = _protocol(features, n_landmarks=5)
    reference = compare_landmark_reference(features, protocol, config)
    destination = tmp_path / "out"
    write_reference_tree(destination, reference)
    with pytest.raises(FileExistsError):
        write_reference_tree(destination, reference)
    assert json.loads((destination / "MANIFEST.json").read_text())["schema_version"] == 1
    assert Path(destination / "report.json").is_file()
