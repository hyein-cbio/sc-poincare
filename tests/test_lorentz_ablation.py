"""Distance/gradient isolation, historical fidelity, and singularity controls."""
import ast
import hashlib
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sc_poincare import controlled_probe as cp, experimental, _lorentz_ablation as ab
from sc_poincare._probe_record import ROOT, parse_probe_args


@pytest.fixture
def case():
    features = np.random.default_rng(11).normal(size=(19, 4))
    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig(n_neighbors=4))
    initial = cp.classical_jitter(setup, 0)
    return setup, initial


def inputs(case):
    setup, initial = case
    return (torch.tensor(cp.lorentz_from_disk(initial), requires_grad=True),
            torch.tensor(np.array(setup.target)), torch.tensor(np.array(setup.teacher)))


def frozen():
    path = ROOT / "examples/fixtures/pre_pr3_lorentz.py"
    spec = importlib.util.spec_from_file_location("frozen_lorentz_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_function_text_matches_pinned_git_source_digest():
    text = (ROOT / "examples/fixtures/pre_pr3_lorentz.py").read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "fit_lorentz")
    function = "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
    assert hashlib.sha256(function.encode()).hexdigest() == ab.HISTORICAL_FUNCTION_SHA256


@pytest.mark.parametrize("gradient", ab.GRADIENTS)
def test_stable_arm_matches_existing_fitter(case, gradient, monkeypatch):
    setup, initial = case
    if gradient == "legacy":
        from functools import partial
        monkeypatch.setattr(cp, "_lorentz_tangent_gradient", partial(ab.tangent_gradient, mode="legacy"))
    before = initial.copy(), setup.target.copy(), setup.teacher.copy()
    expected = cp.fit_lorentz(setup, initial, epochs=2, chunk_size=5, max_dense_bytes=1 << 20)
    result = ab.fit_control(setup, initial, distance_mode="disk_stable", gradient_mode=gradient,
                            epochs=2, chunk_size=5, max_dense_bytes=1 << 20)
    assert result["status"] == "completed"
    np.testing.assert_array_equal(result["current"], expected["current"])
    np.testing.assert_array_equal(result["loss_trace"], expected["loss_trace"])
    for actual, original in zip((initial, setup.target, setup.teacher), before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("distance", ab.DISTANCES)
def test_chunked_and_full_backward_agree(case, distance):
    setup, _ = case
    point, target, teacher = inputs(case)
    chunked = ab.objective(point, target, teacher, setup.config, 5, distance_mode=distance, backward=True)
    gradient = point.grad.clone()
    full = ab.objective(point, target, teacher, setup.config, 5,
                        distance_mode=distance, backward=True, backward_mode="full")
    assert full == pytest.approx(chunked, rel=0, abs=1e-14)
    torch.testing.assert_close(point.grad, gradient, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_exact_acosh_and_old_gradient_match_verbatim_historical_fitter(case):
    setup, initial = case
    old = frozen().fit_lorentz(setup, initial, epochs=2, chunk_size=5, max_dense_bytes=1 << 20)
    result = ab.fit_control(setup, initial, distance_mode="acosh_exact", gradient_mode="legacy",
                            epochs=2, chunk_size=5, max_dense_bytes=1 << 20, backward_mode="full")
    if result["status"] == "failed_numerical":
        # Some supported Torch runtimes propagate the historical singularity.
        assert not np.isfinite(old["current"]).all()
        assert result["failure"]["stage"] == "backward"
        assert result["epochs_completed"] < 2
    else:
        np.testing.assert_allclose(result["loss_trace"], old["loss_trace"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(result["current"], old["current"], atol=1e-14, rtol=0)


def test_corrected_gradients_agree_across_distance_charts_but_legacy_ones_differ(case):
    setup, _ = case
    point, target, teacher = inputs(case)
    grads, values = {}, {}
    for mode in ("disk_stable", "acosh_masked"):
        values[mode] = ab.objective(point, target, teacher, setup.config, 5,
                                    distance_mode=mode, backward=True)
        grads[mode] = point.grad.clone()
    assert values["disk_stable"] == pytest.approx(values["acosh_masked"], rel=0, abs=1e-12)
    torch.testing.assert_close(ab.tangent_gradient(point.detach(), grads["disk_stable"], "corrected"),
                              ab.tangent_gradient(point.detach(), grads["acosh_masked"], "corrected"),
                              rtol=1e-9, atol=1e-10)
    assert not torch.allclose(ab.tangent_gradient(point.detach(), grads["disk_stable"], "legacy"),
                              ab.tangent_gradient(point.detach(), grads["acosh_masked"], "legacy"))


@pytest.mark.parametrize("mode", ("disk_stable", "acosh_masked"))
def test_stabilized_coincident_distances_and_gradients_are_finite(mode):
    initial = np.array([[0., 0.], [0., 0.], [.2, -.1]])
    point = torch.tensor(cp.lorentz_from_disk(initial), requires_grad=True)
    distance = ab.distance_rows(point, 0, 3, mode)
    distance.sum().backward()
    assert torch.isfinite(point.grad).all()
    assert float(distance[0, 1].detach()) == 0.
    assert torch.equal(distance.diag(), torch.zeros(3, dtype=torch.float64))


def test_exact_acosh_preserves_literal_historical_values_and_backward(case):
    _, initial = case
    initial[:2] = 0.
    point = torch.tensor(cp.lorentz_from_disk(initial), requires_grad=True)
    actual = ab.distance_rows(point, 0, len(point), "acosh_exact")
    actual.sum().backward()
    gradient = point.grad.clone()
    point.grad = None
    inner = -point[:, :1] * point[:, 0] + point[:, 1:] @ point[:, 1:].T
    expected = torch.acosh(torch.clamp(-inner, min=1.)).clone()
    rows = torch.arange(len(point))
    expected[rows, rows] = 0
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The treatment of acosh'(1) is runtime-dependent. Do not require a NaN
    # when newer Torch releases return a finite boundary derivative.
    torch.testing.assert_close(gradient, point.grad, rtol=0, atol=0, equal_nan=True)


def test_failed_backward_is_labeled_and_never_repaired(case, monkeypatch):
    setup, initial = case
    original = ab.objective
    def nonfinite(point, *args, **kwargs):
        value = original(point, *args, **kwargs)
        if kwargs.get("backward"):
            point.grad.fill_(float("nan"))
        return value
    monkeypatch.setattr(ab, "objective", nonfinite)
    result = ab.fit_control(setup, initial, distance_mode="acosh_exact", gradient_mode="corrected",
                            epochs=20, chunk_size=5, max_dense_bytes=1 << 20)
    assert result["status"] == "failed_numerical"
    assert result["epochs_completed"] == 0
    assert len(result["loss_trace"]) == 1
    assert result["failure"]["nonfinite_gradient_entries"] > 0
    np.testing.assert_array_equal(result["current"], cp.lorentz_from_disk(initial))


def test_factorial_flag_is_opt_in_and_uses_fresh_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old = parse_probe_args("lorentz", [])
    new = parse_probe_args("lorentz", ["--distance-ablation"])
    assert not old.distance_ablation
    assert new.output.name == "moignard-lorentz-factorial-2026-11-03.json"
    assert new.output != old.output
    with pytest.raises(SystemExit):
        parse_probe_args("lorentz", ["--historical-replay"])


@pytest.mark.parametrize("artifact", ("json", "npz", "dangling"))
def test_factorial_refuses_existing_output_before_loading(tmp_path, artifact):
    out = tmp_path / "new.json"
    if artifact == "dangling":
        out.symlink_to(tmp_path / "missing")
    else:
        (out if artifact == "json" else out.with_suffix(".npz")).write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        parse_probe_args("lorentz", ["--distance-ablation", "--output", str(out)])


@pytest.fixture(scope="module")
def saved_factorial(tmp_path_factory):
    """Produce a complete tiny record through the public CLI entrypoint."""
    from unittest.mock import patch
    path = ROOT / "examples/run_moignard_lorentz_probe.py"
    spec = importlib.util.spec_from_file_location("factorial_driver_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    features = np.random.default_rng(11).normal(size=(19, 4))
    labels = np.array(["A"] * 9 + ["B"] * 10)
    setup = experimental.prepare_graph_stress(features, config=experimental.GraphStressConfig())
    output = tmp_path_factory.mktemp("factorial") / "tiny.json"
    with patch.object(driver, "load_moignard", return_value=(features, labels)):
        driver.main(["--distance-ablation", "--historical-replay", "--output", str(output)])
    return output, features, labels, setup


def verifier():
    path = ROOT / "examples/verify_moignard_lorentz_factorial.py"
    spec = importlib.util.spec_from_file_location("factorial_verifier_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factorial_saved_record_rereads_without_fitting(saved_factorial, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Verifier must not optimize")
    monkeypatch.setattr(ab, "fit_control", forbidden)
    result = verifier().verify(*saved_factorial)
    assert len(result["checks"]) == 6
    assert all(check["population_metrics_exact"] for check in result["checks"])
    assert result["checker_provenance"]["matches_recorded"] is True


@pytest.fixture
def factorial_record_copy(saved_factorial, tmp_path):
    import json
    import shutil

    original, *case = saved_factorial
    record = tmp_path / original.name
    report = json.loads(original.read_text())
    shutil.copyfile(original.with_suffix(".npz"), tmp_path / report["arrays"]["file"])
    return record, report, case


def test_bundled_factorial_keeps_fitting_hashes_and_reports_changed_checker():
    import json

    report = json.loads((ROOT / "docs/moignard-lorentz-factorial-2026-11-03.json").read_text())
    module = verifier()
    checker = module.check_source_hashes(report["provenance"]["source_sha256"])
    assert checker["path"] == module.CHECKER_PATH
    assert checker["recorded_sha256"] == report["provenance"]["source_sha256"][module.CHECKER_PATH]
    assert checker["current_sha256"] == hashlib.sha256((ROOT / module.CHECKER_PATH).read_bytes()).hexdigest()
    assert checker["matches_recorded"] is False


@pytest.mark.parametrize("relative", ["src/sc_poincare/controlled_probe.py",
                                      "examples/run_moignard_lorentz_probe.py",
                                      "examples/fixtures/pre_pr3_lorentz.py"])
def test_bundled_factorial_still_requires_other_source_hashes(relative):
    import json

    report = json.loads((ROOT / "docs/moignard-lorentz-factorial-2026-11-03.json").read_text())
    report["provenance"]["source_sha256"][relative] = "0" * 64
    with pytest.raises(ValueError, match="Source SHA256 mismatch"):
        verifier().check_source_hashes(report["provenance"]["source_sha256"])


@pytest.mark.parametrize("fault", [None, "nonfinite_objective", "biological_validation", "promoted",
                                  "fitting_source"])
def test_factorial_record_from_older_checker_still_checks_measurements(factorial_record_copy, fault):
    import json

    record, report, case = factorial_record_copy
    module = verifier()
    bundled = json.loads((ROOT / "docs/moignard-lorentz-factorial-2026-11-03.json").read_text())
    checker_path = "examples/verify_moignard_lorentz_factorial.py"
    old_hash = bundled["provenance"]["source_sha256"][checker_path]
    report["provenance"]["source_sha256"][checker_path] = old_hash
    if fault == "nonfinite_objective":
        report["results"][-1]["final_objectives"]["acosh_exact"] = float("nan")
        message = "Nonfinite saved objective"
    elif fault in ("biological_validation", "promoted"):
        report["results"][-1][fault] = True
        message = "Invalid development-data claims"
    elif fault == "fitting_source":
        report["provenance"]["source_sha256"]["src/sc_poincare/controlled_probe.py"] = "0" * 64
        message = "Source SHA256 mismatch"
    record.write_text(json.dumps(report, allow_nan=True))
    if fault is not None:
        with pytest.raises(ValueError, match=message):
            module.verify(record, *case)
    else:
        result = module.verify(record, *case)
        assert len(result["checks"]) == 6
        assert all(check["population_metrics_exact"] for check in result["checks"])
        assert result["checker_provenance"]["recorded_sha256"] == old_hash
        assert result["checker_provenance"]["matches_recorded"] is False


@pytest.mark.parametrize("mode", ab.DISTANCES)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "positive_inf", "negative_inf"])
def test_factorial_verifier_rejects_nonfinite_saved_objectives(factorial_record_copy, mode, invalid):
    import json

    record, report, case = factorial_record_copy
    report["results"][-1]["final_objectives"][mode] = invalid
    # Python's JSON reader accepts these nonstandard constants in damaged input.
    record.write_text(json.dumps(report, allow_nan=True))
    with pytest.raises(ValueError, match="Nonfinite saved objective"):
        verifier().verify(record, *case)


@pytest.mark.parametrize("field", ["biological_validation", "promoted"])
@pytest.mark.parametrize("row_index", range(len(ab.DISTANCES) * len(ab.GRADIENTS)))
def test_factorial_verifier_rejects_unsupported_result_claims(factorial_record_copy, field, row_index):
    import json

    record, report, case = factorial_record_copy
    report["results"][row_index][field] = True
    record.write_text(json.dumps(report, allow_nan=False))
    with pytest.raises(ValueError, match="Invalid development-data claims"):
        verifier().verify(record, *case)


@pytest.mark.parametrize("fault", ("objective", "target", "trace", "status", "population",
                                   "source_hash", "condition", "npz_hash", "initial", "historical"))
def test_factorial_verifier_rejects_corrupted_records(saved_factorial, tmp_path, fault):
    import json
    import shutil
    original, *case = saved_factorial
    record = tmp_path / original.name
    report = json.loads(original.read_text())
    row = report["results"][0]
    sidecar = tmp_path / report["arrays"]["file"]
    shutil.copyfile(original.with_suffix(".npz"), sidecar)
    if fault == "objective":
        row["final_objectives"]["acosh_exact"] += .1
    elif fault == "target":
        report["shared_sha256"]["target"] = "0" * 64
    elif fault == "trace":
        row["stability"]["loss_trace"][1] += .1
    elif fault == "status":
        row["epochs_completed"] = 19
    elif fault == "population":
        row["populations"][0]["n_cells"] += 1
    elif fault == "source_hash":
        report["provenance"]["source_sha256"]["src/sc_poincare/controlled_probe.py"] = "0" * 64
    elif fault == "condition":
        report["results"].pop()
    elif fault == "npz_hash":
        report["arrays"]["sha256"] = "0" * 64
    elif fault == "historical":
        report["historical_preflight"]["nonfinite_coordinate_entries"] += 1
    elif fault == "initial":
        with np.load(sidecar, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["initial"][0, 0] += .1
        np.savez_compressed(sidecar, **arrays)
        report["arrays"]["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    record.write_text(json.dumps(report, allow_nan=False))
    with pytest.raises(ValueError):
        verifier().verify(record, *case)
