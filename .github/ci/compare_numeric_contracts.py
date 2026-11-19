"""Compare fixed wheel records to the Python 3.12 / Scanpy 1.12 baseline."""

import argparse
import json
import math
from pathlib import Path

BASELINE = "py312-scanpy112"
LANES = {
    BASELINE: ("3.12", "1.12.4"),
    "py314-scanpy112": ("3.14", "1.12.4"),
    "py314-scanpy113": ("3.14", "1.13.0"),
    "py310-scanpy111": ("3.10", "1.11.5"),
}
LENGTHS = {
    "mu_target": 1024,
    "mu_teacher": 1024,
    "mu_fixed_start_loss_terms": 3,
    "mu_fixed_start_gradient": 64,
    "mu_initial_distances": 496,
    "mu_best_distances": 496,
    "mu_loss_trace": 7,
    "rfa_best_distances": 496,
    "rfa_loss_trace": 7,
}
# Float64 geometry/optimizer checks, not byte equality across LAPACK/BLAS wheels.
RTOL, ATOL = 1e-7, 2e-9


def load(root, name):
    path = root / f"numeric-{name}" / "numeric-contract.json"
    record = json.loads(path.read_text())
    expected_python, expected_scanpy = LANES[name]
    if (record["schema"] != 1 or record["python"] != expected_python
            or record["versions"]["scanpy"] != expected_scanpy):
        raise ValueError(f"{name}: wrong schema or environment")
    arrays = record["quantities"]
    if arrays.keys() != LENGTHS.keys():
        raise ValueError(f"{name}: numerical quantities do not match the contract")
    for key, array in arrays.items():
        if len(array) != LENGTHS[key] or not all(math.isfinite(value) for value in array):
            raise ValueError(f"{name}: {key} has the wrong size or nonfinite values")
    return record


def compare(root):
    reference = load(root, BASELINE)
    for name in LANES:
        record = load(root, name)
        if record["features_sha256"] != reference["features_sha256"]:
            raise ValueError(f"{name}: fixed feature input differs from {BASELINE}")
        # Keep the modern pairs controlled; the legacy interpreter needs older dependencies.
        pins = reference["versions"]
        packages = ("sc-poincare", "torch") if name == "py310-scanpy111" else pins
        for package in packages:
            if package != "scanpy" and record["versions"][package] != pins[package]:
                raise ValueError(f"{name}: {package} differs from the pinned comparison environment")
        differences = {}
        for key, array in record["quantities"].items():
            expected = reference["quantities"][key]
            pairs = list(zip(array, expected))
            differences[key] = max(abs(a - b) for a, b in pairs)
            for i, (a, b) in enumerate(pairs):
                if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
                    raise ValueError(f"{name}: {key}[{i}]={a:.17g}, baseline={b:.17g}, "
                                     f"max absolute difference={differences[key]:.3g}")
        print(f"{name}: max absolute differences {differences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing the four lane artifacts")
    compare(parser.parse_args().root)
