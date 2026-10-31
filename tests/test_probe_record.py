"""Research reruns must preserve old evidence before reading or fitting."""
import hashlib
import json

import numpy as np
import pytest

from sc_poincare._probe_record import load_moignard, parse_probe_args, write_record


@pytest.mark.parametrize("kind", ["lorentz", "curvature"])
@pytest.mark.parametrize("artifact", ["json", "npz", "symlink", "dangling"])
def test_rerun_refuses_existing_outputs(tmp_path, kind, artifact):
    output = tmp_path / "rerun.json"
    target = output.with_suffix(".npz") if artifact == "npz" else output
    original = b"preserved research"
    if artifact in ("symlink", "dangling"):
        destination = tmp_path / "archived.json"
        if artifact == "symlink":
            destination.write_bytes(original)
        target.symlink_to(destination)
    else:
        target.write_bytes(original)
    with pytest.raises(FileExistsError):
        parse_probe_args(kind, ["--source", str(tmp_path / "missing.csv"), "--output", str(output)])
    if artifact != "dangling":
        assert target.read_bytes() == original


def test_write_record_is_exclusive_and_array_hash_is_verifiable(tmp_path):
    output = tmp_path / "new.json"
    values = np.arange(6).reshape(3, 2)
    write_record(output, {"biological_validation": False}, {"current": values})
    report = json.loads(output.read_text())
    sidecar = output.with_suffix(".npz")
    assert report["arrays"]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    with np.load(sidecar, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["current"], values)
    before = output.read_bytes(), sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        write_record(output, {}, {"current": values + 1})
    assert before == (output.read_bytes(), sidecar.read_bytes())


def test_hash_mismatch_precedes_csv_parsing(tmp_path):
    source = tmp_path / "wrong.csv"
    source.write_text("not a reproducible dataset")
    with pytest.raises(ValueError, match="CSV hash mismatch"):
        load_moignard(source)
