"""Exercise the installed wheel and emit a cross-environment numerical record."""

import argparse
import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path
import sys

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy
import torch

import sc_poincare
from sc_poincare import experimental as mu, tl
from sc_poincare._disk import pairwise_disk_length


def fixtures():
    # Fixed, exactly representable inputs without a dependency-specific RNG or PCA.
    raw = hashlib.shake_256(b"sc-poincare wheel CI v1").digest(32 * 12 * 2)
    cells = np.frombuffer(raw, dtype="<u2").reshape(32, 12).astype(np.float64)
    features = (cells[:, :5] - 32768) / 16384
    initial_disk = (cells[:, 5:7] - 32768) / 262144
    expression = 1 + (cells % 192)
    return features, initial_disk.copy(), expression


def check_layout(adata, method, use_rep, epochs, coordinate_space="raw_training"):
    coords = adata.obsm["X_poincare"]
    meta = adata.uns["poincare"]
    assert coords.shape == (adata.n_obs, 2)
    assert np.isfinite(coords).all() and np.max(np.linalg.norm(coords, axis=1)) < 1
    assert meta["method"] == meta["params"]["method"] == method
    assert meta["params"]["use_rep"] == use_rep
    assert meta["current_epoch"] == epochs
    assert meta["coordinate_space"] == coordinate_space
    trace = np.r_[meta["initial_loss"], meta["loss"]]
    assert trace.shape == (epochs + 1,) and np.isfinite(trace).all()
    assert 0 <= meta["best_epoch"] <= epochs
    assert meta["connectivities_key"] in adata.obsp
    if method == "mu":
        assert meta["local_teacher_key"] in adata.obsp
    return trace, coords


def check_scanpy_roundtrip(expression, workdir):
    adata = anndata.AnnData(expression.copy())
    adata.obs_names = [f"cell_{i:02d}" for i in range(adata.n_obs)]
    scanpy.pp.scale(adata, max_value=10)
    scanpy.pp.pca(adata, n_comps=5, svd_solver="full")
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)
    for method in ("mu", "rfa"):
        fitted = adata.copy()
        legacy = {"burn_in_epochs": 0, "patience": 4} if method == "rfa" else {}
        tl.poincare_map(fitted, method=method, use_rep="X_pca", n_neighbors=5,
                        n_epochs=4, batch_size=7, random_state=0, **legacy)
        trace, raw_coords = check_layout(fitted, method, "X_pca", epochs=4)
        tl.poincare_pseudotime(fitted, root=0)
        coords = fitted.obsm["X_poincare"]
        np.testing.assert_allclose(pairwise_disk_length(coords), pairwise_disk_length(raw_coords),
                                   rtol=1e-12, atol=1e-12)
        pseudotime = fitted.obs["poincare_pseudotime"].to_numpy()
        assert np.isfinite(pseudotime).all() and pseudotime[0] == 0
        path = workdir / f"{method}.h5ad"
        fitted.write_h5ad(path)
        restored = anndata.read_h5ad(path)
        assert restored.obs_names.equals(adata.obs_names)
        reread_trace, reread_coords = check_layout(restored, method, "X_pca", epochs=4,
                                                  coordinate_space="root_isometry")
        np.testing.assert_array_equal(reread_coords, coords)
        np.testing.assert_array_equal(reread_trace, trace)
        np.testing.assert_array_equal(restored.obs["poincare_pseudotime"].to_numpy(), pseudotime)
        ax = scanpy.pl.embedding(restored, basis="poincare", color="poincare_pseudotime", show=False)
        assert ax is not None
        plt.close("all")


def numeric_record(features, initial_disk):
    # No initializer jitter: LAPACK may flip eigenvector signs between builds.
    # Distances, rather than signed coordinate columns, are the geometric contract.
    setup = mu.prepare_graph_stress(features, config=mu.GraphStressConfig(n_neighbors=5, jitter=0))
    target, teacher = torch.tensor(setup.target), torch.tensor(setup.teacher)
    u = torch.tensor(mu._tangent_from_disk(initial_disk), requires_grad=True)
    chunked = mu._objective(u, target, teacher, setup.config, "poincare", 7, backward=True)
    gradient = u.grad.detach().numpy().copy()
    full = mu._objective(u, target, teacher, setup.config, "poincare", len(u), backward=True)
    np.testing.assert_allclose(chunked, full, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(gradient, u.grad.detach().numpy(), rtol=1e-12, atol=1e-12)
    mu_fit = mu.fit_graph_stress(setup, epochs=6, chunk_size=7, random_state=0)

    rfa = anndata.AnnData(features.copy())
    tl.poincare_map(rfa, method="rfa", use_rep="X", n_neighbors=5, sigma="median",
                    initial_coordinates=initial_disk, n_epochs=6, batch_size=7,
                    burn_in_epochs=0, patience=6, random_state=0)
    rfa_trace, rfa_disk = check_layout(rfa, "rfa", "X", epochs=6)
    idx = np.triu_indices(len(features), k=1)
    quantities = {
        "mu_target": setup.target,
        "mu_teacher": setup.teacher,
        "mu_fixed_start_loss_terms": chunked,
        "mu_fixed_start_gradient": gradient,
        "mu_initial_distances": pairwise_disk_length(mu_fit.initial)[idx],
        "mu_best_distances": pairwise_disk_length(mu_fit.coordinates)[idx],
        "mu_loss_trace": mu_fit.loss_trace,
        "rfa_best_distances": pairwise_disk_length(rfa_disk)[idx],
        "rfa_loss_trace": rfa_trace,
    }
    assert all(np.isfinite(array).all() for array in quantities.values())
    return {name: np.asarray(array).ravel().tolist() for name, array in quantities.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", required=True, help="Expected interpreter major.minor")
    parser.add_argument("--scanpy", required=True, help="Expected installed Scanpy version")
    parser.add_argument("--torch", required=True, help="Expected installed CPU Torch version")
    args = parser.parse_args()
    installed = Path(sc_poincare.__file__).resolve()
    dist = distribution("sc-poincare")
    assert installed == Path(dist.locate_file("sc_poincare/__init__.py")).resolve()
    assert not installed.is_relative_to(args.source_root.resolve())
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == args.python
    assert version("scanpy") == args.scanpy and version("torch") == args.torch
    assert torch.version.cuda is None and torch.version.hip is None
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    versions = {name: version(name) for name in ("sc-poincare", "scanpy", "anndata", "numpy",
                                                 "scipy", "scikit-learn", "pandas", "h5py", "torch")}
    print(f"Installed wheel: {installed}; Python {sys.version.split()[0]}; {versions}")
    features, initial_disk, expression = fixtures()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    check_scanpy_roundtrip(expression, args.output.parent)
    result = dict(schema=1, python=args.python, versions=versions,
                  features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                  quantities=numeric_record(features, initial_disk))
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
