# CI compatibility matrix

The [Tests workflow](../.github/workflows/tests.yml) targets Ubuntu 24.04 on
x86-64, with CPU PyTorch `2.14.1+cpu`. Each lane gets a clean installation with
its own entry point under [`.github/ci/`](../.github/ci/).

| Lane | Python | Scanpy | What it checks |
|---|---|---|---|
| `py312-scanpy112` | 3.12 | 1.12.4 | Reference environment for the numerical comparison |
| `py314-scanpy112` | 3.14 | 1.12.4 | Change Python while keeping the dependency pins identical |
| `py314-scanpy113` | 3.14 | 1.13.0 | Change Scanpy and its additional dependencies, holding the shared pins fixed |
| `py310-scanpy111` | 3.10 | 1.11.5 | Keep coverage of the declared minimum Python; use a separate compatible stack |

The workflow also tests the pinned Python 3.12 core without Torch or Scanpy.
The [first Linux run on `5d27794`](https://github.com/hyein-cbio/sc-poincare/actions/runs/44549412270)
passed all six jobs. The MU and RFA loss traces matched exactly across lanes
on the fixed fixture; the largest absolute difference across the numerical
quantities was `3.33e-16`. This is a recorded run, not a guarantee for later
revisions or other datasets.

The matrix is not the full Cartesian product: Scanpy 1.12.4 requires Python
3.12+ and Scanpy 1.13.0 requires Python 3.13+. It is also not a support guarantee
for every version permitted by `pyproject.toml`. Python 3.11/3.13, macOS, Windows,
GPU backends, and the minimum permitted dependency versions are outside this
matrix. CI pins do not change users' package requirements.

## What equivalence means here

- Every full lane builds and installs the wheel under its pins, runs `pip check`,
  then runs the integration check with `python -I` outside the checkout.
  Both MU and RFA run through Scanpy scaling and PCA, fitting, pseudotime,
  H5AD save/reload, and Scanpy plotting. The test checks cell order, loss and
  coordinate round trips, and the root isometry. These stages must run, not
  silently skip for lack of an optional dependency.
- A separate small float64 fixture holds the feature bytes fixed across lanes
  and bypasses Scanpy preprocessing. It compares MU targets and teacher rows,
  objective and gradient at a fixed start, and MU/RFA loss traces and pairwise
  Poincaré distances. The aggregate job compares with the Python 3.12
  baseline (`rtol=1e-7`, `atol=2e-9`). Embedding axes may have arbitrary signs,
  so this is not a byte-level coordinate or rendered-pixel comparison.
- Existing source tests still cover analytic checks, all-column row chunking,
  and same-environment checkpoint recovery. Cross-environment checkpoint
  portability and atlas-scale performance are not tested by this workflow.

## Updating the pins

Keep `py312-scanpy112` and `py314-scanpy112` on the same pins, and update the
modern shared constraints together. The 1.13 lane adds only requirements that
are not installed by 1.12. Review any unexpected divergence or unavailable
wheel rather than allowing the installer to substitute an unpinned stack.

For example, obtain the Linux binary resolution of the baseline with:

```sh
uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_39 \
  --only-binary=:all: --torch-backend cpu --no-header --no-annotate \
  --extra benchmark --extra scanpy --extra torch \
  -c .github/ci/constraints-py312-scanpy112.txt \
  pyproject.toml .github/ci/requirements.in
```

For an intentional upgrade, adjust the relevant pins and resolve all affected
lanes before accepting changes. Preserve the separate CPU wheel index for
Torch; install the wheel's other dependencies under the lane constraints.
