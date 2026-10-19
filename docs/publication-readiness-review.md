# Publication-readiness review

This reviews the current package surface. It does not decide journal or
preprint eligibility, and it does not relicense datasets.

## License

`LICENSE` is the MIT license, copyright 2026 Hyein Cho. `pyproject.toml` names
the same author and `license = { text = "MIT" }`. The README states that input
artifacts keep their own licenses. `data/` is gitignored, so the Moignard
feature matrix used for the landmark reference tree is not part of the
installed wheel.

## Privacy

Tracked Python, Markdown, YAML, TOML, and JSON files have no `api_key`,
`secret`, `password`, or private-key markers. `.gitignore` excludes `data/`,
`/work/`, and `pi-session-*.html`. The Moignard restore record stores hashes
and metric values, not expression values.

## Installation

The documented install is `pip install -e '.[torch]'`, with `benchmark` and
`scanpy` extras. Runtime requires Python 3.10+, NumPy, SciPy, scikit-learn,
AnnData, and Matplotlib. MU fitting imports Torch lazily. The test workflow
builds a wheel and runs the suite outside the source tree.

## Reproducibility

The public Moignard CSV hash and the label-stripped 3,934×42 feature hash match
the recorded provenance. The zero-epoch landmark check and a checksummed local
reread agree exactly. Public Paul.csv matches its source hash, and sklearn
1.9.1 full-SVD PCA20 does not match the recorded Paul or Olsson feature hashes.
Those PCA matrices were not substituted. The same package versions on this host
produce a deterministic Paul PCA20 hash, `20a9f31a…`, which is not the recorded
`0386248d…`. The Drive folder `sc-poincare backups` holds the Moignard manifest
and report. Hex pieces 00–25 and 31 of `arrays.npz` are there. Chunks 26 through 30
are not, so the coordinate archive is not yet an off-machine copy.
One BLAS thread and the default thread pool produce two different deterministic
Paul PCA20 hashes, `20a9f31a…` and `8efb90c0…`.
