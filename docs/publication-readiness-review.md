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
`0386248d…`. The Drive folder `sc-poincare backups` holds the Moignard manifest,
report, and hex pieces 00–30 of `arrays.npz` at 8,000 bytes plus chunk 31 at
5,896 bytes. Reading those 32 download responses back from the persisted tool
transcript, decoding them, and concatenating the hex produced 126,948 bytes
whose SHA256 is `75da11742aa4773b28f62463c3c6b63146024e812fb7aaaf494d2f864ad5ca17`.
That matches the local `arrays.npz`. The short earlier uploads of chunks 08 and
13 remain in the trash and were not used.
One BLAS thread and the default thread pool produce two different deterministic
Paul PCA20 hashes, `20a9f31a…` and `8efb90c0…`.
