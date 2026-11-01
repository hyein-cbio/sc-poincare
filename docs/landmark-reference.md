# Landmark reference check

`sc_poincare.landmark` compares one declared approximation with exact mutual-graph
stress. It does not change `prepare_graph_stress`, `fit_graph_stress`, the
public MU default, or the checkpoint source hashes of `experimental.py`,
`_proximity.py`, `_fitting.py` and `_disk.py`.

## Interpretation of historical controlled probes

The recorded Moignard and pbmc3k JSON files remain unchanged historical
measurements; they have not been regenerated with the current implementation.
In particular, `moignard-lorentz-probe.json` predates the corrected Minkowski
gradient conversion and finite-gradient distance calculation, and
`moignard-curvature-probe.json` predates evaluation in the matching curvature
geometry. Their optimizer and curvature comparisons must not be treated as
validation of the corrected code. New numerical results require a separately
authorized rerun, not a retrospective edit of the archived measurements.

Lorentz and intrinsic RMSProp probes now backpropagate one row chunk at a time.
Replacement teachers must be nonnegative, zero-diagonal row distributions.
Neighborhood evaluation still accepts higher-dimensional embeddings, but the
fixed-disk visibility diagnostics and plots require exactly two dimensions.
These corrections do not promote any probe to a default.

## What is frozen before the check

`freeze_comparison` binds the feature hash, `GraphStressConfig`, landmark count,
seeds, primary metric, sensitivity metrics and a positive `max_dense_bytes`.
`max_epochs` must be 0. Labels must already be excluded. Known development
identifiers `paul`, `moignard`, `moignard2015`, `pancreas` and `nestorowa`
cannot be declared `unseen_biological`. Any unseen role can be declared for
another identifier and is still refused by `compare_landmark_reference`.
Refusal is not a certificate that some other name is a blind holdout.

## What the check measures

On the exact classical-scaling initializer, with no Adam step:

1. Independent landmark shortest paths against the exact target columns, both
   scaled by the exact nonself geodesic mean.
2. Triangle completion `min_l d(i,l)+d(l,j)` against the exact geodesic. This
   is an upper bound. Its own nonself mean is compared with the exact mean.
3. Landmark MDS, after the same maximum-radius scaling, against exact classical
   scaling through orthogonal Procrustes, including reflection.
4. When requested, both-endpoint gradients of the unchanged exact objective on
   the completed target, and of the landmark-column objective. The column
   objective keeps the exact local cross-entropy and the `N*(N-1)` denominator.
   Pairs that do not touch a landmark contribute nothing. When every cell is a
   landmark, both gradients are the exact objective's gradient. Chunk size must
   not change that column gradient.

`fits_executed` is 0 and `biological_validation` is false. `write_reference_tree`
stores `report.json`, `arrays.npz` and `MANIFEST.json`, then rereads the
checksums. That reread is not an external backup and not a fitted-model
re-evaluation.

## What this does not establish

Synthetic recovery with every cell used as a landmark shows the reference path
agrees with exact MU. The tested small landmark set has a different target:
the tests require a strict triangle excess and a changed column gradient.
This is not true of every smaller set: N−1 landmarks include every nonself
pair and recover the exact gradient when exact distances and the exact
`global_mean` are supplied, as in `column_gradient_relative_l2`. This does
not make `prepare_landmark_columns` exact at N−1: its landmark-column mean
need not equal the exact mean. Neither result is the Paul (2,730) or Moignard (3,934) reference, and neither authorizes
disabling the dense guard, full Planaria, or a new default.

The completion still allocates an N-by-N matrix inside a run that already built
the exact target. `prepare_landmark_columns` / `fit_landmark_columns` are the
separate matrix-free path: they store landmark distances and a sparse teacher.
`max_dense_bytes=None` is rejected. When every cell is a landmark, the column
mean is the exact nonself mean and a synthetic Adam trajectory matches
`fit_graph_stress`. Fewer landmarks generally use a different normalization
and objective.

## Exact Moignard reference

`Moignard2015.csv` from the public PoincaréMaps dataset matches the published
source SHA256 `3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1`.
Dropping the last column reproduces the 3,934×42 feature SHA256
`674f8b380ba701d56afdeae8b35edf8b1269dfaa8e8483bc7e9c7e9c12999aa0`.
Labels were removed before preparation. The zero-epoch check used 32 landmarks,
seed 0, and the default 512 MiB guard. It took 12.3 seconds and fitted nothing.
Landmark columns agree with the exact target to 1.8e-14 absolute error. The
triangle excess max is 11.27, the completion-mean relative error is 0.437, and
the column-objective endpoint gradient differs from exact MU by relative L2
0.875. See [moignard-landmark-reference.json](moignard-landmark-reference.json).

A separate 2-epoch landmark-column fit on that same feature array stores no
dense target (workspace 4.0 MiB against an exact-dense estimate of 472 MiB).
Its normalization is the landmark-column mean, not the exact nonself mean, so
it is not an exact-MU continuation. Phase RSS samples are in
[moignard-landmark-column-fit.json](moignard-landmark-column-fit.json).

## Exact Paul reference

Ignored `data/paul/features.npy` is the recorded 2,730×20 matrix. Its file
SHA256 is `0d1d53e52f404a74bc8211b2441174adace7f63cbd57965d91c440552f20fba5`,
and its float64 bytes are
`0386248d2c02c99f82cda2eb18b8e357b8e0399b10aba7e67575599fb115d90b`.
The zero-epoch check used 32 landmarks, seed 0, and the default 512 MiB guard.
It took 7.8 seconds and fitted nothing. Landmark columns agree with the exact
target to 2.2e-14 absolute error. Triangle excess max is 11.48, the
completion-mean relative error is 0.465, and the column-objective endpoint
gradient differs from exact MU by relative L2 0.844. A reread of
`data/paul_landmark_reference` reproduced the primary value with absolute
difference 0. See [paul-landmark-reference.json](paul-landmark-reference.json)
and [paul-landmark-restore.json](paul-landmark-restore.json).

`data/paul/pca.npz` stores components, mean, and explained variance.
`(X - mean) @ components.T` on the label-stripped public CSV stays within
1.5e-13 absolute of the recorded features. The recomputed bytes were not
used. sklearn full-SVD PCA20 still does not reproduce `0386248d…`.

The same Moignard matrix was then used for a tolerance declared before the
schedule: relative L2 at most 1e-3. Counts 32, 128, 512, 2048, and 3,934 were
scored with the full denominator and with an included-pair mean. None of
the tested partial counts passes. Every-cell landmarks pass for both reductions, at relative L2
8.3e-17, and stay unpromoted. The 32-landmark full-denominator result matches
the earlier reference to 1.1e-16. See
[landmark-gradient-tolerance.json](landmark-gradient-tolerance.json).

`column_gradient_relative_l2` validates the prepared setup and a positive
`max_dense_bytes` (default 512 MiB), rejects malformed indices/distances, and
checks chunk invariance with a neighboring row-block size rather than forcing
a full-N autograd batch. These safeguards do not make the exact setup sparse.
See the [critical-review follow-up](paul-landmark-critical-review.md) for
regression coverage and source-pinned Moignard replay after integration.

Public `Paul.csv` matches source SHA256
`fed836ed28f148d9ee8efd8ec13ebd468f06ef9b92eeb6f9e218235d4ad2f21d`.
The provenance runtime names NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1,
and pandas 3.0.6. On this host those same versions, with one BLAS thread,
turn the 2,730×1,000 label-stripped matrix into PCA20 scores whose contiguous
float64 SHA256 is
`20a9f31a06557fef8e8aa5b0e5cddd96e5adc157663d22cf0fda0959b43ae300`.
A second call produced the same bytes. The recorded feature SHA256 is
`0386248d2c02c99f82cda2eb18b8e357b8e0399b10aba7e67575599fb115d90b`.
NumPy's SVD of the centered matrix differs from the sklearn scores by at most
1.8e-12 and hashes differently. OpenBLAS 0.3.34 one-thread Haswell, Sandybridge,
Nehalem, and Penryn kernels hash to `93eb6f3a…`, `dc67f1b0…`, `d5b1d157…`, and
`b24766fa…`. SciPy `gesvd` on one SkylakeX thread hashes to `b96251a8…`.
sklearn's full solver matches SciPy `gesdd` with the V-based sign flip. A
gray-code enumeration of all 1,048,576 column-sign patterns misses the recorded
hash for the one-thread scores, the default-thread scores, the one-thread
Haswell, Sandybridge, Nehalem, and Penryn scores, SciPy `gesvd`, and the stable
Torch 2.14.1 scores `c6e41671…`. The same full-SVD call misses the recorded
Olsson hash. This checkout has no `data/astra37_handoff/` capsule. Those PCA
reconstructions were not substituted. The matching bytes are the stored
`data/paul/features.npy` matrix used for the Paul reference above.

The ignored tree `data/moignard_landmark_reference` stores the checked feature
matrix, report, and checksum manifest. `reevaluate_study_tree` reran the frozen
check from those bytes and reproduced the primary value with absolute
difference 0. See [moignard-landmark-restore.json](moignard-landmark-restore.json).
The checksum manifest and the 2,095-byte report were copied to the maintainer's
Google Drive backup folder and the downloaded manifest matches
`262fa3960267a859eac9a29d3b47f3ba0246ae789eea339df6da63f0a535bf19`.
Drive holds `arrays.npz.hex.00` through `arrays.npz.hex.30` at 8,000 bytes
and `arrays.npz.hex.31` at 5,896 bytes. The earlier short uploads of chunks 08
and 13 stay in the trash and were not part of the restore. Each of the 32
current download responses was read back from the persisted tool transcript,
base64-decoded, and matched the corresponding local hex file byte for byte.
Concatenating those hex texts and decoding them produced 126,948 bytes whose
SHA256 is `75da11742aa4773b28f62463c3c6b63146024e812fb7aaaf494d2f864ad5ca17`,
the same as the local `arrays.npz`. The zip contains `landmark_index.npy`,
`exact_initial.npy`, and `landmark_initial.npy`.
