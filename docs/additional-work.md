# Work worth doing next

These items are outside the original checklist in `development-roadmap.md`.
Zero-epoch reference checks are recorded below. They are not fits, and they
do not authorize a later expensive run.

## 1. Land one approximation, after the Paul matrix matches

`sc_poincare.landmark` keeps a sparse teacher and landmark-column distances.
Tests match exact graph stress when every cell is a landmark. The recorded
Paul matrix is ignored `data/paul/features.npy`, file SHA256 `0d1d53e5…`,
float64 SHA256 `0386248d…`. The frozen zero-epoch check used 32 landmarks,
seed 0, and the 512 MiB guard. Columns agree within 2.2e-14. The column
gradient differs from exact MU by relative L2 0.844. sklearn full-SVD PCA20
still does not reproduce the hash. `(X - mean) @ components.T` from
`data/paul/pca.npz` is within 1.5e-13 absolute and was not substituted.
Record: [paul-landmark-reference.json](paul-landmark-reference.json).

The follow-up tolerance was relative L2 1e-3, declared before the Moignard
schedule. Counts 32, 128, 512, 2048, and all 3,934 cells were scored with the
full denominator and with an included-pair mean. The published 32-landmark
value 0.875 is reproduced. No partial count meets 1e-3. The included-pair mean
is worse at 32 landmarks (relative L2 5.20). Every-cell landmarks meet the
tolerance (relative L2 8.3e-17) because that case is the exact objective. It
is not promoted. Speed without a partial match is not a scalability result.
Record: [landmark-gradient-tolerance.json](landmark-gradient-tolerance.json).

The local Moignard `arrays.npz` still hashes to
`75da11742aa4773b28f62463c3c6b63146024e812fb7aaaf494d2f864ad5ca17`, the same
digest as the restored Drive hex pieces. This session did not upload another
copy.

## 2. Treat the short intrinsic-optimizer probes as hypotheses

The same unmerged branch ran 20-update Moignard comparisons of tangent Adam,
SGD, ball dimensions 2 and 3, curvatures 1 and 0.25, a Lorentz retraction and
Riemannian RMSProp. RMSProp had lower loss and higher macro purity than Adam
on that single development run. SGD did not move. Dimension 3 lowered loss
without raising macro purity.

None of these is on `main`, and none should become a default from a
20-update development fit. The pre-registered Adam-versus-RMSProp rule is
[adam-rmsprop-protocol.md](adam-rmsprop-protocol.md): the same graph-stress
objective and initialization, seed 0 for selection, seeds 1–4 only as
confirmation, 1,500 updates, and the existing Paul and Moignard development
screen. That screen does not promote the optimizer, because those scores are
transductive. The authorized seed-0 screen is recorded in
[adam-rmsprop-screen.json](adam-rmsprop-screen.json). RMSProp misses every
guardrail on Paul and Moignard, so seeds 1–4 were not run and the tangent-Adam
default is unchanged.

## 3. Separate held-out quality from transductive neighbor scores

Current gene MSE and neighbor purity use the cells that were just embedded.
They can reward a graph-construction artifact. A promotion claim needs one
predeclared score that the fit does not see. `place_heldout_cells` now places
held-out cells from feature distances to a frozen training disk and does not
write the training coordinates. It is not one of the Adam-versus-RMSProp
guardrails, and calling it is not a promotion.

## 4. Independent data beyond pbmc3k

pbmc3k was not in the Paul, Moignard, Olsson, ToggleSwitch, Myeloid,
krumsiek, or Planaria development tables. A frozen symmetric-KL rule and a
1,500-update local-radius graph-stress rule have now been run there. Overall
K15 purity is high, and dendritic cells stay near 0.48–0.53. Those runs are
not a certified blind holdout.

Paul, Moignard, Olsson, Pancreas, and Nestorowa remain development data.
LARRY still has no fit. Metadata, the normalized count matrix, and the clone
matrix all have 130,887 rows; that is not an author row-order contract. Full
exact MU would budget about 510.6 GiB. No subset size was frozen, and no
LARRY extract was backed up off this machine. The readiness decision is to
leave LARRY unfitted.

## 5. Full Planaria only under the frozen gate

The 1,000-update corrected-versus-conditional comparison is specified in
`frozen-conditional-kl-protocol.md`. The public CSV hash and the 51-label
inventory are attached. Rare labels are those with at most 50 cells. The
default dense guard still refuses 21,612 cells. On this host the estimate is
13.92 GiB against the 0.50 GiB limit, with about 4.4 GiB available and no
swap. The fit has not started. Run it only where that guard can pass without
being disabled.

## 6. External review of `494367a`

Install CodeRabbit where a reviewer can authenticate. Review only that
commit against its parent `a4d836d`. A review of the current tree, and the
local note on the unmerged branch, do not close this item. The CodeRabbit
CLI is not installed on this host, so this item stays open.

## 7. Publication checks at submission time

Confirm the license, citation and packaging rules that apply on the day of
submission. Package MIT does not relicense PoincaréMaps inputs or reference
coordinates. Do not promise JOSS eligibility from the current checklist.
