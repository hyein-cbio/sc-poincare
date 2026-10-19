# Work worth doing next

These items are outside the checklist in `development-roadmap.md`. They are
not authorized runs. Each one is narrower than repeating work already on
`main`.

## 1. Land one approximation, after the Paul matrix matches

`main` has no matrix-free graph-stress path. A separate unmerged branch,
`cursor/landmark-reference-check-a572` at `a9f03d1`, already compares landmark
columns with exact mutual-graph stress and refuses a disabled dense guard.
This branch does not add a second implementation.

That branch's open Paul check is still blocked. This session confirmed the
public `Paul.csv` source hash and confirmed that scikit-learn 1.9.1 full-SVD
PCA20 does not reproduce feature SHA256 `0386248d…`. The next step is to
recover the original 2,730×20 float64 matrix, or to pin the exact transform
that hashes to it, and only then rerun the frozen zero-epoch check. Do not
insert the nonmatching PCA scores.

The same branch's Moignard 32-landmark column gradient differs from exact MU
by relative L2 0.875. A useful follow-up is an approximation that meets a
predeclared gradient tolerance against exact MU on that matrix, with the
512 MiB guard left on. Speed without that match is not a scalability result.

The Moignard manifest upload is not an off-machine copy of `arrays.npz`.
That file still needs a checksummed copy outside the execution host.

## 2. Treat the short intrinsic-optimizer probes as hypotheses

The same unmerged branch ran 20-update Moignard comparisons of tangent Adam,
SGD, ball dimensions 2 and 3, curvatures 1 and 0.25, a Lorentz retraction and
Riemannian RMSProp. RMSProp had lower loss and higher macro purity than Adam
on that single development run. SGD did not move. Dimension 3 lowered loss
without raising macro purity.

None of these is on `main`, and none should become a default from a
20-update development fit. The worthwhile study is a pre-registered
Adam-versus-RMSProp comparison: same objective, same classical initialization,
more than three seeds, a budget long enough that the loss curve is not still
in its first drop, and the population and stage rules already used for
promotion. Write the rule before fitting.

## 3. Separate held-out quality from transductive neighbor scores

Current gene MSE and neighbor purity use the cells that were just embedded.
They can reward a graph-construction artifact. A promotion claim needs one
predeclared score that the fit does not see, for example held-out genes or a
held-out cell mapped without moving the training coordinates. The present API
is transductive and has no such mapping.

## 4. Independent data, not another development seed

Paul, Moignard, Olsson, Pancreas and Nestorowa are already development data.
A 20-update pbmc3k probe on the unmerged branch is not the independent-data
gate. The next biological run should freeze the input-only rule, keep labels
out of fitting, and use a dataset that was not used to choose that rule.
LARRY was identified as unused and still has no fit. Do not start it until
row order, subset size and an external backup are fixed.

## 5. Full Planaria only under the frozen gate

The 1,000-update corrected-versus-conditional comparison is specified in
`frozen-conditional-kl-protocol.md`. It is still the scientific gap on
`main`. Run it only on a host that clears the memory gate, with the marker
list attached first. Five seeds and the feature-distance rank are already
part of that protocol.

## 6. External review of `494367a`

Install CodeRabbit where a reviewer can authenticate. Review only that
commit against its parent `a4d836d`. A review of the current tree, and the
local note on the unmerged branch, do not close this item.

## 7. Publication checks at submission time

Confirm the license, citation and packaging rules that apply on the day of
submission. Package MIT does not relicense PoincaréMaps inputs or reference
coordinates. Do not promise JOSS eligibility from the current checklist.
