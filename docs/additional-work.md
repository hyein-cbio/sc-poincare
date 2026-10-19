# Work worth doing next

These items are outside the checklist in `development-roadmap.md`. They are
not authorized runs. Each one is narrower than repeating work already on
`main`.

The checklist itself still has three open items: the external review of
`494367a` (CodeRabbit is not authenticated), the full Planaria 1,000-update
comparison (this host has about 4.2 GiB available and no swap), and a fresh
approval before any further expensive run.

## 1. Land one approximation, after the Paul matrix matches

`main` has no matrix-free graph-stress path. This branch now has
`sc_poincare.landmark`: sparse teacher, landmark-column distances, and tests
that match exact graph stress when every cell is a landmark. A separate
unmerged branch, `cursor/landmark-reference-check-a572` at `a9f03d1`, also
ran the Moignard zero-epoch check. That measured run is not repeated here.

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
20-update development fit. The pre-registered Adam-versus-RMSProp rule is
[adam-rmsprop-protocol.md](adam-rmsprop-protocol.md): the same graph-stress
objective and initialization, seed 0 for selection, seeds 1–4 only as
confirmation, 1,500 updates, and the existing Paul and Moignard promotion
guardrails. The fit is not started. Riemannian
RMSProp is not implemented here, and Paul still needs feature SHA256
`0386248d…` before that comparison can be scored.

## 3. Separate held-out quality from transductive neighbor scores

Current gene MSE and neighbor purity use the cells that were just embedded.
They can reward a graph-construction artifact. A promotion claim needs one
predeclared score that the fit does not see, for example held-out genes or a
held-out cell mapped without moving the training coordinates. The present API
is transductive and has no such mapping.

## 4. Independent data beyond pbmc3k

pbmc3k was not in the Paul, Moignard, Olsson, ToggleSwitch, Myeloid,
krumsiek, or Planaria development tables. A frozen symmetric-KL rule and a
1,500-update local-radius graph-stress rule have now been run there. Overall
K15 purity is high, and dendritic cells stay near 0.48–0.53. Those runs are
not a certified blind holdout.

Paul, Moignard, Olsson, Pancreas, and Nestorowa remain development data.
LARRY was identified as unused and still has no fit. Do not start it until
row order, subset size, and an external backup are fixed.

## 5. Full Planaria only under the frozen gate

The 1,000-update corrected-versus-conditional comparison is specified in
`frozen-conditional-kl-protocol.md`. The public CSV hash and the 51-label
inventory are attached. Rare labels are those with at most 50 cells. The
default dense guard still refuses 21,612 cells on this host, so the fit has
not started. Run it only where that guard can pass without being disabled.

## 6. External review of `494367a`

Install CodeRabbit where a reviewer can authenticate. Review only that
commit against its parent `a4d836d`. A review of the current tree, and the
local note on the unmerged branch, do not close this item.

## 7. Publication checks at submission time

Confirm the license, citation and packaging rules that apply on the day of
submission. Package MIT does not relicense PoincaréMaps inputs or reference
coordinates. Do not promise JOSS eligibility from the current checklist.
