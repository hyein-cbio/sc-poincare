# Global-study completion and first stress-weighting pilot

## Authorized scope and provenance boundary

This batch completed the interrupted Astra global-target protocol, inspected
same-objective/gradient mechanisms, and tried one relative-stress weighting
change. The subsequent broad comparison/promotion stage and graph-confidence
redesign were **not** started. No package objective, defaults, teacher rule or
initialization jitter was changed.

The original 37-fit capsule remains immutable under `data/astra37_handoff/`.
New records live separately in
`data/graph_diagnostics_integration/algorithm_followup/`. Original closure and
cancellation records are history, not rewritten as if the prior run continued
uninterrupted. No previous checkpoint hash was changed to authorize resume.

The capsule excluded regenerable dense setup caches. Regenerating them with
frozen source `3489e58` on the receiver reproduced feature/source hashes, but
not target/teacher/initializer bytes. Initial seed-0 coordinates differed from
saved baseline starts by at most 5.7e-15; original and receiver Python/Torch/CPU
runtimes differ. The following are therefore **new receiver-reconstructed fits
under the same mathematical recipe**, not exact old-state continuations.
New metadata records actual reconstructed hashes, and archived controls retain
their original provenance. Mixed-runtime baseline reuse is a limitation, not
silently relabeled exact replication.

## 1. Finish the interrupted global-target study

The missing `moignard_mutual_raw_euclidean_seed0` screening fit reached 1500
updates. The original global selector then ran with a caller-only loader/writer
adaptation: original seven screening outcomes and original baselines were
preserved, the new eighth outcome supplied, and output written to the new tree.
Frozen thresholds, candidate names and tie-breaking were not changed.

**No candidate qualified.** The conditional development-seed confirmations were
therefore not activated. Following the original no-candidate rule, both fixed
hypotheses transferred to Olsson in both geometries and seeds 0/1/2, reusing the
six archived Olsson baselines. This is completion of the old protocol, not the
new broad-validation stage deferred by the user.

All **13 new 1500-update fits completed**: one missing screening plus twelve
transfer fits. Elapsed supervisor time was **634.3 seconds**, with sampled
maximum child RSS about **1.51 GiB**. Limits were one worker, two threads, 45
minutes overall, 20 minutes per fit, 6 GiB child RSS and at least 2 GiB system-
available memory. No guard was disabled. States/checkpoint traces and Adam
step 1500 were verified on all thirteen outputs.

The newly completed Moignard Euclidean raw-edge screen, versus its archived
baseline, had macro15 delta -.000181, MSE15 +.000200, MSE30 -.000407 and stage
Spearman +.004910. These mixed endpoints do not overturn the frozen Poincare
selection rule or the previously observed guardrail violations.

### Completed Olsson transfer

Deltas below are candidate minus the paired archived baseline. Negative MSE
means improved reconstruction. Every-seed macro15/MSE30 gains plus the original
K15 protection gates are required; a favorable individual seed is insufficient.

| Condition | Geometry | Macro15 delta range | MSE30 delta range | All three seeds pass |
|---|---|---:|---:|---|
| union_log | Poincare | -.004730 to -.000829 | +.000266 to +.000423 | no |
| union_log | Euclidean | -.001745 to +.006545 | -.002448 to -.002009 | no |
| mutual_raw | Poincare | +.000655 to +.006472 | -.000019 to +.000906 | no |
| mutual_raw | Euclidean | -.002385 to +.003428 | -.000426 to -.000231 | no |

Union-log Euclidean improves MSE30 in every seed but fails other protection/
consistency gates. Mutual-raw Poincare improves macro15 in every seed but MSE30
worsens in two seeds. Thus neither is a joint, seed-consistent improvement.
Individual successes and failures are retained in `global_transfer.json`.

Together with the original seven global executions, this completes twenty
global quality executions; the original thirty local-teacher executions remain
separate. New receiver fits are not exact replays, blind cohorts or biological
replicates. No public target variant is promoted.

## 2. Same-objective columns and gradient anatomy, without updates

Seventy-two objective-column entries were freshly computed on fixed teacher/
global-study endpoints, including the newly completed screening point. Different
coordinates are compared within one target/teacher column, not by ranking each
model's own target loss. The maximum own-column difference from stored best
loss is **2.8e-17** despite the receiver cache byte differences. Existing
available archived cross-objective fields also agree to floating-point precision.

Four saved main endpoints (Paul/Moignard x MU fixed/local-radius) were separately
evaluated with no optimizer step. Their recomputed objective agrees with saved
loss within **8.3e-17**. The global and weighted-local gradients oppose one
another for essentially all cells (99.97% to 100%); median cosine is near -1.
This is compatible with cancellation near a fitted stationary compromise. It
is **not causal evidence** that opposing gradients caused a biological error,
or a proof that one component should be removed or intensified.

The target-distance floor region below normalized .01 has no nonself pairs on
these inputs. Roughly 20% of nonself pairs in normalized target range .1-.5
account for about 53% of relative-stress loss. Short-distance weighting matters,
but the evidence does not support blaming the floor on nearly zero distances
as the dominant problem in these two cases.

This receiving analysis uses actual available endpoints and explicit input
representations. Historical diagnostic scripts referring to missing old cache/
run paths were not blindly restarted, and earlier unavailable fixed-rule source
runs are not claimed as newly reproduced bit-for-bit.

## 3. Try one denominator-power change, not a new default

An isolated research harness changes only

`((d/s - T/s)^2) / max(T/s, floor)^power`

from power 2 to power 1. No graph, teacher, temperature, local loss weight,
optimizer, initialization or seed rule is changed. The production package is
not monkeypatched and this research state schema is explicitly not a public
MU resume payload. Changing the exponent changes the effective distribution
and scale of global pressure relative to the unchanged local term; it is not
an attribution experiment that isolates every optimizer/weight effect.

Before fitting, both geometries passed independent NumPy formula, finite-
difference and full-versus-chunk gradient checks. Power 2 matched the frozen
objective/gradients and ten-update baseline trajectories exactly.

Pilot: existing small Paul/Moignard inputs, 384 cells each, one fixed seed (0),
300 updates, powers 2/1 and native Poincare/Euclidean coordinates: **eight fits**.
Inputs and initial arrays match within each pair. Small-stage representations
are not pooled with full-stage PCA as if identical. This is a predeclared
mechanism smoke pilot, not the later multi-seed/convergence comparison stage.

| Input | Geometry | Macro15 delta, p1 minus p2 | MSE15 delta | MSE30 delta |
|---|---|---:|---:|---:|
| Paul | Poincare | -.003032 | +.001379 | +.000662 |
| Paul | Euclidean | +.003335 | -.001025 | +.001478 |
| Moignard | Poincare | +.005784 | +.004057 | +.003780 |
| Moignard | Euclidean | -.001889 | +.003159 | +.006495 |

Every K30 gene-MSE comparison is worse. Some macro15/K15 endpoints improve, so
not every metric failed, but this simple weighting change has **no consistent
joint gain**. No favorable-seed selection, extra exponent sweep or compensating
weight search was performed. It should not become a default or be described as
a general algorithm improvement from these results. Longer-budget behavior,
uncertainty and independent biology remain untested.

## Outcome and what was deferred at this stage

A subsequent user-authorized [stage3/4 study](stress-validation-34.md) extends
these pilots to three seeds/1500-update budgets and tests one separate input-k-
sensitivity weighting rule. It does not alter the original results or imply
independent-data/full-size validation.


- Interrupted protocol: completed under an explicit receiver reconstruction
  boundary; neither global candidate meets its original joint transfer rule.
- Requested mechanism check: freshly evaluated, but gradient opposition alone
  does not identify an intervention with biological benefit.
- First stress-power candidate: numerically valid and tested, with trade-offs
  rather than a consistent improvement; retain the negative evidence.
- Broad multi-seed/convergence/independent-data comparison and graph-confidence
  design are still deferred. The pilot does not justify automatically spending
  that larger budget on power 1.
- Main numerical defaults and initialization jitter remain unchanged. Existing
  source changes from the earlier teacher-observability adoption remain separate.
- The original capsule transport hashes are rechecked after this batch; no new
  training is run inside its tree. New execution outputs and compact metadata
  stay in the ignored latest-results tree, not Git or an external backup service.
- Main software regression suite: **517 passed**, two existing zero-target
  warnings. Research formulas/gradients/state checks are separately recorded;
  these are not new remote-CI claims or publication validation.

Fifteen regenerable dense caches (about 1.85 GB) were checksum-inventoried and
removed after state/objective verification. Frozen builders, actual setup hashes,
input paths and new unique checkpoints/results remain in the compact local
follow-up tree (about 5.6 MB); no raw source/input capsule was duplicated.

No commit or push was made by this batch. No research process is intentionally
left running after the final audit.
