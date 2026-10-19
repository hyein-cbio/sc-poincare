# Stage 3/4: longer-budget stress validation and k-sensitivity weights

## Scope and fixed protocol

This user-authorized follow-up separates two experiments:

1. Extend the prior power1 denominator pilot to multiple seeds and longer budgets.
2. Diagnose input kNN sensitivity and test one input-only pair-weighting rule.

No candidate combines those changes. The baseline is MU fixed/global local
teacher, power2 relative stress. Teacher, k15 target, initializer, jitter .0002,
Adam, gamma, local loss weight and row chunk size remain shared within paired
comparisons. No package/default or public checkpoint schema was changed.

The protocol and numerical checks were frozen before the new fits. Inputs are
the existing small Paul/Moignard matrices (384 cells each) and Olsson (382 cells),
seeds 0/1/2, both geometries, 1500 updates. There are **54 fits**: 3 inputs x
3 conditions x 3 seeds x 2 geometries. Baselines are shared between the two
comparisons rather than refit or counted twice. The eight earlier 300-update
seed0 runs are exact prefix reproductions, not additional independent evidence.

Olsson was already known and used previously; it is a transfer check, not a
blind cohort. Small Paul PCA is its own subset representation, not the full-data
PCA. This study is not an independent-data/full-size benchmark. LARRY's provenance,
cohort extraction and data-use gates are not resolved by this experiment.

## Candidate definition: sensitivity, not biological confidence

Using only fitted features:

- Construct stitched mutual graphs at k10 and k30.
- Apply log1p(edge / original k15 median) to both, preserving the edge mapping.
- Normalize each geodesic target to nonself mean2, as for the original k15 target.
- Compute `a=(abs(T10-T15)+abs(T30-T15))/max(T15,scale*relative_floor)`.
- Set symmetric pair weights `w=1/(1+a)`, diagonal0, then divide by the nonself
  mean weight so total global pair weight remains unchanged.
- Multiply only the original power2 global stress pair terms by w. Keep the
  local teacher, target T15, optimizer and initialization unchanged.

No pairs are removed, no Gaussian/bridge floors are inserted, and labels/roots
never enter the builder. The existing relative floor is explicit in the
sensitivity denominator. Mean pair-weight normalization does not equalize each
cell's or label population's influence and does not equalize the actual loss or
gradient scale. The weights are a sensitivity heuristic, not calibrated confidence
that an input relation is biologically correct.

Both geometries passed independent NumPy formula, finite-difference and
full/chunk gradient checks. Constant weights reproduce the baseline objective
and gradients exactly; the unweighted ten-update trajectories match main exactly.
Permutation equivariance and positive nonself mean-one weights were also checked.
Research states have their own schema and semantic digest, not a main-MU resume
contract.

## Execution and epoch selection

All **54 fits completed** in **241.4 seconds**, with two one-thread workers and
sampled peak aggregate child RSS about **1.05 GiB**. The fixed caps were 20 minutes
overall, 120 seconds per job, 6 GiB aggregate child RSS and 2 GiB minimum available
system memory. Failed/partial/unstarted slots remain part of the supervisor
schema; none occurred in this run.

At budgets 300, 750 and 1500, each condition reports its own-objective best among
all earlier epochs, plus current coordinates and a residual tangent-gradient
norm. No metric-selected epoch or favorable-seed selection is used. Fixed update
counts do not imply equal compute; per-budget timing and total fitting time are
recorded separately. Same initial tangent arrays across geometries do not mean
identical initial distances or equal physical step sizes.

All 54 final stored best arrays/traces, semantic state digests and Adam step1500
were verified. For both geometries and powers1/2, the eight prior seed0 pilot
loss prefixes through epoch300 matched exactly. Data and source hashes are kept
in the local environment inventory.

## Stage 3 result: power1 does not recover with the longer budget

Ranges are candidate minus paired power2 baseline at the 1500 budget. Positive
MSE means worse reconstruction. These are three jitter-seed ranges, not confidence
intervals.

| Input | Geometry | Macro15 delta range | MSE30 delta range |
|---|---|---:|---:|
| Paul | Poincare | -.012230 to -.006502 | +.000681 to +.001943 |
| Paul | Euclidean | -.002020 to -.000490 | +.000869 to +.001033 |
| Moignard | Poincare | -.002109 to +.001458 | +.003102 to +.004953 |
| Moignard | Euclidean | -.000615 to +.000818 | +.005064 to +.005819 |
| Olsson | Poincare | +.000212 to +.003037 | +.001612 to +.002134 |
| Olsson | Euclidean | +.000909 to +.003222 | +.001915 to +.002205 |

MSE30 worsens in all 18 comparisons. Olsson macro improves consistently but
its gene reconstruction worsens; this is a trade-off, not joint transfer success.
Moignard Poincare stage Spearman also declines in each seed (about -.0113 to
-.0144). The necessary joint rule fails on every dataset/geometry seed group.

For Paul Poincare, final tangent-gradient norms are roughly 4e-10 to 4.8e-7 for
power1, and relative current-loss improvement from1000 to1500 is at most .000181.
The observed disadvantage is therefore not removed simply by extending the
300-update pilot to1500. This does not prove global optimality, arbitrary-start
robustness or universal convergence; other cases retain residual motion and some
Olsson final losses oscillate. No default change is warranted.

## Stage 4 result: sensitivity is informative, but weighting is not a joint fix

| Input | Geometry | Macro15 delta range | MSE30 delta range |
|---|---|---:|---:|
| Paul | Poincare | -.004076 to +.004374 | +.000341 to +.000810 |
| Paul | Euclidean | +.004026 to +.015678 | -.000087 to +.000813 |
| Moignard | Poincare | -.000475 to +.003446 | +.000624 to +.001093 |
| Moignard | Euclidean | -.001154 to +.001212 | +.000324 to +.002480 |
| Olsson | Poincare | +.005385 to +.007750 | +.001434 to +.002058 |
| Olsson | Euclidean | +.000813 to +.004064 | -.000088 to +.000385 |

Poincare MSE30 worsens in all nine comparisons. Paul Euclidean and Olsson
Poincare show repeatable macro gains, but these do not satisfy the joint MSE/
protection rule across seeds. No dataset/geometry group passes all three seeds.
Moignard stage changes are mixed (+.00136, -.01253, -.00303), not a consistent
ordering improvement.

### Diagnostic associations and bias

On the predeclared seed0 baseline endpoints, average input k-disagreement has
Spearman association with per-cell purity error of about .167 (Paul), .355
(Moignard) and .237 (Olsson). Its association with gene MSE is .297/.299/-.109,
respectively. Thus disagreement is not a universal reconstruction-error proxy.
These are descriptive associations on known data, not causal/significance or
confidence-calibration results.

Pair weights vary materially: approximately .170-1.260 in Paul, .190-1.225 in
Moignard and .146-1.187 in Olsson. Mean per-cell weights also vary; Paul's Baso/GMP
subgroups include some below-average rows. The rule can reduce global pressure
on sparse but genuine relations. Incoming/row weighting alone does not explain
all observed biological-label losses, and was not adjusted using labels.

Rare-group interpretation is particularly limited in the fixed small Paul input:
17Neu/18Eos/19Lymph have only 2/3/4 cells, with K15 ceilings1/15,2/15,3/15.
Power1 loses .0667 Lymph purity in every Poincare seed. Sensitivity weighting
loses .0667 Eos purity in seeds0/2 and .05 Lymph in seed2. Counts and ceilings
are retained rather than resampling a favorable cohort. These small counts cannot
validate rare-population biology, even if an aggregate gate had passed.

### Display usability remains separate

Predeclared seed0 raw Poincare views use600px disk axes and4px markers, without
root search, jitter, fan or repulsion. Native Euclidean views have common native
limits per input, not disk projection or fabricated disk-overlap scores.

Paul cross-label marker-center proximity is .0208 baseline, .0469 power1,
.0104 sensitivity-weighted: a real display-proxy gain for weighted seed0 despite
quality losses. Moignard is .0469/.0859/.0625, respectively; Olsson is zero for
all three. These are center-distance proxies, not raster occlusion, biology or
an assertion that every view is worse. All six comparison figures are retained.

## Decision and preservation

Neither candidate is promoted. Stage3 strengthens the negative/mixed power1
evidence beyond one short seed; stage4 shows that identifying sensitivity and
using it as a training weight are different questions. Keeping k-sensitivity as
an observational warning may be useful, but this study does not justify an
automatic downweighting/default rule or a dataset-specific exception.

No extra k/exponent/weight sweep, candidate combination, curvature/optimizer
change, root search or retries were added after seeing results. No full Planaria
or independent LARRY fitting was attempted. The next scientific gate remains
independent biological validation, not extending this small study indefinitely.

Local records, source/protocol hashes, complete per-population/gene scores,
budget trajectories, semantic states and figures live under
`data/research_archive/graph_diagnostics_integration/stress_validation_34/`.
The completed tree is preserved and excluded from normal replay; see
[containment policy](research-containment.md). Dense setup targets
are transient rather than archived repeatedly. Original capsule files are
reverified after the batch. These ignored local results are not an off-machine
backup and do not get pushed with code.

Software regression: **517 tests passed**, two existing exact-zero-target
warnings. Research formula/gradient/prefix/state checks are separately recorded.
No package algorithm/default or prior checkpoint was modified by this batch.
No commit/push or unattended research process is left by this work.
