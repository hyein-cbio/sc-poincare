# Controlled follow-up: what limits the graph stress embedding?

The completed local-teacher study selected no candidate. The separate global-target
study was closed at the user's request before screening was complete. Union
edges improved Moignard's stage-order score but violated Paul population
protection criteria. No general improvement or uniquely hyperbolic benefit
has been established, so none of these variants is promoted to the API.

## Completion boundary

The local-teacher protocol completed all **30** scheduled 1,500-update fits.
The global-target protocol completed **seven** further screening fits. Its
Moignard `mutual_raw` Euclidean screening fit, formal selection, and all
conditional confirmations/Olsson transfer work were **not executed**. These
are cancelled work, not failed quality results. The queue was stopped from
launching new fits while its active Poincare fit completed normally. Frozen
plans and sources, completed states, interruptions and cancellation records
remain distinguishable in the private research capsules.

The [aggregate results](embedding-followup-results.json) report **37 fit
executions**: 27 candidate fits and ten baseline fits, including four exact
replays of prior configurations. Phase2 reuses four Phase1 development controls;
these are not additional executions or biological replicates. No global-target
candidate was selected, rejected by a completed selector, or tested for transfer.
The observed Paul guardrail violations already exclude `union_log` under its
frozen rule; the status of `mutual_raw` is incomplete.

## Why the current method remains unsatisfactory

The optional graph stress method makes the optimization and its targets
explicit, but its training objective cannot establish biological quality.
It fits distances derived from a selected graph and a distribution derived
from the same input features. Optimizing those targets can preserve a
construction artifact just as consistently as a useful biological relation.
The earlier measured comparison found Moignard improvements alongside Paul
population losses, and comparable native Euclidean improvements. Those
results motivate controlled mechanism tests rather than a general claim.

The global target depends on which edges are kept, how disconnected
components are joined, how edge lengths are transformed, and how shortest
paths are normalized. The local target depends on graph support, Gaussian
bandwidth and normalization. A single relative-stress/local-CE weighting
offers no guarantee that these requirements fit simultaneously in two
dimensions or that a fixed optimization budget reaches a suitable solution.
The global term's relative-error denominator gives short target distances
more weight for the same absolute distance error, up to its fixed floor.
Normalizing every target to mean two does not equalize this distribution
of weights or the optimization problem across graph choices.

Row-normalized cross entropy weights each cell by 1/n. It does not weight
each label or lineage equally. For symmetric pair distance d_ij and
Q_i=softmax(-d_i/gamma), the local term's derivative with respect to a
shared pair distance is

```
(P_ij + P_ji - Q_ij - Q_ji) / (n * gamma).
```

The teacher coefficient incident to cell i is therefore
`1 + sum_j P_ji`. Making P doubly stochastic makes this coefficient two;
it does not give each population equal total weight. A rare population
still supplies only its cell-count fraction of the local rows. This
distinction matters when overall purity and macro/population purity diverge.
For example, Paul's target labels contain 22 Neu, nine Eos and 31 Lymph
cells, or 62/2,730 = 2.27% of the outgoing local-row weight. Balancing does
not turn that into equal population weighting.
The baseline teacher comes from row-normalizing symmetric nonnegative
affinities and is reversible. Its resulting asymmetry is expected, not an
invalid probability model or evidence of asymmetric geometric distances.

A sparse local teacher and a full-support softmax also cannot coincide
exactly at finite distances: P assigns zero to nonedges while Q remains
positive for every nonself pair. Coupling the two row normalizations through
symmetric distances imposes further constraints. These are properties of
the objective, not proof that a particular cell type must be misplaced.

## Frozen mechanism tests and their planned follow-up

Both studies retain the original local-radius baseline, initialization,
optimizer, 1,500-update budget, temperature and local/global loss weighting.
Each candidate changes only the stated teacher or global target. Native
Euclidean controls use the same target and initialization array; this does
not equalize initial pairwise distances, physical step lengths or convergence.

| Study | Candidate | Controlled change |
|---|---|---|
| Local teacher | balanced | Symmetric Sinkhorn scaling on the original Gaussian support; no inserted edges or epsilon |
| Local teacher | multiscale | Equal mixture of independently row-normalized local-radius k15 and k30 teachers |
| Global target | union_log | Replace mutual global k15 graph with union k15; keep the original mutual-edge median in log1p |
| Global target | mutual_raw | Keep the stitched mutual k15 graph but use raw edge lengths before shortest paths |

The global-target candidates retain the original mutual-log initializer and
the original local teacher. Teacher and global variants are not combined.
Both plans were fixed before their new Olsson transfer results. The second
plan acknowledges the already observed Paul balanced seed-0 outcome.

Balanced starts with d=1 and repeatedly computes `r=d*(W@d)` and
`d=d/sqrt(r)`, returning `diag(d) W diag(d)` once the maximum marginal
error is at most 1e-8, with a 20,000-iteration cap. It rejects nonfinite,
empty or nonconvergent marginals, including an infeasible star-support
example. No epsilon, self edge or replacement support is inserted. The
real-input errors are below 1e-8; exact row stochasticity is the idealized
normalization used in the derivative explanation above.

Selection uses Poincare seed 0 on the known Paul/Moignard development data.
A candidate must meet at least one dataset target and all guardrails:

- Paul: K15 macro purity gain at least .003 and mean gain for the exact
  labels 17Neu, 18Eos and 19Lymph at least .005.
- Moignard: K30 normalized all-gene MSE reduction at least .001 and fixed
  PS input-medoid stage Spearman gain at least .005.
- On each development dataset, K15 overall and macro purity declines at most
  .002, and K15/K30 MSE increases at most .001. Moignard stage decline is at
  most .005. No named Paul rare label may lose more than .01 K15 purity.

The frozen plans allowed at most one candidate per study to proceed to development
seeds 1 and 2. Planned transfer used all three seeds and both geometries. If no development
candidate qualified, both fixed candidates were to transfer for diagnosis, without
reopening selection. Only the local-teacher study reached that stage. Olsson improvement requires positive K15 macro gain
and negative K30 MSE change in each seed, with the frozen K15 guardrails.

Exact replayed baseline fits are computational reproduction, not additional
independent evidence. Phase2 reuses matching Phase1 controls and records
their original hashes rather than counting them as new fits.

## Interpretation limits

All-cell label purity, macro purity, every population, input-neighbor recall
and variance-normalized gene neighbor MSE are reported at K5/10/15/30.
Gene MSE is transductive input smoothing, not prediction of held-out genes
or cells. Labels and developmental roots are evaluation inputs, excluded
from fitting, but the datasets and their earlier counterexamples are known.
Olsson is withheld from this selection, not a blind independent cohort.
Moignard's stage score uses the explicit ordinal map PS=0, NP=1, HF=2,
4SG=4SFG=3 and the input-feature medoid of PS. It is one supplied-label
ordering diagnostic on direct raw embedding distances, not an independently
measured lineage trajectory.

Three small-jitter seeds measure optimization sensitivity around the same
initializer. They are neither confidence intervals nor biological
replicates. Native Euclidean outcomes must accompany Poincare outcomes.
The published reference has different training/preprocessing provenance and
is a contextual comparison, not an equal-budget geometry control.

With raw Euclidean edge lengths, a union kNN graph containing all direct
K15 input neighbors has the same K15 retrieval in the untied case: the
triangle inequality prevents an omitted, more distant input point from
becoming closer by a path. This also holds with one fixed-median transform
`f(d)=log1p(d/m)`: it is increasing and subadditive, so any path satisfies
`sum f(edge) >= f(sum edge) >= f(direct_distance)`. Agreement of these local
graph scores is not multiple independent successes. Mutual topology and
larger-scale ranks still provide different diagnostics.

Finite population size also constrains purity. A population of m cells has
maximum K-neighbor same-label purity min(m-1,K)/K. Paul's nine Eos cells
therefore have a K15 ceiling of 8/15, and a singleton root has zero.
This does not justify changing the frozen metric after observing results.

Raw cross-entropy values across teachers differ in entropy; global stresses
across targets refer to different targets. Evaluate all embeddings against
the same objective column before comparing training values, and keep
biological/input diagnostics separate from those training values. Opposing
local/global gradients near a fitted stationary point alone do not prove
that their conflict caused an observed biological failure.

The exact API is transductive: it optimizes the coordinates of the supplied
cells against dense all-pairs targets. These mechanism tests do not provide
a validated out-of-sample mapping or remove quadratic storage and dense
initialization costs. Small-dataset gains would not validate full Planaria
cold-start fitting or a large-data approximation.

## Preservation and recovery

The study keeps immutable per-run scientific sources, explicit fitted
features, checkpoint/current/best coordinates, the full loss trace, Adam
state, target/teacher/initializer hashes and failed or interrupted runs.
External incremental archives retain unique records, with regenerable dense
caches inventoried separately. Upload access, bytes and SHA256 are confirmed
before declaring a file externally preserved.

During this follow-up a runtime configuration/transport interruption killed
the active processes while files persisted. The interrupted 200-update
checkpoint was validated against frozen sources and its semantic digest,
then resumed into a separate directory. This is an observed process
interruption, not evidence of a filesystem wipe or forced container
replacement. Prefix identity and state lineage must be included in the
final audit.

An actual empty-directory reconstruction from earlier archived bytes rebuilt
the pinned package sources, restored inputs and six dense prepared setups,
and executed a real 1,500-to-1,501 Adam continuation. Restored metrics and
continuation arrays matched separate checks. This uses the existing Python
runtime and does not establish recovery from every dependency/platform
change. Recovery-only updates are not quality-comparison fits.

## Completed local-teacher study

The frozen selector chose no candidate. All 30 scheduled fits finished: four exact earlier baseline replays, eight development candidate fits, and 18 known-dataset Olsson transfer fits. Both candidate rules failed the joint transfer requirement in both geometries.

| Condition | Geometry | Paul macro15 delta | Paul rare mean15 delta | Moignard MSE30 delta | Moignard stage delta |
|---|---|---:|---:|---:|---:|
| balanced | poincare | +0.001006 | +0.004334 | +0.000175 | +0.003351 |
| balanced | euclidean | +0.000294 | +0.007683 | -0.001470 | +0.006677 |
| multiscale | poincare | +0.000299 | +0.000329 | +0.001432 | -0.005423 |
| multiscale | euclidean | +0.000210 | +0.007100 | +0.000346 | -0.003912 |

Balanced misses both development targets. Multiscale also violates the Paul Eos guardrail and Moignard MSE15, MSE30 and stage guardrails. Euclidean balanced shows larger Moignard gains than Poincare; its observed improvement does not change the frozen Poincare selection policy.

| Olsson condition | Geometry | Macro15 delta range | MSE30 delta range | Frozen transfer rule |
|---|---|---:|---:|---|
| balanced | poincare | +0.005765 to +0.010655 | +0.001842 to +0.002263 | fail |
| balanced | euclidean | +0.003550 to +0.010434 | -0.000450 to +0.000186 | fail |
| multiscale | poincare | -0.007754 to -0.001620 | +0.001761 to +0.002250 | fail |
| multiscale | euclidean | -0.012429 to -0.006763 | +0.000079 to +0.000492 | fail |

Within each fixed teacher objective column, candidates lower total loss relative to baseline coordinates, while original global stress and original-teacher total increase. The target change produced a different observed compromise rather than a verified biological improvement. These fixed-budget comparisons do not establish global optima.

All 30 raw states passed integrity checks. Separate full evaluations matched 67,578 numeric fields within 1.13e-14; independent K15 population/MSE calculations matched exactly. These are arithmetic/recovery checks, not additional biological replication.

## Observed global-topology effect

The union global target has a substantial, metric-specific effect on
Moignard's fixed input-medoid stage Spearman: +.097925 in Poincare and
+.123207 in native Euclidean at seed0. Poincare K15 macro also rises
+.002344, while K30 gene MSE changes by +.000180. Euclidean improves
both K15 macro (+.005200) and K30 MSE (-.001211). These are positive
observations for the stated endpoints, without evidence of a
hyperbolic-specific gain.

Paul gives the counterexample under the same input-only rule: Poincare
macro15 changes by -.004837 and Lymph purity by -.023656 despite lower
gene MSE; native Euclidean macro and Lymph also decline. This violates
the frozen guardrails and disqualifies union_log. The result supports
global-graph sensitivity and dataset/endpoint trade-offs rather than
a statement that every metric failed or that a universal improvement
was established.

The completed Paul `mutual_raw` Poincare fit changes macro15 by -.000616
and rare-label mean15 by +.003429: it misses the Paul composite target,
while satisfying the Paul guardrails. Its native Euclidean macro15 changes
by -.003347. Moignard Poincare changes macro15 by +.001215, MSE15 by +.002019,
MSE30 by +.000792 and stage Spearman by -.003843. Its MSE15 increase
exceeds the frozen .001 guardrail, and it misses the Moignard composite
target. Its Euclidean counterpart was cancelled. The incomplete
study provides no seed consistency, selection or transfer conclusion for
this candidate.

## Maintained implementation

The mechanism candidates remain research conditions. The maintained change
is input-only observability: prepared metadata now reports teacher entropy
in nats, entropy and incoming-mass quantiles, maximum asymmetry and maximum
row-normalization error. Quantile levels are 0, .1, .5, .9 and 1. No labels,
population-specific weights, RNG calls, new graph choice or optimizer changes
are introduced.

The mean entropy is a lower bound on cross entropy over probability
distributions. It need not be attainable by a two-dimensional symmetric
distance model. Subtracting it yields teacher-specific KL; it does not make
different teachers comparable biological quality measures. Incoming mass
and asymmetry describe normalization, not population balance or an invalid
teacher. These diagnostics help expose target changes before interpreting
training losses as improvement.

Source and metadata hashes intentionally change with this addition. Strict
historical checkpoint compatibility therefore still requires the archived
original package sources. Numerical equivalence checks do not waive that
provenance contract.

## What remains unresolved

The next decision needs a target rule with explicit population and
developmental-order trade-offs, followed by independent biological data.
This study does not justify adding a dataset-specific exception, choosing
the best-looking seed, combining the tested variants, or changing the public
default. Optimizer/chart comparisons would need matched objectives and
initial distances; target redesign and optimization changes should be tested
separately. Scalable approximations also need an exact-reference comparison
before their speed can be interpreted as a practical improvement.

## Final verification and handoff

All 37 quality-comparison fits used the package core at `3489e58` and the
archived study-specific target/teacher builders, before maintained source edits. Their raw states passed integrity
checks. Separate evaluations covered all 37 unique fits and matched 103,363
numeric fields with maximum absolute difference 1.20e-14; separately computed
K15 purity/MSE checks matched exactly. This validates arithmetic and provenance,
not biological generalization.

After adding metadata, separate-process regression checks against `3489e58`
used the full Paul PCA20 features, both bandwidths, both geometries and 20
updates. All 50 prepared-array, coordinate, loss, best-state and Adam-state
comparisons matched exactly. The complete suite passed **210 tests**, with
one optional Scanpy test skipped. The three new analytic cases check uniform
triangle entropy and an asymmetric line's closed-form teacher.

An actual completed global-target checkpoint also continued from update 1,500
to 1,501 through the portable recovery helper. All 14 lineage/state checks
passed, its loss prefix was exact and its original files were unchanged.
Regression and recovery runs are excluded from the 37 quality executions.

Research is closed at the user's request. No new selection, confirmation,
transfer, hypothesis combination or expensive run is authorized by this
record. Raw research capsules and recovery evidence are retained in the
private handoff, separately from the repository's aggregate results. Local
commits are prepared for the user's manual push; no further push or PR is
part of this handoff.
