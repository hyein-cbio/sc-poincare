# Pre-registered rule: tangent Adam versus Riemannian RMSProp

The selection rule below was recorded before the authorized 1,500-update
screen, following the exploratory 20-update probe. Recording the rule did
not itself authorize a fit or change a default. The opt-in implementation
and later execution are described below. The public graph-stress optimizer
remains tangent Adam, and the public geometry remains `poincare`.

The 20-update Moignard probe is a hypothesis. A lower loss or a higher macro
purity on that single run is not a promotion result.

## Arms

Both arms use one `GraphStressConfig()`: k=15, global bandwidth, and the
default scales, weights, radius, jitter, and learning rate. Geometry is
`poincare`. The objective, initialization, and seed are the same for both
arms. Only the optimizer differs. Dimension, curvature, SGD, and Lorentz are
not part of this comparison.

Selection uses seed 0, as in the existing development rule. Seeds 1, 2, 3,
and 4 are confirmation only after seed 0 meets every guardrail below. They
are a sensitivity description, not a confidence interval, and they are not
a second chance to choose the optimizer. The budget is 1,500 updates, the
existing graph-stress default, not 20. The lowest training loss selects the
checkpoint, including initialization. An epoch limit does not pass this screen.

The default dense guard stays on. Labels stay out of graph construction,
initialization, and fitting.

## Development screen already used

Adam is the baseline and RMSProp is the only candidate. The numbers below
are the development rules in
[embedding-followup.md](embedding-followup.md). They are not refit here.
Meeting them does not promote RMSProp: gene MSE and neighbor purity use the
cells that were just embedded. A promotion claim still needs a predeclared
score the fit does not see. The held-out mapping described below exists,
but this screen does not perform held-out validation.

- Paul: K15 macro purity gain at least 0.003, and mean gain for 17Neu,
  18Eos, and 19Lymph at least 0.005.
- Moignard: K30 normalized all-gene MSE reduction at least 0.001, and fixed
  PS input-medoid stage Spearman gain at least 0.005.
- On each development set, K15 overall and macro purity may fall by at most
  0.002, and K15/K30 MSE may rise by at most 0.001. Moignard stage decline
  is at most 0.005. No named Paul rare label may lose more than 0.01 K15
  purity.

Olsson is not a selection set. A candidate that misses a seed-0 guardrail
stops. It is not refit with a new budget or scored from a friendlier seed.
Paul may be scored only from the
feature matrix whose SHA256 is
`0386248d2c02c99f82cda2eb18b8e357b8e0399b10aba7e67575599fb115d90b`.
A nonmatching PCA is not a substitute. The stored `data/paul/features.npy`
bytes match this hash. That recovery does not start the comparison.

`controlled_probe.fit_intrinsic_rmsprop` implements the short-probe update.
`evaluation.place_heldout_cells` can map a cell the fit did not move, and it
is not one of the guardrails above.

## Authorized seed-0 execution

The 1,500-update screen was run after a separate request. Both arms used
`GraphStressConfig()`, Poincaré geometry, classical jitter seed 0, and the
512 MiB guard. Labels did not enter preparation or fitting. The scored point
is the lowest training loss, including initialization. Paul used the stored
matrix `0386248d…`. Moignard used `674f8b38…`. The PS feature medoid is row
1736. Initial losses match across arms.

RMSProp misses every seed-0 guardrail on both datasets. Its best losses are
higher (Paul 0.2130 at epoch 500 versus Adam 0.2083 at epoch 1500; Moignard
0.2426 at epoch 313 versus Adam 0.2375 at epoch 1499). Paul K15 macro purity
changes by −0.0097 and the 17Neu/18Eos/19Lymph mean by −0.0056. Moignard K30
MSE rises by 0.0072 and the stage Spearman falls by 0.0253. Seeds 1–4 were
not run. The tangent-Adam default is unchanged. Record:
[adam-rmsprop-screen.json](adam-rmsprop-screen.json).
