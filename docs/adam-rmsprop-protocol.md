# Pre-registered rule: tangent Adam versus Riemannian RMSProp

This rule is written before any RMSProp fit. It does not authorize that fit,
add the optimizer, or change a default. The public graph-stress optimizer
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
checkpoint, including initialization. An epoch limit is not promotion.

The default dense guard stays on. Labels stay out of graph construction,
initialization, and fitting.

## Promotion guardrails already used

Adam is the baseline and RMSProp is the only candidate. The numbers below
are the development rules in
[embedding-followup.md](embedding-followup.md). They are not refit here.

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
A nonmatching PCA is not a substitute, so this comparison cannot be scored
until that matrix is recovered.

Riemannian RMSProp is not implemented in this tree. Implementing it is a
separate change and still does not start the fit.
