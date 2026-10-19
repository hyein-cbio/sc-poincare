# Decision on the population and order trade-off

No rule is promoted. The decision uses the runs already recorded in this
branch. It does not pick the seed or the objective with the most attractive
score.

On Paul, with 1,000 genes and 200 updates, dropping the RFA diagonal lowers
all-cell K15 purity in every seed from 0 to 4 and does not lower the
predeclared rare-label mean. 17Neu stays near 0.02 under both objectives
([rare-population-diagonal.md](rare-population-diagonal.md)). The unique root
cell stays near disk radius 0.72, inside the median radius and far from the
origin. Ery label integers are perfectly anti-aligned with root distance in
every seed ([root-pseudotime.md](root-pseudotime.md)).

On pbmc3k, corrected symmetric KL reaches overall K15 purity near 0.882 while
dendritic cells stay near 0.476 ([pbmc3k-frozen.md](pbmc3k-frozen.md)).
Local-radius graph stress, on the same PCA and for 1,500 updates, reaches
about 0.896 overall and about 0.525 for dendritic cells
([pbmc3k-graph-stress.md](pbmc3k-graph-stress.md)). The best epoch is the
budget limit.

Overall purity, rare-population purity, and developmental order do not move
together. A higher overall score is not a reason to adopt conditional KL or
graph stress. Training loss is not compared across those objectives. The
public default stays corrected self-inclusive `symmetric_kl`.
