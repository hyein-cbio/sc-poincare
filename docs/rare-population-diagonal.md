# Rare populations when the RFA diagonal is discarded

The rule was fixed before fitting. A population is rare when its label count is
at most 31. On this Paul table that is 11DC (30), 17Neu (22), 18Eos (9),
19Lymph (31), and root (1). Labels were not used for the graph, the median
bandwidth, the spectral initialization, or the objective.

The input is public `Paul.csv`, SHA256
`fed836ed28f148d9ee8efd8ec13ebd468f06ef9b92eeb6f9e218235d4ad2f21d`.
All 1,000 numeric gene columns were used. No PCA was applied, so this is not
the historical PCA20 feature hash. k=15, sigma=median (30.072), gamma=2,
learning rate 0.01, burn-in 30, 200 updates, seeds 0–4. Both objectives share
the spectral initialization within a seed. Returned coordinates are the lowest
training loss. An epoch limit is not convergence. Exact values are in
[rare-population-diagonal.json](rare-population-diagonal.json).

K15 purity is the fraction of a cell's 15 embedding neighbors that share its
label. A singleton root has no same-label neighbor, so its purity is zero for
any embedding.

| Objective | Overall purity | Rare-label mean |
|---|---:|---:|
| symmetric KL | 0.2584–0.2630 | 0.2211–0.2233 |
| conditional KL | 0.2491–0.2553 | 0.2261–0.2267 |

Conditional minus self-inclusive, across the five seeds:

| Population | Purity change |
|---|---:|
| All cells | -0.0139 to -0.0059 |
| 11DC | +0.0022 |
| 17Neu | +0.0030 to +0.0091 |
| 18Eos | 0 |
| 19Lymph | +0.0086 to +0.0194 |
| root | 0 |
| Rare mean | +0.0034 to +0.0049 |

Discarding the diagonal lowers all-cell purity in every seed. It does not lower
the predeclared rare-label mean. 17Neu stays near 0.02 under both objectives, so
the diagonal is not what removes that population, and removing the diagonal does
not restore it. Five seeds agree on the sign of the overall and rare-mean
changes. This is not an independent cohort and not a reason to change the
default.
