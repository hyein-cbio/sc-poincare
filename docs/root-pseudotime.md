# Root distance and pseudotime

The root rule was fixed before fitting: the one cell labeled `root`.
Pseudotime is the hyperbolic distance from that cell. Ery stage is the leading
integer of a label that ends in `Ery`. Labels did not enter the graph, the
median bandwidth, the initialization, or the objective.

The fit matches the [rare-label inspection](rare-population-diagonal.md):
public `Paul.csv` SHA256
`fed836ed28f148d9ee8efd8ec13ebd468f06ef9b92eeb6f9e218235d4ad2f21d`,
1,000 genes, no PCA, k=15, sigma=median, gamma=2, 200 updates, seeds 0–4.
Both objectives share the spectral start within a seed. Coordinates are the
lowest training loss. The epoch limit is not convergence. Aggregates are in
[root-pseudotime.json](root-pseudotime.json).

| Objective | Root disk radius | Median disk radius | Ery-stage Spearman |
|---|---:|---:|---:|
| symmetric KL | 0.7169–0.7201 | 0.8282–0.8285 | -1 |
| conditional KL | 0.7100–0.7111 | 0.8187–0.8189 | -1 |

The root sits inside the median radius and far from the origin. That is not
central branching. Conditional KL moves the root slightly inward and does not
put it at the origin.

Seed 0 symmetric medians, nearest to farthest from the root: root, 14Mo,
13Baso, 9GMP, 10GMP, 12Baso, 8Mk, 17Neu, 15Mo, 7MEP, 16Neu, 19Lymph, 18Eos,
11DC, then 6Ery through 1Ery. The Ery integers run in the opposite order from
root distance in every seed and both objectives. The integer is the label
text, not a separately measured differentiation clock. Five seeds give the
same Spearman value, so this reversal is not a one-seed accident. The ranges
above are the seed spread, not a confidence interval.
