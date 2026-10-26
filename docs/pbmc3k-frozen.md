# Frozen rule on pbmc3k

pbmc3k raw counts are not one of the development tables used for Paul,
Moignard, Olsson, ToggleSwitch, Myeloid, krumsiek, or Planaria. The rule was
written before the scores below were read.

1. Keep every cell in `scanpy.datasets.pbmc3k`. Do not filter by labels or by
   a threshold chosen from an embedding.
2. Normalize each cell to 10,000 counts and apply `log1p`.
3. Select 1,000 Seurat highly variable genes and take a full-SVD PCA of those
   dense columns, 20 components.
4. Fit corrected self-inclusive `symmetric_kl` with k=15, sigma=median, gamma=2,
   learning rate 0.01, burn-in 30, and 200 updates. Seeds 0, 1, and 2 share
   that rule. Labels do not enter the graph, PCA, initialization, or fit.
5. After fitting, score K15 label purity only for the 2,638 barcodes that also
   appear in `pbmc3k_processed`, using its `louvain` column.

Sigma resolved to 3.667. Every run stopped at the epoch limit. Returned
coordinates are the lowest training loss. The loss range is 13187.74–13189.87.
Feature-distance rank is 0.8208 in every seed. Overall K15 purity is
0.8820–0.8821. Per-group purity:

| Group | K15 purity |
|---|---:|
| B cells | 0.9702–0.9737 |
| CD14+ Monocytes | 0.9297–0.9310 |
| CD4 T cells | 0.9242–0.9247 |
| NK cells | 0.8190–0.8247 |
| FCGR3A+ Monocytes | 0.7773–0.7827 |
| Megakaryocytes | 0.7022–0.7022 |
| CD8 T cells | 0.6897–0.6922 |
| Dendritic cells | 0.4757–0.4775 |

The three seeds agree. Dendritic cells and CD8 T cells stay well below the
overall purity, so the stable overall number does not mean every population
is recovered. This is not a certified blind holdout, not a 1,500-update study,
and not a reason to change the default. Aggregates are in
[pbmc3k-frozen.json](pbmc3k-frozen.json).
