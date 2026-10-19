# Graph stress on pbmc3k

The preprocessing is the rule already fixed for the
[symmetric-KL pbmc3k run](pbmc3k-frozen.md): every raw cell, 10,000-count
normalization, `log1p`, 1,000 Seurat genes, and full-SVD PCA 20. The embedding
rule, also fixed before these scores, is local-radius graph stress, k=15,
tangent Adam, Poincaré geometry, and 1,500 updates. Seeds are 0, 1, and 2.
Labels are the processed `louvain` column, joined by barcode after fitting,
for 2,638 of 2,700 cells.

Every seed stops at the epoch limit. The best epoch is 1,500, 1,500, and
1,499, so the budget is not a plateau. Training loss is about 0.2762 and is
not comparable to the symmetric-KL loss. Feature-distance rank is
0.8817–0.8820. Overall K15 purity is 0.8960–0.8966.

| Group | K15 purity |
|---|---:|
| B cells | 0.9809–0.9821 |
| CD4 T cells | 0.9339–0.9350 |
| CD14+ Monocytes | 0.9314–0.9340 |
| NK cells | 0.8550–0.8580 |
| FCGR3A+ Monocytes | 0.8156–0.8187 |
| Megakaryocytes | 0.8089–0.8089 |
| CD8 T cells | 0.7146–0.7158 |
| Dendritic cells | 0.5225–0.5279 |

On the same preprocessing, corrected symmetric KL had overall purity near
0.882 and dendritic-cell purity near 0.476. Graph stress is higher on this
evaluation and still leaves dendritic cells and CD8 T cells well below the
overall number. Three seeds agree. This does not promote the experimental API
or change the default. Aggregates are in
[pbmc3k-graph-stress.json](pbmc3k-graph-stress.json).
