# Reference-condition search: scope, recipes, and limitations

## Historical numerical policy

The results below were measured before the log-domain KL correction, on the
source preserved in commit `9993dd0` (with the former `1e-12` probability floor).
They are historical evidence, not a guarantee that the same recipes reproduce
these values on current code. At that time, the stress-based research fitter
was separate from the default RFA and experimental conditional KL objective.
The current provisional AnnData default is MU; explicit `method="rfa"` selects
the RFA path. See the [migration guide](mu-default-migration.md),
[objective definitions](objectives.md) and the later
[objective validation study](objective-validation.md).

## What was reached

A search on the six previously selected small/medium datasets found conditions
whose **two prespecified graph-distance diagnostics** match or exceed the
published Facebook PoincareMaps coordinate artifacts, for each of seeds 0, 1, 2:

- local: 15-neighbor overlap with graph-geodesic nearest neighbors;
- global: Spearman correlation of graph-geodesic and hyperbolic distances.

This is **not equivalence of all analyses in the paper**, a statistical
non-inferiority study, or evidence of superior performance on unseen datasets.
Five selected recipes use a new experimental objective. Only the selected Olsson
recipe uses the existing pure-RFA KL fitter. A pure-RFA Paul recipe passed the
numeric diagnostics too, but its root-centered plot was too crowded to recommend.
It remains in the local trial history rather than being hidden.

Planaria and C. elegans were excluded. Reference preprocessing, actual optimizer
update counts and elapsed training budgets are not fully verified, so this is not
an equal-compute comparison. The implementation source of Facebook/CPM was not
consulted for this search; reference coordinates were used only for evaluation
and final orientation alignment, never as an initialization or fitting target.

## Evaluation and search

The evaluation graph uses k=20 and the same CSV feature representation/PCA for
both methods, with edge lengths determining shortest paths. Up to 256 anchor
cells were chosen with permutation seed 419. The first half was designated for
condition selection, the second for diagnostic validation. Thresholds were not
relaxed: both scores had to be at least the reference scores on both splits, for
all three seeds. The 18 selected raw embeddings were re-evaluated after selection.

Validation scores were logged during the search, not hidden from the operator.
Consequently this is **not a blinded or uncontaminated holdout test**. The fit is
transductive and sees all cells; validation anchors are not unseen observations.
Hyperparameters are dataset-specific. Labels were used only for plotting and
root selection, not fitting or condition selection. The metrics are not the
paper's Qlocal/Qglobal implementation.

The local archive contains 105 completed trial records, including multiple
seeds, failed parity candidates, pure-KL alternatives and staged continuations.
Exploration varied gamma, bandwidth, learning rate, epoch count, initialization,
optimizer and later geodesic regularization strength / graph neighborhood size.
Execution-error logs are retained separately; they are not performance results.

## Selected results

Diagnostic-validation means across three seeds (higher is better):

| Dataset | Reference local | Ours local | Reference global | Ours global |
|---|---:|---:|---:|---:|
| ToggleSwitch | .868 | .891 | .886 | .932 |
| MyeloidProgenitors | .788 | .821 | .950 | .980 |
| krumsiek11_blobs | .504 | .530 | .782 | .906 |
| Olsson | .643 | .689 | .938 | .953 |
| Paul | .297 | .353 | .893 | .936 |
| Moignard2015 | .252 | .272 | .936 | .964 |

Every selected seed individually passed, not just these averages. A subsequent
sensitivity check with evaluation graphs k=15 and k=30 also passed both metrics
for all selected seeds on all six datasets. This sensitivity check was not used
to choose conditions. With direct feature-space Euclidean distances instead of
graph geodesics, simultaneous dominance does NOT hold for Olsson and Moignard.
Thus the conclusion is specific to manifold-distance preservation.

## What changed in the experimental objective

Let P be the ordinary dense RFA target, d_ij the learned Poincaré distances,
D_ij the training graph's shortest-path distances and Dbar their off-diagonal
mean. The research fitter minimizes

```
mean_row_symmetric_KL(P, softmax(-d/gamma))
  + weight * sum_ij ((d_ij - scale * D_ij/Dbar) / scale)^2 / (N*(N-1))
```

The second term preserves global distances during optimization; it is **not a
display force**. It directly favors graph-distance preservation, so improvements
on graph-distance diagnostics must be interpreted in that light. In particular,
Moignard uses k=20 for both its training and primary evaluation graphs.

Initialization is classical scaling of the training graph distances to two
dimensions, shrunk near the origin and given small seeded noise. Adam optimizes
free spatial coordinates u mapped to the disk by

```
x = u / (1 + sqrt(1 + ||u||^2))
```

This is NOT Riemannian Adam. Both terms retain all comparison columns with
row-chunked gradient accumulation. No labels, reference coordinates, jitter,
repel or fan enter the objective. Each stage returns its lowest-objective
checkpoint, not the best evaluation-metric checkpoint.

## Recipes

Features are used as provided, with PCA only where listed; no added feature
standardization. Median sigma is the explicit positive-edge median policy.

| Dataset | k | sigma | PCA | gamma | Optimizer / initialization | Epochs / weight |
|---|---:|---|---:|---:|---|---|
| ToggleSwitch | 15 | 1 | none | .5 | spatial Adam / geodesic | 800 / 10 |
| MyeloidProgenitors | 30 | 2 | none | .5 | spatial Adam / geodesic | 800 / 10 |
| krumsiek11_blobs | 30 | 1 | 11 | .25 | spatial Adam / geodesic | 800 / 3 |
| Olsson | 15 | median (~22.30) | 20 | .25 | RSGD / spectral | 500 / 0 |
| Paul | 15 | median (~4.03) | 20 | .5 | spatial Adam / geodesic | 800 / 10, then 150 / 20 |
| Moignard2015 | 20 | median (~16.90) | none | .5 | spatial Adam / geodesic | 700 / 20, then 250 / 15 |

Spatial Adam learning rate is .02; Olsson RSGD uses .01 with 30 burn-in epochs
at .001. All use row chunks of 256. Stages start from the same seed's preceding
best coordinates and **reset Adam's state**. A staged run's total budget includes
all stages. Numeric loss values from different weights or targets are not
comparable.

## Reproduce without loading reference coordinates

These are **historical-only** selected recipes, not current recommended
workflows or new defaults. They live under `examples/historical/` and require
explicit acknowledgement; closed research/capsule inputs are not replay targets:

```bash
pip install -e '.[torch,benchmark]'
python examples/historical/run_selected_conditions.py --acknowledge-historical MyeloidProgenitors \
  data/facebook_comparison/reference/datasets/MyeloidProgenitors.csv \
  --seed 0 --output data/reproduced/myeloid_seed0.npz
```

Choose any dataset name in the recipe table and a corresponding local CSV.
Repeat with seeds 1 and 2. The runner writes raw coordinates, labels, per-stage
loss traces and metadata; it does not load or fit against reference coordinates.
It refuses to overwrite existing outputs. The extracted research fitter lives
in `examples/historical/experimental_geodesic_fit.py`. Its outputs reproduced the searched
ToggleSwitch and Myeloid seed-0 coordinates exactly in the test environment.
Unit tests cover its declared objective, stage accounting, row-chunk invariance,
reproducibility and input validation.

Both RFA and shortest paths/classical scaling require dense matrices. This is
not a scalable method and may use more memory than the pure-RFA fitter. The
existing working-array guard applies, but is not a process-RSS limit.

## Local evidence (not vendored into the MIT package)

- `data/parity_search/PROTOCOL.md`: initial operational criterion
- `data/parity_search/trial_index.json`: completed trial ledger
- `data/parity_search/selected_recipes.json`: exact selected runs and NPZ hashes
- `data/parity_search/summary.json`: threshold audit, source/diff provenance
- `data/parity_search/sensitivity.json`: alternative evaluation metrics
- `data/parity_search/*_selected.png`: reference versus selected seed-0 plots
- `data/parity_search/search.py`: original exploratory runner

The saved plots use the same labels, palette, cell order, point sizes and shared
root per dataset. Only an orthogonal rotation/reflection aligns orientations.
All cells are drawn; no display forces or nonlinear radius remapping is applied.
All downloaded data, reference coordinates and images remain under ignored
`data/`, subject to their original licenses. Nothing was committed or pushed.
