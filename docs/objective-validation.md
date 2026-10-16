# Objective ablation: evidence and limits

This study motivated the log-domain numerical correction and an **opt-in**
self-excluding objective. It is not a declaration that the package is universally
superior or biologically equivalent to the reference method. Within explicit
`method="rfa"`, the objective default remains self-inclusive symmetric KL.
The current provisional AnnData method default is MU, a separate model; see
[objective definitions](objectives.md) and [migration](mu-default-migration.md).

## Mechanism and protocol

On full Planaria's median-bandwidth RFA, 62.31% of nonself probabilities were
positive but below the former `1e-12` floor. Their summed probability mass across
all rows was about 1.57e-5. Small forward mass does not imply unimportant reverse
KL: that term uses log-probability differences.

Candidate mechanisms were declared before fitting: historical floored symmetric
KL, log-domain symmetric KL, global nonself conditional symmetric KL, and a
forward-only screen. Within each dataset/seed the graph and initial coordinates
were shared; gamma .5, learning rate .01, Riemannian norm cap2.5, burn-in30 and
row chunks256 were fixed. No labels or reference coordinates entered fitting.
No root pinning, added stress term, radius remapping or display force was used.
Fitting retained all comparison cells, not a batch-conditional submatrix.

The controls used already recorded input/graph settings, not newly selected
Planaria-only rules. Seed0 screens were followed by seeds1/2 for promising modes.
The outlier/blob control was added after the initial screen as a safety check,
not a blinded confirmation dataset. All failed/partial records were retained
locally; interrupted seeds were completed from durable checkpoints without a
second burn-in. The final audit covered 51 condition-level outcomes.

- ToggleSwitch: 200 cells, k15, sigma1.
- MyeloidProgenitors: 640 cells, k30, sigma2.
- Olsson: 382 cells, PCA20, k15, sigma median (about22.30).
- krumsiek11_blobs: 1280 cells, PCA of the11 available features, k30, sigma1.
- Planaria nested3000 and full21612: provided50 PCA scores, k15, sigma median
  (about7.464 and6.891 respectively); no extra PCA/standardization.
- Small/subset comparisons: 400 updates ×3 seeds. Full comparison: 300 updates.
  Full conditional seeds1/2 were recovered as200+100 updates.

Planaria subsets are not independent biological datasets. All datasets and
previous reference diagnostics were already known: this is transductive research
with development-selection bias, not unseen-dataset generalization or a blinded
biological holdout. Epoch limits are not convergence evidence.

## Fixed-budget full Planaria

Primary diagnostic: mutual-k15 geodesic graph, 128 anchors with seed714.
Local means15-NN overlap; global means Spearman of input graph geodesics versus
learned hyperbolic distances. These are not all of the paper's biological analyses
or its exact Qlocal/Qglobal implementation.

| Method | Global rank | Local overlap |
|---|---:|---:|
| Published coordinates, evaluated on the same graph/anchors | .7863 | .1156 |
| Historical floor, seed0 | .7025 | .4318 |
| Log-domain symmetric KL, seed0 | .7542 | .4307 |
| Conditional symmetric KL, seed0 | .7962 | .4115 |
| Conditional symmetric KL, seed1 | .8052 | .3979 |
| Conditional symmetric KL, seed2 | .8013 | .4073 |

A post-hoc sensitivity check changed evaluation graph to k20 and anchors to
seed716 without changing any fitted checkpoint. Reference global was .8003;
historical floor .6328; log-domain .7249; conditional .7562–.7581. Thus the
primary reference threshold is **not** simultaneously met on the other graph.
Conditional local overlap on this second graph was .3479–.3536 versus reference
.1172. Matching two primary numbers is not formal method equivalence.

Conditional full-data seed distance-rank agreement (128 anchors, seed41) was
.927–.965. This is promising but three seeds do not establish statistical
stability. Full-data1000-update persistence of this objective was not tested.
Visually, boundary concentration and incomplete central branching remain.

## Other data: mean of three seeds

The baseline in this table is the historical floored objective, not the corrected
self-inclusive default. Objective values themselves are not compared.

| Dataset | Global baseline → conditional | Local baseline → conditional |
|---|---:|---:|
| ToggleSwitch | .8384 → .8511 | .8651 → .8583 |
| MyeloidProgenitors | .9211 → .9223 | .8127 → .8252 |
| Olsson | .9112 → .9334 | .7040 → .6944 |
| krumsiek11_blobs | .5967 → .7137 | .5578 → .5425 |
| Planaria3000 | .6912 → .8041 | .5313 → .5477 |

All controls improved on this global diagnostic, but some sacrificed local
fidelity. Log-domain alone nearly preserved the three smaller controls; on
Planaria3000 global changed .6912→.7229 and local .5313→.5427. Forward-only
KL weakened global structure in the screen and was not promoted to full data.
Self-accessibility can be meaningful isolation information, so ignoring it is
not an unconditional improvement. Original RFA self-mass diagnostics remain.

## Historical long-run and landmark controls

An earlier scale-stability study used the former `1e-12` probability floor,
not the corrected self-inclusive default or conditional KL. On full Planaria,
1,000-update runs reached global rank .355–.414 and local overlap .470–.485.
For seed 0, global rank fell from .6939 at 300 updates to .4136 at 1,000 while
the best training loss continued to improve. This motivates trajectory checks,
but does **not** establish how either current objective behaves at 1,000 updates.

A separate input-graph landmark experiment used 128 farthest-point-sampled
landmarks, spatial-coordinate Adam and a geodesic stress term of weight 10 for
600 updates across three seeds. Global rank was .358–.370 despite seed
agreement of .980–.994. A further 100 updates for seed 0 with weight 100 and reset
Adam state reached .5060, still below the reference. Stability is not accuracy,
and increasing stress weight did not resolve the problem in these tests.
These are separate research conditions, not conditional-KL results.

Those scores used a k15 graph and 128 anchors with seed 41; reference global was
.7723. They must not be treated as the primary seed 714 ablation above. Seed 1/2
long runs were continued from 300-update raw checkpoints, with thread count
changing from 12 to 6 and stopping history reset. Their common 300-update runs
are the cleaner identical-condition seed comparison.

The archived raw-coordinate audit and source/harness snapshots remain under
ignored `data/scale_stability/`. No C. elegans 40k run was performed. Completing
full Planaria is not an atlas-scale or general biological-quality validation.

## Package integration check

At the published integration baseline (`494367a`), the package's self-inclusive
and conditional 400-update Planaria3000 runs each matched their archived
fixed-condition experiment with maximum raw-coordinate difference 0. The record
is kept locally in `data/publish_validation/production_parity.json`. This checks
package/prototype agreement under those conditions, not independent biological
validation or byte-identical reproduction in arbitrary environments.

## Reproduce comparisons on your own inputs

Install `pip install -e '.[torch,benchmark]'`. The driver takes numeric feature
columns and one label column; labels are removed before any PCA, graph or fit.
Neither code nor data from a reference implementation is downloaded/executed.

```bash
PYTHONPATH=src python examples/benchmark_objectives.py INPUT.csv \
  --n-neighbors 15 --sigma median --gamma .5 --epochs 400 \
  --seeds 0 1 2 --threads 4 --evaluation-k 15 20 \
  --include-legacy-floor --output data/objectives/report.json
```

Use `--pca 20` only when deliberately reproducing a PCA input configuration.
Use `--label-column NAME` for a label column other than the last column. The
optional historical reference is isolated in the benchmark, not a production
runtime override; it replicates this package's old numerical floor, **not the
Facebook implementation**. The driver saves raw NPZ embeddings and loss traces,
input/script hashes, initial-source provenance and fixed-budget diagnostics.
Initial coordinates are shared per seed. Results are never chosen by graph score.

For full21612 cells explicitly opt into, for example, `--max-dense-mib 16384`.
The default512MiB work-array guard is not disabled. This budget is not a process
RSS cap. The exact RFA remains dense/quadratic; two-worker study timings and
whole-process RSS high-water marks are not controlled speed comparisons.
Users must independently monitor machine memory and runtime for large inputs.

The public driver is a clean generic reproduction path, not a promise of
byte-identical output from the local recovered/parallel harness. Library/thread
versions, selected input columns/PCA, and evaluation configuration must match.
Published reference preprocessing and actual training budgets are still unverified.

## Provenance and next checks

Core source has no dataset-specific branches or hidden calibration. The study
supports general mechanisms, but future promotion requires independent data,
rare/isolation behavior, branch/pseudotime markers and long-run checks. Generic
probability correction and experimental objective redesign are separate changes.

The method mathematics follows [Klimovskaia et al. (2020)](https://www.nature.com/articles/s41467-020-16822-4).
Only [published coordinate artifacts](https://github.com/facebookresearch/PoincareMaps/tree/main/results)
were used for reference evaluation. Their source/license is separate from this
package; no reference implementation, CSV, original figure, or generated image
is vendored here. This is an independent implementation, not a new strict
clean-room provenance claim. Inputs, coordinate artifacts and local logs remain
ignored, rather than silently relicensed as MIT.
