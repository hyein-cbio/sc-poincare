# Graph topology experiment: results and software tests

This document preserves the initial two-subset probe. The subsequent
[display-visibility extension](graph-visibility-study.md) adds four cases,
separates screen overlap from structure preservation, and records a failed
stress-case preparation. The initial neighborhood findings are not a rejection
of display usability as a separate endpoint.

## Decision

**Keep mutual as the default.** Union is available as an explicit experimental
option, not an unconditional improvement. All six paired comparisons show
lower primary k15 input-neighbor recall with union, despite better global
feature-distance rank and removal of bridge dependence. Paul k15 macro purity
also falls in every seed; k30 gives different population-level trade-offs.

## What changed

`graph_mode='union'` keeps either directed kNN choice; default `'mutual'` keeps
reciprocal choices. Both use the same component-MST stitcher, actual Gaussian
weights, RFA formula, self-inclusive symmetric KL and Riemannian fitter. Mode
and experimental status are persisted in AnnData metadata. No optimizer,
objective, clipping, display transform or hidden dataset-specific rule changed.

## Completed fixed protocol

- Two known development subsets, each 384 cells, uniform sample seed 714,
  retaining original CSV row order. Neither is an independent holdout.
- Moignard: supplied 42 features; Paul: labels removed before subset full-SVD
  PCA20. Fitted matrices exactly match the prior diagnostic subsets.
- k15 graph; sigma resolved once on mutual positive edges and reused numerically
  in union. Mutual spectral initialization is generated per seed, then the
  exact same array is passed into both fits, with hashes checked.
- Seeds 0/1/2; 500 updates; gamma 2; learning rate .01; burn-in 5;
  patience 500; Riemannian max_grad_norm 2.5; exact row chunks 128; CPU float64.
- Local evaluation k15 primary / k30 sensitivity; fixed 128 global anchors
  with seed 714. All global-anchor correlations were defined.
- Seed 0 was fixed in advance for figures, not selected by quality.
- One thread, default 512 MiB dense preflight, 600-second whole-run timer. **12/12 fits completed in 25.87 seconds**, with no failed or partial fits.
- Every fit reached its 500-update epoch limit and its own best checkpoint was
  epoch 500. This is not evidence of convergence. Raw loss values refer to
  different targets and are not a method-accuracy comparison.
- These are new 500-update paired fits, not a comparison of a prior 80-update
  preparation plot with a 500-update candidate.

See the [frozen experiment protocol](algorithm-experiment-01.md) and
`examples/benchmark_graph_modes.py` for the reusable runner.

## Input identity

| Dataset | Original CSV SHA-256 |
|---|---|
| Moignard2015 | `3f45057370d568f87b0ab313bfa91992d76e8551824e3a545014b8d4a47c2ea1` |
| Paul | `fed836ed28f148d9ee8efd8ec13ebd468f06ef9b92eeb6f9e218235d4ad2f21d` |

No cell matrices, IDs, reference coordinates or source snapshots are included
in this report. Original input licenses apply separately from the package MIT
license. Local shared input artifacts retain exact feature/label/ID alignment.

## Graph diagnostics

Counts are before Gaussian weighting; edge counts are undirected after stitching.

| Dataset | Mode | Components before | Isolated before | Bridges | Edges after | Shared sigma |
|---|---|---:|---:|---:|---:|---:|
| Moignard2015 | mutual | 13 | 11 | 12 | 1628 | 21.446833301 |
| Moignard2015 | union | 1 | 0 | 0 | 4144 | 21.446833301 |
| Paul | mutual | 40 | 34 | 39 | 1331 | 4.972634258 |
| Paul | union | 1 | 0 | 0 | 4468 | 4.972634258 |

Fewer components/bridges do not guarantee a better target. Union also changes
degree and total affinity; no degree renormalization was inserted.

## Aggregate results: ranges across the three paired seeds

Ranges are not confidence intervals. Delta always means union minus mutual.

| Dataset | K | Metric | Mutual | Union | Paired delta |
|---|---:|---|---|---|---|
| Moignard2015 | 15 | overall_purity | 0.527083 to 0.531597 | 0.526389 to 0.530556 | -0.003993 to +0.002083 |
| Moignard2015 | 15 | macro_purity | 0.491089 to 0.496239 | 0.489709 to 0.494449 | -0.004837 to +0.002025 |
| Moignard2015 | 15 | input_neighbor_recall | 0.447222 to 0.451563 | 0.355035 to 0.359028 | -0.096354 to -0.088194 |
| Moignard2015 | 30 | overall_purity | 0.510851 to 0.511372 | 0.514583 to 0.514931 | +0.003559 to +0.003819 |
| Moignard2015 | 30 | macro_purity | 0.473205 to 0.473645 | 0.474922 to 0.475247 | +0.001602 to +0.001877 |
| Moignard2015 | 30 | input_neighbor_recall | 0.580903 to 0.582205 | 0.523003 to 0.525087 | -0.058854 to -0.056337 |
| Moignard2015 | — | Global feature-distance rank | 0.718612 to 0.718721 | 0.743486 to 0.743714 | +0.024775 to +0.025102 |
| Paul | 15 | overall_purity | 0.476562 to 0.477431 | 0.469618 to 0.471528 | -0.007118 to -0.005903 |
| Paul | 15 | macro_purity | 0.314778 to 0.315955 | 0.312975 to 0.314481 | -0.002979 to -0.000297 |
| Paul | 15 | input_neighbor_recall | 0.455556 to 0.456597 | 0.412326 to 0.413542 | -0.043924 to -0.042014 |
| Paul | 30 | overall_purity | 0.427431 to 0.428385 | 0.435069 to 0.436806 | +0.007292 to +0.008420 |
| Paul | 30 | macro_purity | 0.268863 to 0.270156 | 0.280155 to 0.281869 | +0.010827 to +0.011713 |
| Paul | 30 | input_neighbor_recall | 0.568142 to 0.569705 | 0.561285 to 0.564323 | -0.008420 to -0.004080 |
| Paul | — | Global feature-distance rank | 0.823541 to 0.825521 | 0.846788 to 0.848427 | +0.021267 to +0.024404 |

## All paired seeds: primary and global deltas

| Dataset | Seed | K15 input recall delta | K15 macro purity delta | K30 input recall delta | Global rank delta |
|---|---:|---:|---:|---:|---:|
| Moignard2015 | 0 | -0.094271 | -0.004837 | -0.056337 | +0.024777 |
| Moignard2015 | 1 | -0.088194 | +0.002025 | -0.057118 | +0.024775 |
| Moignard2015 | 2 | -0.096354 | -0.001790 | -0.058854 | +0.025102 |
| Paul | 0 | -0.043750 | -0.002979 | -0.004948 | +0.021267 |
| Paul | 1 | -0.043924 | -0.001393 | -0.004080 | +0.023194 |
| Paul | 2 | -0.042014 | -0.000297 | -0.008420 | +0.024404 |

## Population-level results, including losses

Every observed population is retained below. Entries are paired delta ranges
across the same three seeds. Purity is not adjusted by its size ceiling.
For a population of m cells, its ceiling is min(m-1,k)/k; check the counts,
fine-label granularity and legitimate continuous differentiation before
interpreting mixing as biological damage. The Paul subset omits 11DC and root;
it cannot establish their preservation or characterize every rare population.

Notable k15 losses include Moignard PS purity and recall, and Paul 16Neu/6Ery
purity. Paul 18Eos purity is unchanged while its input-neighbor recall falls;
unchanged label purity therefore does not imply unchanged input structure.

### Moignard2015

| Population | Cells | K15 purity delta | K15 input recall delta | K30 purity delta | K30 input recall delta |
|---|---:|---|---|---|---|
| 4SFG | 95 | -0.007719 to +0.000702 | -0.078596 to -0.060351 | +0.019649 to +0.021053 | -0.042456 to -0.034386 |
| 4SG | 69 | +0.037681 to +0.044444 | -0.057005 to -0.052174 | +0.022222 to +0.022222 | -0.006280 to -0.004348 |
| HF | 103 | -0.001942 to +0.003883 | -0.080906 to -0.073139 | -0.000647 to +0.000000 | -0.055663 to -0.053074 |
| NP | 54 | -0.009877 to +0.008642 | -0.130864 to -0.114815 | -0.012346 to -0.011111 | -0.098765 to -0.088272 |
| PS | 63 | -0.050794 to -0.039153 | -0.187302 to -0.166138 | -0.022751 to -0.021164 | -0.119577 to -0.111111 |

### Paul

| Population | Cells | K15 purity delta | K15 input recall delta | K30 purity delta | K30 input recall delta |
|---|---:|---|---|---|---|
| 10GMP | 21 | -0.088889 to -0.066667 | +0.073016 to +0.082540 | -0.022222 to +0.001587 | +0.131746 to +0.139683 |
| 12Baso | 12 | +0.161111 to +0.166667 | -0.088889 to -0.022222 | +0.066667 to +0.072222 | +0.019444 to +0.036111 |
| 13Baso | 44 | +0.115152 to +0.119697 | +0.077273 to +0.086364 | +0.118939 to +0.123485 | +0.062121 to +0.064394 |
| 14Mo | 60 | +0.068889 to +0.072222 | -0.014444 to -0.010000 | +0.053889 to +0.055556 | -0.030000 to -0.025556 |
| 15Mo | 19 | +0.059649 to +0.084211 | -0.108772 to -0.094737 | +0.021053 to +0.028070 | -0.047368 to -0.045614 |
| 16Neu | 22 | -0.184848 to -0.178788 | -0.239394 to -0.221212 | -0.013636 to -0.009091 | -0.012121 to -0.009091 |
| 17Neu | 2 | +0.000000 to +0.033333 | -0.066667 to +0.000000 | +0.016667 to +0.016667 | +0.033333 to +0.050000 |
| 18Eos | 3 | +0.000000 to +0.000000 | -0.155556 to -0.066667 | +0.000000 to +0.000000 | -0.133333 to -0.066667 |
| 19Lymph | 4 | +0.000000 to +0.000000 | +0.033333 to +0.066667 | +0.000000 to +0.000000 | +0.108333 to +0.133333 |
| 1Ery | 4 | +0.066667 to +0.100000 | +0.383333 to +0.433333 | +0.033333 to +0.041667 | +0.116667 to +0.158333 |
| 2Ery | 55 | -0.061818 to -0.042424 | -0.149091 to -0.136970 | -0.124242 to -0.112121 | -0.161212 to -0.153333 |
| 3Ery | 35 | -0.030476 to -0.019048 | -0.118095 to -0.102857 | +0.039048 to +0.047619 | -0.037143 to -0.024762 |
| 4Ery | 14 | -0.009524 to -0.000000 | -0.147619 to -0.142857 | +0.014286 to +0.014286 | -0.030952 to -0.026190 |
| 5Ery | 22 | -0.106061 to -0.103030 | -0.151515 to -0.142424 | -0.101515 to -0.095455 | -0.115152 to -0.112121 |
| 6Ery | 27 | -0.192593 to -0.185185 | +0.051852 to +0.051852 | -0.041975 to -0.037037 | +0.071605 to +0.077778 |
| 7MEP | 17 | +0.043137 to +0.050980 | -0.043137 to -0.039216 | +0.041176 to +0.066667 | +0.031373 to +0.062745 |
| 8Mk | 13 | +0.046154 to +0.046154 | +0.148718 to +0.158974 | +0.028205 to +0.028205 | +0.220513 to +0.223077 |
| 9GMP | 10 | -0.026667 to +0.000000 | -0.066667 to -0.033333 | +0.010000 to +0.026667 | +0.036667 to +0.056667 |

## Software validation

- Full local suite: **268 passed**, two existing exact-zero-target warnings.
- Separate Torch-free installed-wheel environment: **110 passed** on geometry,
  graph modes, evaluation and comparison tests. Wheel import/union smoke passed
  outside the source tree. See the [Actions workflow](../.github/workflows/tests.yml).
- New tests cover independent dense directed-kNN topology for both metrics,
  reverse-only edges, disconnected union MST weights/underflow, duplicate zero
  edges, capped k, invalid-mode rejection, AnnData copy/H5AD persistence, shared
  sigma/initialization execution, timeout/failure records and timer restoration.
  A late-timeout bookkeeping race was fixed after review, with regression tests
  ensuring terminal status updates occur only after the timer is disarmed.
- Default mutual graph/CSR/RFA exactly match the pre-change source on both
  measured subsets; a synthetic 80-update default fit matches raw coordinates
  and loss exactly. The fitter source is unchanged.
- `git diff --check`: passed. A post-run timer-bookkeeping fix affects terminal-status
  handling only; numerical sources and the reported experiment outputs are unchanged.

## Limitations and next decision

Three seeds perturb the same mutual spectral layout; they do not test arbitrary
initialization robustness. This probe fixes a 500-update budget, not equal
convergence. Another initialization or longer schedule could change trade-offs,
but neither was searched here. Direct feature-distance rank does not validate
developmental ordering, root pseudotime, marker continuity or biological accuracy.
No claim is made about full Moignard/Paul, independent data or full Planaria.

**Do not promote union as the default.** Keep the option for explicit research.
First explain degree/affinity, local/global and population trade-offs; a new
normalization, hybrid graph, optimizer or longer-run comparison needs a separate
fixed protocol rather than quietly changing this experiment.

The original local artifacts used `data/graph_diagnostics_integration/`.
That directory is intentionally reused by later studies rather than archived
per run; the numerical tables above preserve this historical probe. See the
[current MU study](mu-scaleup-study.md) for the latest artifact layout.
