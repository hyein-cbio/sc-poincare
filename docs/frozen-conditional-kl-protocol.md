# Frozen protocol: corrected KL versus conditional KL

This protocol was written before any new fit. It does not authorize execution
on the machine that recorded it, and it does not change the public default.
The default remains corrected, self-inclusive `symmetric_kl`.

## Starting point

Use the [long-run checklist](benchmarking.md#long-run-preparation-and-recovery)
before any later execution. The 2026-10-19 record is:

| Item | Value |
|---|---|
| Comparison base | `feat/experimental-graph-stress` at `9f369a9`, plus the driver change that adds feature-distance Spearman |
| Import | editable `sc-poincare` 0.1.0 from `/workspace`; `sc_poincare.__file__` is `/workspace/src/sc_poincare/__init__.py` |
| Python | 3.12.3 |
| NumPy / SciPy / scikit-learn | 2.5.4 / 1.18.1 / 1.9.1 |
| Torch / Scanpy | 2.14.1+cpu / 1.12.4 |
| Tests at freeze | `python3 -m pytest tests`: 211 passed |
| Artifact root | ignored `data/`; it was not present |
| Host memory | 15 GiB RAM, about 4.8 GiB available, no swap |

## Resource gate

Public `Planaria.csv` from PoincaréMaps `7ffa518` was measured on this host.
SHA256 `d59a30b1d2ddf959912d4a1fc4c20910de88823f2a67223d9e1809db065e382d`,
12,133,109 bytes, 21,612 data rows, 50 numeric columns plus a label. One
float64 N-by-N matrix is 3.48 GiB. The default guard budgets four such
arrays, 13.92 GiB, and the limit is 0.50 GiB. `check_dense_budget(21612)`
raises `MemoryError` before any matrix is allocated. Available memory was
about 4.2 GiB with no swap. The historical peak RSS was about 5.10 GiB.

Do not raise `--max-dense-mib` on this host to start the fit. A smaller
dataset, a shorter budget, or a disabled guard is not a substitute result.

## Question and inputs

On full Planaria (21,612 cells, the previously used 50 PCA scores, no extra
PCA or standardization), do 1,000 updates of corrected self-inclusive
`symmetric_kl` and of `conditional_symmetric_kl` still improve the declared
graph diagnostics while training loss decreases?

- Labels, roots and published coordinates stay out of graph construction,
  initialization and fitting. They may be used only for the predeclared
  evaluation below.
- Hash the input file before fitting and store that hash in the output JSON.
- Keep the input under ignored `data/`. Do not vendor it.
- Graph: k=15. Sigma: median positive edge length. Gamma 0.5. Learning rate
  0.01. Riemannian gradient cap 2.5. Burn-in 30. Row chunk 256.
- Both objectives share initial coordinates within each seed.
- The decision baseline is corrected self-inclusive `symmetric_kl`, not the
  historical `1e-12` floor. `--include-legacy-floor` may be added only as a
  labeled diagnostic.

## Selection and seeds

Return the lowest training objective, including the initialization. Do not
select by an evaluation score. Preserve raw current and best coordinates,
loss, `gradient_norm_trace`, radius quantiles and stop reason. Each trace
entry is the max per-point Riemannian gradient norm at the checkpoint
before that update.

Seeds are 0, 1, 2, 3 and 4. The five-seed range is a sensitivity description,
not a confidence interval and not a biological replicate.

## Metrics declared before fitting

Primary, mutual k=15 graph, 128 anchors, anchor seed 714:

- global Spearman of graph-geodesic distances versus learned hyperbolic
  distances;
- 15-neighbor overlap with those graph neighbors.

Sensitivity, not used for selection:

- the same scores at evaluation k=20 and k=30;
- the same scores with anchor seed 716;
- `feature.global_distance_rank`: Spearman of direct feature distances versus
  learned hyperbolic distances on the same anchors.

The public driver records the feature-distance rank and accepts
`--evaluation-k 15 20 30`. Alternative anchors are a second invocation with
`--anchor-seed 716` and a new output prefix. Losses are not compared across
objectives.

## Biological checks that must be attached before a run

Before execution, attach a marker, branch and root list taken from a source
that was fixed without looking at the new coordinates. If that list is not in
the input manifest, do not start. Graph scores do not fill that gap. Report
every rare and outlier population that the list names. A population loss
remains a result.

## Decision

Explain global, local, biological and seed-to-seed trade-offs under this
protocol. Reaching 1,000 updates is not sufficient to change the default.
Conditional KL stays experimental unless a later decision says otherwise.

## Command shape, after the resource and marker gates pass

```bash
PYTHONPATH=src python examples/benchmark_objectives.py INPUT.csv \
  --n-neighbors 15 --sigma median --gamma .5 --epochs 1000 \
  --learning-rate .01 --burn-in 30 --batch-size 256 \
  --seeds 0 1 2 3 4 --threads 4 \
  --evaluation-k 15 20 30 --anchor-seed 714 --anchors 128 \
  --max-dense-mib 16384 \
  --output data/objectives/planaria-full-1000.json
```

Repeat with `--anchor-seed 716` and a different `--output` prefix. Do not
overwrite an existing output. Confirm the process import path and Git
revision again immediately before the first fit.
