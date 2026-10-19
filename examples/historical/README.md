# HISTORICAL ONLY — selected-condition reproduction

These examples are retained for inspection/reproduction of old research, not
recommended current pipelines, package defaults or evidence of general quality.
The current API is `sc_poincare.tl.poincare_map` with explicit MU/RFA selection.

- `experimental_geodesic_fit.py`: old RFA-KL + absolute geodesic stress with
  spatial-coordinate Adam; it is not current tangent-Adam MU.
- `run_selected_conditions.py`: dataset-name-specific selected recipes, confined
  to the historical recipe block. It requires `--acknowledge-historical`.
- `tests/test_experimental_geodesic.py`: dedicated historical formula/chunk/input
  tests; run explicitly with `python -m pytest examples/historical/tests`. CI has
  a separate historical step; default `pytest tests` does not discover this tree.

Example of deliberately opting into a new active-input reproduction:

```sh
python examples/historical/run_selected_conditions.py --acknowledge-historical \
  MyeloidProgenitors path/to/active-input.csv --seed 0 --output data/new-reproduction/result.npz
```

Closed-research/capsule paths are rejected even with that acknowledgement.
Read archived results instead of restarting queues. A separately authorized
study needs an explicit protocol, checked inputs and a new output directory.
No automatic legacy redirect remains at the former `examples/` paths.

See [historical provenance](../../docs/reference-parity.md) and
[containment policy](../../docs/research-containment.md). Old recorded numbers
belong to their frozen source/numerical environment; running today's dependencies
is not a promise to reproduce every pre-log-domain result exactly.
