# Spectral versus random initialization

This comparison uses the existing initializer. It does not add the `tol` /
`maxiter` controls that already exist on `main`, and it does not change the
default. ToggleSwitch has 200 cells and two numeric features. The label column
was removed before the graph and the fit. Settings match the small-data
protocol: k=15, sigma=1, gamma=2, learning rate 0.01, burn-in 30, 200 updates,
seeds 0–2. The input SHA256 is
`25769c7044c1411a043ce00c41177aaf1e2091a950106caaa5976d7649e94fc1`.

Distance rank is the Spearman correlation of pairwise feature distances and
hyperbolic distances. The returned coordinates are the lowest training loss.
An epoch limit is not convergence. Three seeds are not a confidence interval.

| Initialization | Seed | Initial distance rank | Final distance rank | Best loss | Init seconds | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| spectral | 0 | 0.8011 | 0.7170 | 354.648 | 0.0053 | 0.47 |
| spectral | 1 | 0.8009 | 0.7170 | 354.650 | 0.0033 | 0.46 |
| spectral | 2 | 0.8011 | 0.7170 | 354.624 | 0.0032 | 0.45 |
| random | 0 | 0.0142 | 0.6986 | 446.447 | 0.0001 | 0.45 |
| random | 1 | -0.0084 | 0.6632 | 613.574 | 0.0001 | 0.46 |
| random | 2 | -0.0063 | 0.7239 | 593.463 | 0.0001 | 0.45 |

Random initialization is faster and does not preserve feature distances.
Spectral initialization is stable across these seeds and reaches a lower
training loss and a tighter final distance rank. That is a quality difference,
not a reason to judge the initializer by speed alone. The exact values are in
[spectral-init-quality.json](spectral-init-quality.json). No coordinates or
labels are stored there.
