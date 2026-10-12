# Probability objectives and numerical policy

The default remains the row-wise symmetric KL objective between dense relative
forest accessibility (RFA) and hyperbolic-distance softmax probabilities:

```
log Q_ij = log_softmax(-d_H(x_i, x_j) / gamma)
loss = sum_ij P_ij (log P_ij - log Q_ij) + Q_ij (log Q_ij - log P_ij)
```

All comparison cells, including self, are retained. Row chunks reduce workspace,
not the set of comparison columns. Gradients accumulate before one global update.
No label, published coordinates or display forces enter fitting.

## Log-domain calculation

Positive target probabilities are no longer clipped to `1e-12`. That clipping
could flatten log-probability distinctions even when their total forward mass
was small; the reverse term depends on them. Log-softmax preserves finite
prediction logs even when exponentiating a very negative log underflows to zero.
Positive subnormal targets are retained, too. Both Torch and NumPy evaluate the
same formula; the NumPy fallback still uses finite differences and is tiny-data only.

**Exact zero policy:** the target is row-normalized without adding probability
mass. For logarithms of exact zero only, the dtype's smallest normal positive
value is substituted. Thus the reverse KL is finitely regularized when a target
contains zeros; mathematically it would be infinite if Q assigns positive mass
there. Fitting warns about this condition. It does not claim an exact finite KL
for zero-support targets or silently make their probabilities `1e-12`.

The numerical change can change losses, stopping epochs and coordinates on the
same inputs/seed. Historical results from before this correction are not exactly
reproducible by merely rerunning old settings on the new version. Loss values
from different numerical policies are not interchangeable convergence evidence.
The change alone does not guarantee good global or biological structure.

## Experimental self-excluding conditional objective

Explicitly select `objective="conditional_symmetric_kl"` in
`sc_poincare.tl.poincare_map` or low-level `fit_disk_embedding`:

```
P'_ij = P_ij / sum_{k != i} P_ik     (j != i)
Q'_ij = exp(-d_H(x_i,x_j)/gamma) / sum_{k != i} exp(-d_H(x_i,x_k)/gamma)
P'_ii = Q'_ii = 0
```

The symmetric KL is evaluated between P' and Q'. All N-1 nonself columns
remain, including when gradients use row chunks. This is not a batch-submatrix
conditional loss. NumPy evaluation/finite differences use the same definition;
pass the same `objective` to `evaluate_alignment` when recomputing its loss.
Rows without positive off-diagonal target mass are rejected, not invented.

**This changes the model, not just floating-point calculation.** Self-accessibility
can contain meaningful isolation/degree information. Excluding it may reduce
self-probability/volume pressure, but is not proven to solve all boundary crowding.
Some data may sacrifice local fidelity, or need the original isolation semantics.
The original RFA self-mass diagnostics remain in AnnData, even in this mode.
No radius cap, root pinning, stress term or display transform is added.

AnnData records `objective`, `objective_experimental`, `probability_computation`
and `zero_target_policy` under the selected storage prefix's `params`. Low-level
`FitReport` records the method and policy too. Changing objectives changes the
loss scale; loss numbers or patience behavior cannot be compared as if the
objective were unchanged. A fixed epoch limit is not convergence evidence.
The public default remains `objective="symmetric_kl"`.
