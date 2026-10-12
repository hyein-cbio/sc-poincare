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
