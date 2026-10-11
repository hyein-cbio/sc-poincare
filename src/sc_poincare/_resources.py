"""Conservative allocation guard for the exact, dense RFA workflow."""

from __future__ import annotations

import operator

DEFAULT_MAX_DENSE_BYTES = 512 * 1024**2


def check_dense_budget(n_obs: int, max_dense_bytes: int | None) -> None:
    """Budget four float64 N×N working arrays, not total process RSS.

    Sparse factorization fill-in, features, library overhead and autograd
    can use additional memory. Disabling this guard is an explicit opt-in,
    not a claim of large-N support.
    """
    if max_dense_bytes is None:
        return
    try:
        limit = operator.index(max_dense_bytes)
    except TypeError:
        raise ValueError("max_dense_bytes must be a positive integer or None.") from None
    if isinstance(max_dense_bytes, bool) or limit <= 0:
        raise ValueError("max_dense_bytes must be a positive integer or None.")
    estimated = 4 * 8 * int(n_obs)**2
    if estimated > limit:
        raise MemoryError(
            f"Exact dense RFA for {n_obs:,} cells budgets approximately "
            f"{estimated / 1024**3:.2f} GiB of dense working arrays "
            f"(limit {limit / 1024**3:.2f} GiB), excluding other memory. "
            "Subsample the cells or explicitly increase max_dense_bytes. "
            "Row-chunked fitting does not remove RFA's O(N²) storage cost."
        )
