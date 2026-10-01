"""Scanpy-style tools for Poincaré-disk embeddings."""

from __future__ import annotations

from sc_poincare._anndata import (
    infer_lineages,
    layout_cells,
    move_root_to_origin,
    score_pseudotime,
)

poincare_map = layout_cells
poincare_recenter = move_root_to_origin
poincare_pseudotime = score_pseudotime
poincare_lineages = infer_lineages

__all__ = [
    "poincare_map",
    "poincare_recenter",
    "poincare_pseudotime",
    "poincare_lineages",
]
