"""Scanpy-style tools for Poincaré-disk embeddings."""

from __future__ import annotations

from sc_poincare._anndata import (
    fan_branches,
    infer_lineages,
    jitter_embedding,
    layout_cells,
    move_root_to_origin,
    repel_embedding,
    score_pseudotime,
)

poincare_map = layout_cells
poincare_recenter = move_root_to_origin
poincare_pseudotime = score_pseudotime
poincare_lineages = infer_lineages
poincare_fan = fan_branches
poincare_repel = repel_embedding
poincare_jitter = jitter_embedding

__all__ = [
    "poincare_map",
    "poincare_recenter",
    "poincare_pseudotime",
    "poincare_lineages",
    "poincare_fan",
    "poincare_repel",
    "poincare_jitter",
]
