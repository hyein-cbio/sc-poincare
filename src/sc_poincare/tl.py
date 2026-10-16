"""Scanpy-style tools for Poincaré-disk embeddings."""

from __future__ import annotations

from sc_poincare._fit_progress import PoincareProgress
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
    "PoincareProgress",
    "poincare_map",
    "poincare_recenter",
    "poincare_pseudotime",
    "poincare_lineages",
]

# Deprecated display-only API: delete this isolated block and _legacy_display.py
# when retiring the feature. Fitting and core geometry do not import that module.
from sc_poincare._legacy_display import (
    fan_branches as poincare_fan,
    jitter_embedding as poincare_jitter,
    repel_embedding as poincare_repel,
)

__all__ += ["poincare_fan", "poincare_jitter", "poincare_repel"]
