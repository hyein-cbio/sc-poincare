"""Scanpy-style plotting for Poincaré-disk embeddings."""

from __future__ import annotations

from sc_poincare._plotting import draw_disk_embedding

poincare_map = draw_disk_embedding

__all__ = ["poincare_map"]
