"""SAHI-style slicing for small objects (after the baseline exists).

A frame of size (H, W) is cut into overlapping tiles of size (t, t) with overlap ratio o.
Boxes must be re-expressed in each tile's coordinates and clipped to the tile;
boxes that keep less than a minimum fraction of their area after clipping are dropped.
"""

from __future__ import annotations


def tile_origins(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """Top-left (x0, y0) of every tile, covering the whole frame including the edges."""
    # TODO: Implement. Edge case: the last tile in each row/column must end exactly at
    # the frame border, not past it.
    raise NotImplementedError


def boxes_in_tile(boxes_xyxy, x0: int, y0: int, tile: int, min_visible: float):
    """boxes_xyxy: array (K, 4) in frame pixels -> array (K', 4) in tile pixels."""
    # TODO: Implement: shift, clip, filter by visible-area fraction.
    raise NotImplementedError
