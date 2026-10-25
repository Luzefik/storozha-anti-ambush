"""Common interface for every model in the comparison (YOLO, DETR-family, ...).

One adapter per library lives next to this file (e.g. yolo_adapter.py) and implements
`Detector`. Library-specific resize/letterbox/NMS stays inside the adapter; what comes
out is always boxes in ORIGINAL-frame pixel coordinates.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    name: str

    def predict(self, frames: np.ndarray) -> list[np.ndarray]:
        """
        frames:  (N, H, W, 3) uint8. Decide and document: RGB or BGR?
        returns: list of N arrays, each (K_i, 5) = [x1, y1, x2, y2, score],
                 in original-frame pixel coordinates.
        """
        # TODO: Implement in each adapter: library-specific preprocessing, inference,
        # NMS if the model needs it, and mapping boxes back to original coordinates.
        # Most silent bugs live in that back-mapping: print one box before/after.
        ...
