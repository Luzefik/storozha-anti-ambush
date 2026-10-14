"""Video -> frames. Frame i on disk == frame i in CVAT, or labels land on the wrong image.

Sequential read() only: seeking (CAP_PROP_POS_FRAMES) on H.264 can land on a neighbour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2


def extract_video_frames(
    video_path: Path, out_dir: Path, ext: str = "png", expected_n: int | None = None
) -> int:
    """Write every frame to out_dir/frame_{i:06d}.{ext}; return the count."""
    out_dir = Path(out_dir)
    # expected_n comes from COCO: already extracted -> skip
    if expected_n is not None and out_dir.is_dir():
        if len(list(out_dir.glob(f"frame_*.{ext}"))) == expected_n:
            return expected_n

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video {video_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if ext == "png" else []  # fast, ~0.6 MB
        n = 0
        while True:
            ok, frame = cap.read()  # frame: (H, W, 3) uint8 BGR, imwrite expects BGR
            if not ok:
                break
            path = out_dir / f"frame_{n:06d}.{ext}"
            tmp = out_dir / f".partial_{path.name}"  # interrupted run -> no half-written frame
            if not cv2.imwrite(str(tmp), frame, params):
                raise OSError(f"failed to write {path}")
            os.replace(tmp, path)
            n += 1
    finally:
        cap.release()
    return n


def check_frame_count(n_written: int, coco_json_path: Path) -> None:
    """Raise if n_written != len(coco["images"]): a mismatch shifts every label."""
    coco = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))
    n_coco_images = len(coco["images"])
    if n_written != n_coco_images:
        raise ValueError(
            f"{coco_json_path}: extracted {n_written} frames, but the COCO file lists "
            f"{n_coco_images} images. Labels would be misaligned; fix extraction first."
        )
