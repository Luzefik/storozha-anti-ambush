#!/usr/bin/env python
"""raw videos + COCO -> frames (data/interim) -> YOLO labels + split lists (data/processed).

Thin wrapper, logic is in zfpv.data.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import zipfile
from pathlib import Path

from zfpv.config import load_config
from zfpv.data.coco_to_yolo import coco_json_to_yolo
from zfpv.data.frames import check_frame_count, extract_video_frames
from zfpv.data.layout import write_split_lists
from zfpv.data.split import load_manifest

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="process only these videos (e.g. video-11)")
    parser.add_argument("--override", nargs="*", default=[], help="config overrides a.b=c")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(overrides=args.override)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    frames_root = Path(cfg["paths"]["frames_dir"])
    yolo_dir = Path(cfg["paths"]["yolo_dir"])
    labels_root = yolo_dir / "labels"  # next to images/, Ultralytics looks there
    ext = cfg["frames"]["ext"]
    manifest = load_manifest(Path(cfg["paths"]["split_manifest"]))

    videos = args.only or manifest["video"].tolist()
    for video in videos:
        label_zip = raw_dir / "labels" / f"spie-{video}.zip"
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(label_zip) as zf:
                zf.extract("annotations/instances_default.json", tmp)
            coco_json = Path(tmp) / "annotations" / "instances_default.json"

            # frame count from COCO, not container metadata
            n_expected = len(json.loads(coco_json.read_text(encoding="utf-8"))["images"])
            n = extract_video_frames(
                raw_dir / "videos" / f"{video}.mp4", frames_root / video, ext, n_expected
            )
            check_frame_count(n, coco_json)
            coco_json_to_yolo(coco_json, labels_root / video)
        logger.info("%s: %d frames", video, n)

    # split lists + symlinks, frames not copied
    counts = write_split_lists(
        manifest, frames_root, yolo_dir, ext, cfg["dataset"]["class_names"], cfg["split"]["names"]
    )
    logger.info("split lists in %s: %s", yolo_dir, counts)


if __name__ == "__main__":
    main()
