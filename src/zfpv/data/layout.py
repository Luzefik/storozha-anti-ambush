"""Train/val/test layout for the trainer, from the manifest, frames not copied.

yolo_dir/
    images/<video>      relative symlink -> frames_dir/<video>
    labels/<video>/     one .txt per frame (coco_to_yolo)
    train.txt ...       "./images/<video>/<frame>" per line
    data.yaml

symlink: Ultralytics swaps "/images/" -> "/labels/" to find a label; without it every
frame counts as background. relative: survives moving the repo. lists: a new split or
leave-one-video-out fold = new .txt files, no moving frames.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def link_video_frames(frames_dir: Path, yolo_dir: Path, video: str) -> Path:
    """Create yolo_dir/images/<video> -> frames_dir/<video> (relative symlink)."""
    link = yolo_dir / "images" / video
    link.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(frames_dir / video, link.parent)
    if link.is_symlink():
        if os.readlink(link) == target:
            return link
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink; remove it first")
    link.symlink_to(target, target_is_directory=True)
    return link


def write_split_lists(
    manifest: pd.DataFrame,
    frames_dir: Path,
    yolo_dir: Path,
    ext: str,
    class_names: list[str],
    splits: list[str],
) -> dict[str, int]:
    """Write <split>.txt + data.yaml; return frames per split.

    Videos with no frames on disk (after --only) are skipped with a warning.
    """
    counts = {}
    for split in splits:
        lines = []
        for video in manifest.loc[manifest["split"] == split, "video"]:
            frames = sorted((frames_dir / video).glob(f"frame_*.{ext}"))
            if not frames:
                logger.warning("%s: no frames in %s, left out of %s", video, frames_dir, split)
                continue
            labels_dir = yolo_dir / "labels" / video
            missing = [f.name for f in frames if not (labels_dir / f"{f.stem}.txt").exists()]
            if missing:
                raise FileNotFoundError(
                    f"{video}: {len(missing)} frames without a label file, e.g. {missing[0]}"
                )
            link_video_frames(frames_dir, yolo_dir, video)
            lines += [f"./images/{video}/{f.name}" for f in frames]
        (yolo_dir / f"{split}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        counts[split] = len(lines)

    # No "path:" key on purpose: Ultralytics then takes the folder of data.yaml as the
    # dataset root, so the file works on any machine and from any working directory.
    # A relative "path:" would be read relative to the cwd, an absolute one only here.
    data_yaml = {
        **{split: f"{split}.txt" for split in splits},
        "names": dict(enumerate(class_names)),
    }
    (yolo_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return counts
