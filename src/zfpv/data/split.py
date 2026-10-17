"""Video-level split from config/splits/video_split_manifest.csv (one source of truth).

columns: video, n_frames, n_annotated, is_night, split
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"video", "n_frames", "n_annotated", "is_night", "split"}


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def videos_for(df: pd.DataFrame, split: str) -> list[str]:
    """Names of the videos assigned to `split` ('train', 'val' or 'test')"""
    videos = df.loc[df["split"] == split, "video"].tolist()
    if not videos:
        raise ValueError(f"No videos in split {split!r}")
    return videos
