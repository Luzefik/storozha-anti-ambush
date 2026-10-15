"""Dataset audit: tables -> results/audit/, plots -> results/figures/.

size = sqrt(w * h) px. lb = after letterbox, long side -> imgsz (640):
1280x720 x0.5, 1920x1080 x0.33, 576x1024 x0.625.
"""

from __future__ import annotations

import json
import zipfile
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# buckets on letterboxed size (what the model sees)
SIZE_BINS = [0, 8, 16, 32, np.inf]
SIZE_LABELS = ["<8", "8-16", "16-32", ">=32"]


def box_table(raw_dir: Path, imgsz: int = 640) -> pd.DataFrame:
    """One row per annotation, read from the zips."""
    rows = []
    for label_zip in sorted(Path(raw_dir).glob("labels/spie-video-*.zip")):
        video = label_zip.stem.removeprefix("spie-")
        with zipfile.ZipFile(label_zip) as zf:
            coco = json.loads(zf.read("annotations/instances_default.json"))
        images = {img["id"]: img for img in coco["images"]}
        for ann in coco["annotations"]:
            img = images[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            rows.append(
                {
                    "video": video,
                    "frame": Path(img["file_name"]).stem,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "img_w": img["width"],
                    "img_h": img["height"],
                }
            )
    df = pd.DataFrame(rows)
    df["rel_area"] = df["w"] * df["h"] / (df["img_w"] * df["img_h"])
    df["size_px"] = np.sqrt(df["w"] * df["h"])
    df["scale"] = imgsz / df[["img_w", "img_h"]].max(axis=1)  # long side -> imgsz
    df["w_lb"] = df["w"] * df["scale"]
    df["h_lb"] = df["h"] * df["scale"]
    df["size_lb"] = df["size_px"] * df["scale"]
    df["size_bucket"] = pd.cut(df["size_lb"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    return df


def video_resolutions(raw_dir: Path) -> pd.DataFrame:
    """W, H, fps per video from the container (no decode)."""
    rows = []
    for path in sorted(Path(raw_dir).glob("videos/*.mp4")):
        cap = cv2.VideoCapture(str(path))
        rows.append(
            {
                "video": path.stem,
                "video_w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "video_h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
            }
        )
        cap.release()
    return pd.DataFrame(rows)


def per_video_summary(boxes: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-video table: frames, empty share, box sizes, day/night, split."""
    # manifest must agree with labels
    from_labels = boxes.groupby("video")["frame"].nunique().rename("n_annotated_labels")
    df = manifest.set_index("video").join(from_labels)
    bad = df[df["n_annotated"] != df["n_annotated_labels"]]
    if not bad.empty:
        raise ValueError(f"manifest n_annotated disagrees with labels for: {list(bad.index)}")

    g = boxes.groupby("video")

    stats = pd.DataFrame(
        {
            "img_w": g["img_w"].first(),
            "img_h": g["img_h"].first(),
            "n_boxes": g.size(),
            "multi_box_frames": g["frame"].apply(lambda s: int((s.value_counts() > 1).sum())),
            "size_px_p10": g["size_px"].quantile(0.10),
            "size_px_med": g["size_px"].median(),
            "size_px_p90": g["size_px"].quantile(0.90),
            "rel_area_med_pct": g["rel_area"].median() * 100,
            "size_lb_med": g["size_lb"].median(),
            "share_lb_lt16": g["size_lb"].apply(lambda s: (s < 16).mean()),
        }
    )
    df = df.join(stats).drop(columns="n_annotated_labels")
    df["n_empty"] = df["n_frames"] - df["n_annotated"]
    df["empty_share"] = df["n_empty"] / df["n_frames"]
    return df.reset_index().sort_values(["split", "video"])


def per_split_summary(
    boxes: pd.DataFrame, per_video: pd.DataFrame, splits: list[str]
) -> pd.DataFrame:
    """Per-split table: videos, frames, empty share, box sizes, size-bucket shares."""
    boxes = boxes.merge(per_video[["video", "split"]], on="video")
    rows = []
    for split in splits:
        v = per_video[per_video["split"] == split]
        b = boxes[boxes["split"] == split]
        row = {
            "split": split,
            "videos": len(v),
            "night_videos": int(v["is_night"].sum()),
            "frames": int(v["n_frames"].sum()),
            "night_frames": int(v.loc[v["is_night"], "n_frames"].sum()),
            "empty_share": v["n_empty"].sum() / v["n_frames"].sum(),
            "boxes": len(b),
            "size_px_med": b["size_px"].median(),
            "rel_area_med_pct": b["rel_area"].median() * 100,
            "size_lb_p10": b["size_lb"].quantile(0.10),
            "size_lb_med": b["size_lb"].median(),
            "size_lb_p90": b["size_lb"].quantile(0.90),
            # one vote per video (long videos dominate the frame-level median)
            "video_median_of_size_lb_med": v["size_lb_med"].median(),
        }
        shares = b["size_bucket"].value_counts(normalize=True)
        row.update({f"share_lb_{k}": shares.get(k, 0.0) for k in SIZE_LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def size_shift_test(per_video: pd.DataFrame, col: str = "size_lb_med") -> pd.DataFrame:
    """Exact permutation test on per-video median size: train vs val, train vs test.

    unit = video, not frame (frames of one video are near-copies).
    every way to pick the other split from the pooled videos; p = share as extreme (2-sided)
    """
    train = per_video.loc[per_video["split"] == "train", col].to_numpy()
    rows = []
    for other in ("val", "test"):
        o = per_video.loc[per_video["split"] == other, col].to_numpy()
        pooled = np.concatenate([train, o])
        observed = np.median(train) - np.median(o)
        stats = []
        for idx in combinations(range(len(pooled)), len(o)):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(idx)] = True
            stats.append(np.median(pooled[~mask]) - np.median(pooled[mask]))
        stats = np.array(stats)
        rows.append(
            {
                "compare": f"train vs {other}",
                "metric": col,
                "train_videos": len(train),
                "other_videos": len(o),
                "train_median": np.median(train),
                "other_median": np.median(o),
                "difference": observed,
                "n_permutations": len(stats),
                "p_two_sided": float(np.mean(np.abs(stats) >= abs(observed) - 1e-12)),
            }
        )
    return pd.DataFrame(rows)


# Heuristic thresholds for color_stats, read off this dataset's 20 videos (HSV, 0-255).
# They SUGGEST a label for a human to confirm by eye; they are not ground truth.
DARK_V = 60  # night videos: mean V 39-51; darkest day video: 69
MONO_S = 5  # nearly grey frames (thermal or black-and-white): video-9 has 1.8
CHECK_S = 20  # low colour: look at it (night RGB, haze, or thermal with a coloured overlay)


def color_stats(
    frames_dir: Path, videos: list[str], n_frames: int = 20, ext: str = "png"
) -> pd.DataFrame:
    """Mean brightness V and saturation S (HSV) over n_frames evenly spaced frames per video,
    plus suggested labels. An objective check of the hand-set `is_night` in the manifest.

    dark      = low V          -> night (low-light RGB)
    monochrome = S close to 0  -> thermal or black-and-white, at ANY brightness
    A thermal video at night is not necessarily dark, so `dark` cannot judge it.
    """
    rows = []
    for video in videos:
        frames = sorted((Path(frames_dir) / video).glob(f"frame_*.{ext}"))
        if not frames:
            raise FileNotFoundError(f"{video}: no frames in {frames_dir}; run prepare_data.py")
        picks = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
        v, s = [], []
        for i in picks:
            hsv = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2HSV)
            s.append(hsv[..., 1].mean())
            v.append(hsv[..., 2].mean())
        rows.append({"video": video, "mean_v": np.mean(v), "mean_s": np.mean(s)})
    df = pd.DataFrame(rows)
    df["suggest_dark"] = df["mean_v"] < DARK_V
    df["suggest_modality"] = np.select(
        [df["mean_s"] < MONO_S, df["mean_s"] < CHECK_S], ["monochrome", "check"], "color"
    )
    return df
