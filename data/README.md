# data/ — data card

## Name and source

**Z-FPV-v1**, UCU Autonomous UGV Club, 2026.
<https://huggingface.co/datasets/ucu-autonomous-ugv/Z-FPV-v1> (gated).

Grounded "waiter" (ambush) FPV drones seen from a ground / UGV viewpoint: a motionless,
often camouflaged target on the ground, frequently at night or in low light. Assembled
from publicly available footage reposted on social media (OSINT).

## License and terms of use

**Research-use-only, gated.** Access is reviewed manually by the dataset authors. By
requesting access we agreed to:

1. use the data only for non-commercial academic research;
2. not redistribute the data or any derivative that allows reconstruction of the
   source frames;
3. not use it to develop or improve offensive or targeting systems;
4. cite the accompanying paper (citation: TBA by the authors).

Consequences for this repo: no frames, crops, or images with drawn boxes are ever
committed or posted publicly; trained weights stay private; Colab/Drive copies stay in
private, unshared storage. This repo's MIT licence covers the code only.

## Size and format

| | |
|---|---|
| Videos | 20 (`videos/video-*.mp4`, H.264) |
| Frames | 7196 total, 6469 with a box, 727 (10.1%) without |
| Classes | 1: COCO `category_id 1 "drone"` -> YOLO class 0 `waiter_fpv_drone` |
| Labels | one COCO file per video: `labels/spie-video-N.zip -> annotations/instances_default.json`, bbox `[x, y, w, h]` in absolute px |
| Extra | `labels/spie-images.zip` + `images/photo-*.{jpg,png,webp}`: 10 still photos, not part of the video split (status: TODO decide) |

Annotation (from the dataset card): tight 2D boxes in CVAT; key frames labelled by hand,
TransT tracker propagated boxes, propagated labels manually corrected; every annotation
reviewed by the authors. COCO `attributes` also carry `occluded`, `track_id`, `keyframe`.

**Videos are not uniform** (from `ffprobe`): most are 1280×720, but `video-10` is
960×720, `video-23` is 1920×1080, and `video-2` is **portrait 576×1024** (and alone
holds 1451 frames, 20% of the dataset). Frame rates are 24–30 fps. Every video's frame
count matches its row in the manifest.

## raw / interim / processed

- `data/raw/Z-FPV-v1/` — as downloaded, never modified. `scripts/download_data.py`.
- `data/interim/frames/<video>/frame_000000.png` — every decoded frame, names matching
  the COCO `file_name`s. `scripts/prepare_data.py`.
- `data/processed/yolo/` — YOLO labels (`labels/<video>/`), per-video symlinks to the
  frames (`images/<video>`), `train.txt` / `val.txt` / `test.txt` and `data.yaml`.
  `data.yaml` has no `path:` key, so Ultralytics uses its own folder as the dataset root:
  it works on any machine. It is still generated, not committed: run
  `scripts/prepare_data.py` on every machine (laptop, Colab).

All three are gitignored. `data/raw/CHECKSUMS.txt` (sha256 of every video and label
archive) is committed: `python scripts/download_data.py --verify`.

## Splits

**By video, never by frame** (adjacent frames are near-duplicates: a frame split leaks
the test set into training). The assignment lives in
`config/splits/video_split_manifest.csv` and is checked by `tests/test_split.py`.

| Split | Videos (day + night) | Frames | Annotated |
|---|---|---|---|
| train | 11 + 1 | 4678 | 4076 |
| val | 3 + 1 | 1127 | 1027 |
| test | 3 + 1 (`video-10`, `video-23`, `video-14`, night `video-24`) | 1391 | 1366 |

Night: 3 videos, 581 frames (8.1%). Only **one** night video (139 frames) is in train,
and only one (249 frames) is in test: any night-specific conclusion rests on single
videos.

### How `is_night` was set

**Provenance unknown.** `is_night` was filled in by hand when the manifest was created,
before any code in this repo; no script computes it and nobody recorded who set it or
by what rule. TODO(team): write down who set it and how.

Objective check (`zfpv.data.audit.color_stats`, notebook `01-audit-ak`, table
`results/audit/color_stats.csv`): mean HSV brightness over 20 frames per video. The three
`is_night` videos (8, 24, 25) are the three darkest (mean V 39-51; darkest day video 69),
so the brightness check agrees with the manifest on all 20 videos.

It does not settle modality: `video-9` is almost grey (mean S 1.8) but not dark, so most
likely thermal or black-and-white, and brightness cannot tell day from night for thermal.
`video-7` and `video-26` also have little colour. TODO(team): confirm by eye, then add a
`modality` column (rgb / thermal) to the manifest.

## Personal data

None intentionally included (dataset card). If PII is found, report it to the dataset
authors; it will be removed upstream.

## Known limitations and biases

- Small and highly correlated: 20 videos, so effective sample size is closer to 20 than
  to 7196. Expect wide confidence intervals; resample by video.
- OSINT footage: re-compressed social-media video, unknown cameras, mixed resolutions
  and aspect ratios; possible selection bias toward footage that was worth posting.
- Night is under-represented (see Splits).
- Objects are small (tens of pixels); exact size distribution: TODO, audit notebook.

## Ethical risks

The data shows active military threats. The intended use is protective (earlier
detection of ambush drones threatening convoys, medevac and civilian transport). Risks:
misuse for targeting (forbidden by the terms), and over-trust in a detector validated on
very few videos: a miss can cost lives, too many false alarms erode crew trust.
