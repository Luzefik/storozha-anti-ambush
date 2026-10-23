# Experiment log

Every run that produced a number goes here, including the ones that failed.
Same structure as an `exp:` commit body.

## Template

### EXP-000: <short name> (<date>, <who>)

- **Hypothesis:** what you expect and why.
- **Setup:** model, size, `imgsz`, pretrained weights, epochs, seed, git commit,
  config overrides, hardware.
- **Result:** numbers from `results/metrics/<run-id>.json` (shared evaluator only),
  overall and day/night, with the video-level uncertainty.
- **Conclusion:** what changes because of this result.

---

## Shared notes for EXP-001 … EXP-004

- **Evaluator: provisional.** All numbers below come from Ultralytics' own `val` printout
  (`results.csv` of each run), not from the shared evaluator: `zfpv.eval.metrics.evaluate`
  is still `NotImplementedError`. They are fine for comparing runs with each other, not
  for the final report. Day/night and per-video breakdowns need the shared evaluator.
- **Eval split:** the `val` videos of `config/splits/video_split_manifest.csv`
  (`video-20`, `video-6`, `video-18`, night `video-8`; 1127 frames). The `test` videos have
  not been touched.
- **Two numbers per run.** `best` = the epoch Ultralytics keeps as `best.pt`, chosen on the
  same val set, so it is optimistic. `last10` = mean over the last 10 epochs, which is
  what the model sustains. The val curve jumps by ±0.2 mAP between neighbouring epochs
  (`results.csv` of each run), so `last10` is the number to compare.
- **Code:** commit `20e0cf6` (branch `feat/project-scaffold`; no commits since).
  Hardware: Colab GPU (T4 on 2026-10-17; GPU type of later sessions not recorded).
- **Duplicates.** With `seed=42` and `deterministic=True`, restarting the same config on
  the same GPU type reproduces it epoch by epoch. Restarted copies are listed under
  `duplicate_of` in `results/tables/runs_all.csv` and counted once.
- Tables: `results/tables/experiments_summary.{csv,md}`, `results/tables/runs_all.csv`,
  `results/tables/cv_yolo11s_640.csv`. Metrics: `results/metrics/EXP-00N.json`.
  Figures: `results/figures/exp_best_vs_last10.png`, `results/figures/cv_yolo11s_640_folds.png`.
  Regenerated from the Ultralytics run folders by a one-off script (kept outside the repo).

### EXP-001: YOLO26n @640, baseline (2026-10-17 … 10-20, AK)

- **Hypothesis:** an off-the-shelf small YOLO on default settings gives a usable first
  number and shows where it fails.
- **Setup:** `yolo26n.pt` (COCO-pretrained), `imgsz=640`, `epochs=100`, `patience=20`,
  `batch=-1` (auto), `seed=42`, `cache=ram`, default augmentation (`scale=0.5`,
  `mosaic=1.0`). Runs: `yolo26n-640`, `-2`, `-4` (3 independent); `-3` crashed before epoch 1.
- **Result (val, mean ± std over 3 runs):** best mAP50 **0.50 ± 0.04**, best recall
  0.50 ± 0.03; **last10 mAP50 0.16 ± 0.06**, last10 mAP50-95 0.06 ± 0.03.
  Every run peaks at epoch 5–7 and early-stops at 25–27. ~1.9 min/epoch.
- **Conclusion:** the model fits the train videos (train loss falls steadily) but does not
  transfer to the val videos (val cls loss rises): overfitting plus a size shift. Audit:
  median box ~41 px on train vs ~8.5 px on val (`results/audit/per_split.csv`). The single
  "0.56" is a lucky epoch, not the model's level. Next: larger input (EXP-003).

### EXP-002: RT-DETR-l @640 (2026-10-20, AK)

- **Hypothesis:** a transformer detector generalises better across videos than a small CNN.
- **Setup:** `rtdetr-l.pt`, `imgsz=640`, `batch=8`, otherwise as EXP-001 (`cache=False`).
  Runs: `rtdetr-l-640`, `-4` (2 independent); `-2`, `-3` crashed before epoch 1.
- **Result (val, mean ± std over 2 runs):** best mAP50 **0.48 ± 0.02**;
  **last10 mAP50 0.22 ± 0.01**, last10 mAP50-95 0.09. Best epoch 4 in both, early stop at 24.
  **~10.6 min/epoch, 5–6× slower than YOLO26n.**
- **Conclusion:** not better than YOLO26n within the noise, and far more expensive to train
  and to run on a Jetson. Dropped as the main line; a lighter transformer (RF-DETR-S,
  RT-DETR-R18) is the fairer comparison (see "Planned").

### EXP-003: YOLO26n @1280 (2026-10-17 … 10-20, AK)

- **Hypothesis:** the val drones are ~8 px; at 640 they shrink to a few pixels. Doubling the
  input size should raise recall on small objects.
- **Setup:** as EXP-001 but `imgsz=1280`, `cache=disk`. Independent runs: `yolo26n-1280-3`,
  `-6`, `-7`. `yolo26n-1280`, `-2`, `-4`, `-8` reproduce `-3` epoch by epoch, and `-5`
  reproduces `-7` (restarts of the same config).
- **Result (val, mean ± std over 3 runs):** best mAP50 0.51 ± 0.06, best recall 0.46 ± 0.06;
  **last10 mAP50 0.24 ± 0.03** (vs 0.16 at 640), last10 mAP50-95 0.10 ± 0.02 (vs 0.06).
  ~2.9 min/epoch (≈1.5× of 640, not 4×). Still peaks early (epoch 1–15).
- **Conclusion:** the larger input helps the sustained metric (+0.08 mAP50) and is cheap
  enough to become the default. Overfitting remains; `scale` augmentation was never raised
  (all runs `scale=0.5`). Next: `scale=0.9`, and compare 640 vs 1280 under CV, not on one split.

### EXP-004: YOLO11s @640, 4-fold cross-validation by video (2026-10-20, AK)

- **Hypothesis:** with 4 val videos, one split cannot tell configurations apart; CV by
  video shows how much the number depends on which videos are held out.
- **Setup:** pool = train + val videos (16), test untouched. 4 folds by video, night videos
  spread first, then day videos by size, round-robin (notebook `model-evaluations`).
  `yolo11s.pt`, `imgsz=640`, `epochs=50`, `patience=15`, `batch=-1`, `seed=42`, `cache=ram`.
  ~0.9 min/epoch.
- **Result:**

  | fold | val videos | best mAP50 | last10 mAP50 | last10 recall |
  |---|---|---|---|---|
  | 0 | 25 (night), 18, 6, 20 | 0.45 | 0.25 | 0.27 |
  | 1 | 8 (night), 9 (likely thermal), 3, 22 | 0.73 | 0.41 | 0.37 |
  | 2 | 2 (portrait), 21, 17, 12 | 0.57 | 0.40 | 0.41 |
  | 3 | 15, 26, 7, 11 | 0.33 | 0.24 | 0.26 |
  | **mean ± std** | | **0.52 ± 0.15** | **0.33 ± 0.08** | 0.33 ± 0.06 |

- **Conclusion:** the spread between folds (0.33 → 0.73) is larger than any difference
  between models in EXP-001…003. Single-split comparisons on this dataset are not
  reliable; every comparison from now on runs on these 4 folds. Not directly comparable to
  EXP-001…003 (different model and different val videos). Fold 3 (low-colour `video-7`,
  `video-26`) is the weakest: first target for error analysis. Fold 1 (night + thermal) is
  the best: check whether those videos are simply easier (larger boxes).

## Planned

All on the EXP-004 folds, same seed and epochs, compared on `last10` mean ± std.

| id | change | why |
|---|---|---|
| EXP-005 | YOLO26n @640 vs @1280, CV | first fair test of the size hypothesis |
| EXP-006 | best of EXP-005 + `scale=0.9` | train on drones shrunk to val/test size |
| EXP-007 | YOLOv8s-P2 @1280 | extra stride-4 head built for tiny objects |
| EXP-008 | YOLO11s / YOLO26s @1280 | model capacity (n → s) |
| EXP-009 | RF-DETR-S (or RT-DETR-R18) | fair-sized transformer comparison |
| EXP-010 | Faster R-CNN FPN (torchvision) | two-stage reference for small objects |
| EXP-011 | best model + SAHI tiles | proposal item 2 |
| EXP-012 | shared evaluator on test, once | first reportable numbers, day/night, per video |
