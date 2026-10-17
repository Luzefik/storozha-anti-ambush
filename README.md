# storozha-anti-ambush

Vision-based detection of grounded "waiter" FPV drones (*ждуни*) from a ground camera,
in hard conditions: night, small objects, motion blur, partial occlusion.

UCU, course "Artificial Intelligence", semester project, 2026. Built from
[`ucu-ai-course/project-template`](https://github.com/ucu-ai-course/project-template).

![lint](https://github.com/Luzefik/storozha-anti-ambush/actions/workflows/lint.yml/badge.svg)

## Team


## Problem and motivation

Grounded FPV drones fly to a spot, land, and wait for a target with minimal power draw.
They are motionless, often camouflaged, low-contrast, and many are fibre-optic
controlled, so electronic warfare does not see them: visual observation is often the
only channel before the attack. Given a frame from a ground camera, decide whether a
grounded drone is present and return its bounding box.

**Main question:** does targeted expansion of the dataset improve detection on unseen
videos and at night? Secondary: is RGB enough at night, what detection distance is
practical, and does IR help. Full proposal:
[`docs/proposal/`](docs/proposal/).

## Data

Z-FPV-v1: 20 videos, 7196 frames, one class, COCO boxes from CVAT. Gated,
**research-use-only**: nothing derived from the frames is ever committed. Split by
video. Data card, licence terms, split table, checksums: [`data/README.md`](data/README.md).

## Approach

## Results

TODO: none yet. Every number reported here must come from `results/metrics/*.json`
(shared evaluator) and have an entry in [`results/experiments.md`](results/experiments.md).

## Quick start

Local (laptop, CPU: data pipeline, audit, evaluation):


Training (Colab GPU): see [`notebooks/README.md`](notebooks/README.md#running-on-colab).

## Repository structure

```
src/zfpv/data/      frames, coco_to_yolo, split, audit, tiling
src/zfpv/eval/      Detector interface (one adapter per model library) + shared evaluator
scripts/            download_data.py, prepare_data.py: thin CLIs over src/
config/             default.yaml + splits/video_split_manifest.csv (the split is config)
notebooks/          NN-topic-initials.ipynb, one owner each (see notebooks/README.md)
results/            experiments.md, metrics/*.json, figures/ (committed, small)
data/               gitignored except data card + checksums
models/             gitignored; weights stay private (see models/README.md)
tests/              leakage guards and conversion tests (no gated data needed)
docs/               proposal, poster, slides, CONTRIBUTING.md
```

## Reproducibility


## AI usage

## Contribution statement

| Member | Contribution | Responsible for |
|---|---|---|
| Andrii Kulbaba | 50% | TODO |
| Oleksandr Lykhanskyi | 50% | TODO |

## License and citation

| | License |
|---|---|
| Code | MIT, see [`LICENSE`](LICENSE) |
| Data | Z-FPV-v1 terms (research-use-only, gated), see [`data/README.md`](data/README.md) |
| Trained models | Not published; inherit the data terms |
| Text / poster / slides | CC BY 4.0 |
