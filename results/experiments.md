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
