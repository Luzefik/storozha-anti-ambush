# models/

Weights are **not committed** (`models/*` is gitignored) and **not published**.

## Why

- Size: detector checkpoints are tens to hundreds of MB; GitHub rejects files over
  100 MB and `git rm` does not remove a file from history.
- Licence: the models are trained on Z-FPV-v1 (research-use-only, gated, no use for
  offensive or targeting systems). Keep them private.

## Where they go

- Training on Colab: save checkpoints to your **private** Google Drive folder, named
  `<run-id>.pt` where `<run-id>` matches the entry in `results/experiments.md` and the
  file `results/metrics/<run-id>.json`.
- If the team needs a shared copy: a **private** Hugging Face model repo, both
  teammates as members. Never a public one.
- Locally: `models/<run-id>.pt` (ignored by git).

Every metrics JSON records the git commit and config overrides of its run, so a
checkpoint can always be traced back to the exact code and data that produced it.
