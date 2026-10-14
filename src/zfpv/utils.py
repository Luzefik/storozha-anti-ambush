"""Reproducibility helpers: seeding and an environment fingerprint for every run."""

from __future__ import annotations

import importlib
import platform
import random
import subprocess
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except ImportError:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def env_info() -> dict:
    """What a metrics file must record to be reproducible elsewhere, months later."""
    device = "cpu"
    torch_version = _version("torch")
    if torch_version:
        import torch

        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.get_device_name(0)}"
        elif torch.backends.mps.is_available():
            device = "mps"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "git_commit": _git_commit(),
        "packages": {
            name: _version(name)
            for name in ("numpy", "pandas", "cv2", "torch", "ultralytics", "transformers")
        },
    }
