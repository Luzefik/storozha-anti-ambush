#!/usr/bin/env python
"""Download the gated Z-FPV-v1 dataset from Hugging Face and/or verify its checksums.

Access must have been granted to your HF account first (the dataset is gated and
reviewed manually). Authenticate with `hf auth login` once, or set HF_TOKEN.

    python scripts/download_data.py             # download (skips files already there)
    python scripts/download_data.py --verify    # only compare sha256 with CHECKSUMS.txt

data/raw/ is gitignored: the dataset licence (research-use-only) forbids redistributing
the data or derivatives that allow reconstructing the frames. Only CHECKSUMS.txt is
committed.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from zfpv.config import load_config

logger = logging.getLogger(__name__)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    recorded = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        recorded[name] = digest
    return recorded


def verify(raw_root: Path, checksums: Path) -> bool:
    """raw_root is data/raw; names in CHECKSUMS.txt are relative to it."""
    ok = True
    for name, expected in read_checksums(checksums).items():
        path = raw_root / name
        if not path.exists():
            logger.error("missing: %s", name)
            ok = False
        elif sha256_of(path) != expected:
            logger.error("checksum mismatch: %s", name)
            ok = False
    logger.info("checksums %s", "OK" if ok else "FAILED")
    return ok


def download(repo_id: str, raw_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=raw_dir,
        token=os.environ.get("HF_TOKEN") or None,  # None -> cached `hf auth login` token
    )
    logger.info("downloaded %s -> %s", repo_id, raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="only verify checksums")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    checksums = Path(cfg["paths"]["checksums"])

    if not args.verify:
        download(cfg["dataset"]["hf_repo_id"], raw_dir)
    if not verify(raw_dir.parent, checksums):
        sys.exit(1)


if __name__ == "__main__":
    main()
