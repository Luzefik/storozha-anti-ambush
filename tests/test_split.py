"""Leakage guards on the tracked split manifest. Needs no raw data, so CI can run it."""

from pathlib import Path

import pytest

from zfpv.data.split import load_manifest, videos_for

MANIFEST = Path("config/splits/video_split_manifest.csv")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


def test_every_video_in_exactly_one_split(manifest):
    assert manifest["video"].is_unique, "a video appears in more than one row"
    assert set(manifest["split"]) == {"train", "val", "test"}


def test_splits_are_disjoint(manifest):
    train, val, test = (set(videos_for(manifest, s)) for s in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)


def test_annotated_never_exceeds_frames(manifest):
    assert (manifest["n_annotated"] <= manifest["n_frames"]).all()


def test_night_is_represented_in_every_split(manifest):
    night = manifest[manifest["is_night"]]
    assert set(night["split"]) == {"train", "val", "test"}
