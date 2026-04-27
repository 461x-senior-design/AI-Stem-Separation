"""Tests for per-stem augmentation and cross-track remixing in Musdb18HQDataset.

These tests construct a tiny synthetic MUSDB18-HQ-shaped directory in tmp_path and
exercise the dataset with various combinations of (augment, remix, deterministic).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from stemmy.constants import HOP_LENGTH, N_FFT, STEMS_4, TARGET_SAMPLE_RATE
from stemmy.training.musdb18hq_dataset import CropConfig, Musdb18HQDataset
from stemmy.training.stft import StftConfig

requires_audiomentations = pytest.mark.skipif(
    importlib.util.find_spec("audiomentations") is None,
    reason="audiomentations is not installed",
)

TIME_FRAMES = 2  # minimum allowed; segment_samples = N_FFT + HOP*(T-1) = 5120 @ T=2.
SR = TARGET_SAMPLE_RATE


def _make_track(track_dir: Path, rng: np.random.Generator, num_samples: int) -> None:
    """Write 1 mixture + 4 stems with mixture == sum(stems) to track_dir."""
    track_dir.mkdir(parents=True, exist_ok=True)
    stems = {stem: rng.standard_normal(num_samples).astype(np.float32) * 0.1 for stem in STEMS_4}
    mixture = sum(stems.values())
    sf.write(str(track_dir / "mixture.wav"), mixture, SR, subtype="FLOAT")
    for stem_name, stem_wav in stems.items():
        sf.write(str(track_dir / f"{stem_name}.wav"), stem_wav, SR, subtype="FLOAT")


@pytest.fixture
def tiny_musdb(tmp_path: Path) -> Path:
    """Build a fake MUSDB18-HQ root with 8 train tracks + 1 test track."""
    rng = np.random.default_rng(0)
    # Each wav is twice the segment length so the random crop has room to vary.
    num_samples = (N_FFT + HOP_LENGTH * (TIME_FRAMES - 1)) * 2  # = 10240 samples

    train_root = tmp_path / "train"
    for i in range(8):
        _make_track(train_root / f"Track{i:02d}", rng, num_samples)

    test_root = tmp_path / "test"
    _make_track(test_root / "TestTrack00", rng, num_samples)

    return tmp_path


def _make_dataset(
    root: Path,
    *,
    augment: bool = False,
    remix: bool = False,
    deterministic: bool = False,
    partition: str = "train",
    seed: int = 0,
) -> Musdb18HQDataset:
    return Musdb18HQDataset(
        root_dir=str(root),
        subset="train",
        partition=partition,
        valid_fraction=0.25,
        split_seed=0,
        stft_cfg=StftConfig(),
        crop_cfg=CropConfig(time_frames=TIME_FRAMES),
        deterministic=deterministic,
        augment=augment,
        remix=remix,
        seed=seed,
    )


def test_no_augment_no_remix_path_unchanged(tiny_musdb: Path) -> None:
    """With augment=False and remix=False, deterministic crop is bit-identical."""
    ds_a = _make_dataset(tiny_musdb, deterministic=True, partition="train")
    ds_b = _make_dataset(tiny_musdb, deterministic=True, partition="train")
    mix_a, tgt_a = ds_a[0]
    mix_b, tgt_b = ds_b[0]
    assert torch.equal(mix_a, mix_b)
    assert torch.equal(tgt_a, tgt_b)


@requires_audiomentations
def test_augment_changes_targets_between_calls(tiny_musdb: Path) -> None:
    """augment=True with fresh seeds yields different target masks."""
    ds = _make_dataset(tiny_musdb, augment=True, partition="train", seed=123)
    _mix1, tgt1 = ds[0]
    _mix2, tgt2 = ds[0]
    assert not torch.equal(tgt1, tgt2), "augment should randomize per call"


def test_remix_pulls_stems_from_different_tracks(tiny_musdb: Path) -> None:
    """remix=True should load stems from a variety of tracks across many calls."""
    ds = _make_dataset(tiny_musdb, remix=True, partition="train", seed=42)

    seen_tracks_per_stem: dict[str, set[str]] = {stem: set() for stem in STEMS_4}
    original_read = ds._read_mono_segment

    def recording_read(path: Path, start: int, n: int) -> torch.Tensor:
        # path looks like .../<TrackName>/<stem>.wav
        track_name = Path(path).parent.name
        stem_name = Path(path).stem
        if stem_name in seen_tracks_per_stem:
            seen_tracks_per_stem[stem_name].add(track_name)
        return original_read(path, start, n)

    ds._read_mono_segment = recording_read  # type: ignore[method-assign]

    for _ in range(40):
        ds[0]

    # With 6 distinct tracks in train partition (8 - 2 valid) and 40 draws,
    # we should easily see at least 3 different tracks per stem.
    for stem, tracks in seen_tracks_per_stem.items():
        assert len(tracks) >= 3, f"remix did not vary tracks for stem {stem}: {tracks}"


@requires_audiomentations
def test_ratio_masks_sum_to_one_when_augmenting(tiny_musdb: Path) -> None:
    """sum_s targets_norm[s] == 1 everywhere within numerical tolerance."""
    ds = _make_dataset(tiny_musdb, augment=True, remix=True, partition="train", seed=7)
    _mix, tgt = ds[0]
    sum_per_bin = tgt.sum(dim=0)
    assert torch.isfinite(sum_per_bin).all()
    assert torch.allclose(sum_per_bin, torch.ones_like(sum_per_bin), atol=1e-4)


@requires_audiomentations
def test_finite_outputs_under_augmentation(tiny_musdb: Path) -> None:
    """No NaN/inf in mix or target tensors over a sweep of samples."""
    ds = _make_dataset(tiny_musdb, augment=True, remix=True, partition="train", seed=1)
    for _ in range(20):
        mix, tgt = ds[0]
        assert torch.isfinite(mix).all()
        assert torch.isfinite(tgt).all()


def test_augment_with_deterministic_raises(tiny_musdb: Path) -> None:
    """Cannot construct an augmenting dataset in deterministic mode."""
    with pytest.raises(ValueError):
        _make_dataset(tiny_musdb, augment=True, deterministic=True, partition="train")


def test_remix_with_deterministic_raises(tiny_musdb: Path) -> None:
    """Cannot construct a remixing dataset in deterministic mode either."""
    with pytest.raises(ValueError):
        _make_dataset(tiny_musdb, remix=True, deterministic=True, partition="train")


@requires_audiomentations
def test_valid_partition_via_build_dataloaders_never_augments(tiny_musdb: Path) -> None:
    """train.build_dataloaders forces augment=False/remix=False on the val Dataset."""
    from stemmy.train import build_dataloaders

    train_ds, val_ds, _train_loader, _val_loader = build_dataloaders(
        data_root=str(tiny_musdb),
        max_tracks=0,
        waveform_norm="peak",
        spectrogram_norm="freq_minmax",
        batch_size=1,
        num_workers=0,
        stft_cfg=StftConfig(),
        crop_cfg=CropConfig(time_frames=TIME_FRAMES),
        device=torch.device("cpu"),
        train_split="train",
        val_split="valid",
        valid_fraction=0.25,
        train_split_seed=0,
        augment=True,
        remix=True,
    )

    assert train_ds.augment is True
    assert train_ds.remix is True
    assert train_ds._stem_augment is not None

    assert val_ds.augment is False
    assert val_ds.remix is False
    assert val_ds._stem_augment is None
