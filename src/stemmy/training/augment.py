"""Audio-waveform augmentation pipeline for the MUSDB18-HQ training dataset.

`build_augment_pipeline(cfg)` returns a configured `audiomentations.Compose` or
`None`. `None` means no augmentation is active and the dataset takes its
unmodified baseline code path. The dataset applies the pipeline once per stem
with `freeze_parameters()` so all stems in a single training example receive
identical random parameters — preserving the mix=sum(stems) invariant after
the dataset recomputes the mix as the sum of augmented stems.
"""

from dataclasses import asdict, dataclass
from typing import Optional

from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    PitchShift,
    PolarityInversion,
    Shift,
    TimeStretch,
)


@dataclass(frozen=True)
class AugmentConfig:
    gain_p: float = 0.0
    gain_db: float = 6.0

    pitch_p: float = 0.0
    pitch_semitones: float = 2.0

    time_stretch_p: float = 0.0
    time_stretch_range: float = 0.1

    shift_p: float = 0.0
    shift_fraction: float = 0.1

    polarity_p: float = 0.0

    noise_p: float = 0.0
    noise_amplitude: float = 0.005

    def to_dict(self) -> dict:
        return asdict(self)


def build_augment_pipeline(cfg: AugmentConfig) -> Optional[Compose]:
    transforms = []

    if cfg.gain_p > 0:
        transforms.append(
            Gain(min_gain_db=-cfg.gain_db, max_gain_db=cfg.gain_db, p=cfg.gain_p)
        )
    if cfg.pitch_p > 0:
        transforms.append(
            PitchShift(
                min_semitones=-cfg.pitch_semitones,
                max_semitones=cfg.pitch_semitones,
                p=cfg.pitch_p,
            )
        )
    if cfg.time_stretch_p > 0:
        transforms.append(
            TimeStretch(
                min_rate=1.0 - cfg.time_stretch_range,
                max_rate=1.0 + cfg.time_stretch_range,
                leave_length_unchanged=True,
                p=cfg.time_stretch_p,
            )
        )
    if cfg.shift_p > 0:
        transforms.append(
            Shift(
                min_shift=-cfg.shift_fraction,
                max_shift=cfg.shift_fraction,
                shift_unit="fraction",
                rollover=True,
                p=cfg.shift_p,
            )
        )
    if cfg.polarity_p > 0:
        transforms.append(PolarityInversion(p=cfg.polarity_p))
    if cfg.noise_p > 0:
        transforms.append(
            AddGaussianNoise(
                min_amplitude=1e-6,
                max_amplitude=max(cfg.noise_amplitude, 1e-6),
                p=cfg.noise_p,
            )
        )

    if not transforms:
        return None
    return Compose(transforms, shuffle=False)
