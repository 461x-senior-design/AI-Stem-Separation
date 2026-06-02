"""Utilities for generating text EQ animation frames."""

import re
from pathlib import Path

DEFAULT_EQ_MIDI_TEXT = """
# Mini step-sequencer format:
# key=value headers
# track lines as "<name>: token token token ..."
# tokens: "."/"-" = rest, "x" = max velocity, "0..8" = velocity.
steps=16
subframes=3
#      1 . 2 . 3 . 4 . 5 . 6 . 7 . 8 .
kick:  8 . . 8 . . . . . . 8 . . . . .
snare: . . . . 8 . . . . . . . 8 . . .
hat:   . . 5 . . . 6 . . . 6 . . . 8 .
"""

_EQ_FRAMES_PATTERN = re.compile(r"(?ms)^EQ_FRAMES:\s*tuple\[str,\s*\.\.\.\]\s*=\s*\(\n.*?^\)\n?")


def make_eq_frames(midi_text=DEFAULT_EQ_MIDI_TEXT, bars=1):
    """Create a 5-band EQ animation loop from a small MIDI-like text pattern."""
    levels = ("▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")

    def to_cell(value):
        clamped = max(1, min(8, int(round(value))))
        return levels[clamped - 1]

    def parse_step_token(token):
        token = token.strip().lower()
        if token in {".", "-", "_"}:
            return 0
        if token == "x":
            return 8
        value = int(token)
        if value < 0 or value > 8:
            raise ValueError(f"Velocity token must be 0..8, got '{token}'.")
        return value

    def parse_midi_like_text(text):
        cfg = {"steps": 8, "subframes": 6}
        tracks = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line and ":" not in line:
                key, value = [chunk.strip().lower() for chunk in line.split("=", 1)]
                if key not in cfg:
                    raise ValueError(f"Unknown config key '{key}'.")
                cfg[key] = int(value)
                continue
            if ":" not in line:
                raise ValueError(f"Invalid pattern line: '{line}'.")
            name, body = [chunk.strip().lower() for chunk in line.split(":", 1)]
            tokens = [t for t in body.split() if t]
            tracks[name] = [parse_step_token(token) for token in tokens]

        steps = cfg["steps"]
        if steps <= 0:
            raise ValueError("steps must be > 0.")
        if cfg["subframes"] <= 0:
            raise ValueError("subframes must be > 0.")

        for track_name in ("kick", "snare", "hat"):
            if track_name not in tracks:
                tracks[track_name] = [0] * steps
            if len(tracks[track_name]) != steps:
                raise ValueError(
                    f"Track '{track_name}' has {len(tracks[track_name])} steps, expected {steps}."
                )
        return cfg, tracks

    cfg, tracks = parse_midi_like_text(midi_text)
    frames = []
    kick_env = {
        "low": (8, 7, 6, 5, 4, 3),
        "low_mid": (4, 4, 3, 2, 2, 1),
        "mid": (2, 2, 1, 1, 1, 1),
        "mid_high": (1, 1, 1, 1, 1, 1),
    }
    snare_env = {
        "low": (3, 2, 2, 1, 1, 1),
        "low_mid": (3, 4, 4, 3, 2, 1),
        "mid": (4, 6, 7, 6, 4, 3),
        "mid_high": (2, 3, 4, 3, 2, 1),
    }
    hat_env = {
        "mid_high": (2, 2, 1, 1, 1, 1),
        "high": (8, 7, 6, 5, 4, 3),
    }
    subframes = cfg["subframes"]

    for _ in range(bars):
        for step_idx in range(cfg["steps"]):
            kick_vel = tracks["kick"][step_idx] / 8.0
            snare_vel = tracks["snare"][step_idx] / 8.0
            hat_vel = tracks["hat"][step_idx] / 8.0

            for i in range(subframes):
                env_i = i % len(kick_env["low"])
                low = 1 + kick_env["low"][env_i] * kick_vel + snare_env["low"][env_i] * snare_vel
                low_mid = (
                    1
                    + kick_env["low_mid"][env_i] * kick_vel
                    + snare_env["low_mid"][env_i] * snare_vel
                )
                mid = 1 + kick_env["mid"][env_i] * kick_vel + snare_env["mid"][env_i] * snare_vel
                mid_high = (
                    1
                    + kick_env["mid_high"][env_i] * kick_vel
                    + snare_env["mid_high"][env_i] * snare_vel
                    + hat_env["mid_high"][env_i] * hat_vel
                )
                high = 1 + hat_env["high"][env_i] * hat_vel
                frames.append(
                    to_cell(low)
                    + to_cell(low_mid)
                    + to_cell(mid)
                    + to_cell(mid_high)
                    + to_cell(high)
                )

    return tuple(frames)


def format_eq_frames_literal(frames):
    """Format frames as the constants.py tuple literal body."""
    frame_lines = [f'    "{frame}",' for frame in frames]
    return "EQ_FRAMES: tuple[str, ...] = (\n" + "\n".join(frame_lines) + "\n)\n"


def replace_eq_frames_block(constants_text, frames_literal):
    """Replace the EQ_FRAMES tuple assignment block in constants.py text."""
    replaced_text, count = _EQ_FRAMES_PATTERN.subn(frames_literal, constants_text, count=1)
    if count != 1:
        raise ValueError("Unable to locate a unique EQ_FRAMES tuple block in constants.py.")
    return replaced_text


def update_constants_eq_frames(
    constants_path=None,
    midi_text=DEFAULT_EQ_MIDI_TEXT,
    bars=1,
):
    """Generate EQ frames and update constants.py in place.

    Returns:
        bool: True if constants.py changed, False if already up to date.
    """
    if constants_path is None:
        constants_path = Path(__file__).resolve().parents[1] / "constants.py"
    path = Path(constants_path)

    frames = make_eq_frames(midi_text=midi_text, bars=bars)
    frames_literal = format_eq_frames_literal(frames)

    original_text = path.read_text(encoding="utf-8")
    updated_text = replace_eq_frames_block(original_text, frames_literal)
    if updated_text == original_text:
        return False
    path.write_text(updated_text, encoding="utf-8")
    return True


if __name__ == "__main__":
    changed = update_constants_eq_frames(bars=1)
    if changed:
        print("Updated EQ_FRAMES in src/stemmy/constants.py")
    else:
        print("EQ_FRAMES already up to date in src/stemmy/constants.py")
