from pathlib import Path

import pytest

from stemmy.tool.eq_frames import (
    format_eq_frames_literal,
    make_eq_frames,
    replace_eq_frames_block,
    update_constants_eq_frames,
)


def test_replace_eq_frames_block_replaces_assignment() -> None:
    """Replace only the EQ_FRAMES block in constants-like text."""
    original = (
        'HEADER = "x"\n'
        "EQ_FRAMES: tuple[str, ...] = (\n"
        '    "a",\n'
        ")\n"
        'FOOTER = "y"\n'
    )
    replacement = format_eq_frames_literal(("█▅▃▄▇", "▁▁▁▁▁"))

    updated = replace_eq_frames_block(original, replacement)

    assert 'HEADER = "x"\n' in updated
    assert 'FOOTER = "y"\n' in updated
    assert '    "█▅▃▄▇",' in updated
    assert '    "▁▁▁▁▁",' in updated
    assert '    "a",' not in updated


def test_replace_eq_frames_block_raises_when_missing() -> None:
    """Raise when no EQ_FRAMES block exists."""
    with pytest.raises(ValueError, match="EQ_FRAMES"):
        replace_eq_frames_block('HEADER = "x"\n', format_eq_frames_literal(("x",)))


def test_update_constants_eq_frames_noop_when_unchanged(tmp_path: Path) -> None:
    """Updating an already-matching constants file should return False."""
    pattern = """
steps=4
subframes=2
kick: 8 . 8 .
snare: . 7 . 7
hat: 5 5 5 5
"""
    frames = make_eq_frames(midi_text=pattern, bars=1)
    constants_text = (
        "# test constants\n"
        + format_eq_frames_literal(frames)
        + 'TAIL = "ok"\n'
    )
    constants_path = tmp_path / "constants.py"
    constants_path.write_text(constants_text, encoding="utf-8")

    changed = update_constants_eq_frames(
        constants_path=constants_path,
        midi_text=pattern,
        bars=1,
    )

    assert changed is False
    assert constants_path.read_text(encoding="utf-8") == constants_text
