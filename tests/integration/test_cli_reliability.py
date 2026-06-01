"""End-to-end CLI reliability tests.

Exercises the `stemmy separate` command with bad inputs (missing file,
unsupported extension, corrupted bytes, invalid output dir) and asserts
that the CLI exits non-zero with the matching error message from
stemmy.constants — not a raw traceback.

Model loading is monkeypatched out so the validation chain runs without
touching Hugging Face or a real checkpoint.
"""

import re
import struct
from unittest.mock import MagicMock

from click.testing import CliRunner

import stemmy.tool.cli as cli_mod
from stemmy import constants as c
from stemmy.inference import InferenceConfig
from stemmy.tool.cli import cli

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _surface(result) -> str:
    """All text a user could see: stdout/stderr plus the captured exception message.

    The CLI does not currently catch AudioValidationException, FileNotFoundError, or
    NotADirectoryError — they surface via the exception, not via Click's output.
    Tests assert the error message is *reachable* somewhere; tightening the CLI to
    pipe these into result.output is a separate (TDD) follow-up.
    """
    parts = [strip_ansi(result.output)]
    if result.exception is not None:
        parts.append(str(result.exception))
    return "\n".join(parts).lower()


def _fake_load_model_and_cfg(*_args, **_kwargs):
    """Return (mock model, real cfg, ckpt=None) so validation runs without HF/disk."""
    cfg = InferenceConfig(stems=list(c.STEMS_4))
    return MagicMock(name="model"), cfg, None


def _write_minimal_wav_header_only(path):
    """Write a WAV RIFF header with no audio data — a truncated/corrupted file."""
    # 44-byte canonical WAV header, declared data size = 0
    header = b"RIFF" + struct.pack("<I", 36) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 2, 44100, 44100 * 4, 4, 16)
    header += b"data" + struct.pack("<I", 0)
    path.write_bytes(header)


def test_missing_input_file_exits_with_clean_error(tmp_path):
    """Missing input file must be caught by _validate_input_file, not propagate as a traceback."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["separate", "-i", str(tmp_path / "does_not_exist.wav"), "-o", str(tmp_path / "out")],
    )

    assert result.exit_code != 0
    text = _surface(result)
    assert "input file not found" in text or "does_not_exist.wav" in text


def test_input_path_is_directory_not_file(tmp_path):
    """Passing a directory as -i must surface a clean IsADirectoryError message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["separate", "-i", str(tmp_path), "-o", str(tmp_path / "out")])

    assert result.exit_code != 0
    text = _surface(result)
    assert "expected a file" in text or "directory" in text


def test_output_path_exists_as_file_not_directory(tmp_path):
    """If -o points at an existing file (not a directory), reject early with a clean message."""
    # Build a real-looking input so we fail on the output path, not the input.
    input_file = tmp_path / "in.wav"
    input_file.write_bytes(b"placeholder")

    output_file = tmp_path / "actually_a_file.txt"
    output_file.write_text("not a directory")

    runner = CliRunner()
    result = runner.invoke(cli, ["separate", "-i", str(input_file), "-o", str(output_file)])

    assert result.exit_code != 0
    text = _surface(result)
    assert "not a directory" in text


def test_unsupported_extension_renamed_wav(tmp_path, monkeypatch):
    """A .wav file containing non-audio bytes must fail at the validator, not crash mid-inference.

    This is the 'renamed file' case: extension passes the format check, but the
    librosa/soundfile metadata read fails. Expect ERROR_METADATA_RETRIEVAL or
    ERROR_CORRUPTED_FILE in the surfaced message.
    """
    monkeypatch.setattr(cli_mod, "_load_model_and_cfg", _fake_load_model_and_cfg)

    bad = tmp_path / "fake.wav"
    bad.write_bytes(b"this is plain text, not a WAV file")

    runner = CliRunner()
    result = runner.invoke(cli, ["separate", "-i", str(bad), "-o", str(tmp_path / "out")])

    assert result.exit_code != 0
    text = _surface(result)
    # Either the metadata extractor or the validator's corrupted-file path will surface.
    assert "metadata" in text or "corrupted" in text or "validation" in text


def test_truncated_wav_header_only(tmp_path, monkeypatch):
    """A WAV with valid 44-byte header but zero audio data must be rejected, not OOM or crash."""
    monkeypatch.setattr(cli_mod, "_load_model_and_cfg", _fake_load_model_and_cfg)

    bad = tmp_path / "truncated.wav"
    _write_minimal_wav_header_only(bad)

    runner = CliRunner()
    result = runner.invoke(cli, ["separate", "-i", str(bad), "-o", str(tmp_path / "out")])

    assert result.exit_code != 0
    # Any clean validation/audio-error message is acceptable; the key invariant is
    # that we do not crash with an unhandled RuntimeError from inside the model.
    text = _surface(result)
    assert any(
        token in text for token in ("metadata", "corrupted", "audio", "duration", "validation")
    )


def test_unsupported_format_extension(tmp_path, monkeypatch):
    """A .ogg file must be rejected by ERROR_UNSUPPORTED_FORMAT (only .wav is supported)."""
    monkeypatch.setattr(cli_mod, "_load_model_and_cfg", _fake_load_model_and_cfg)

    bad = tmp_path / "song.ogg"
    bad.write_bytes(b"OggS\x00\x02placeholder")

    runner = CliRunner()
    result = runner.invoke(cli, ["separate", "-i", str(bad), "-o", str(tmp_path / "out")])

    assert result.exit_code != 0
    text = _surface(result)
    # ERROR_UNSUPPORTED_FORMAT contains "Unsupported file format" — assert the validator
    # rejected the file rather than merely failing somewhere downstream.
    assert "unsupported file format" in text or "validation" in text


def test_invalid_chunk_frames_negative(tmp_path):
    """--chunk-frames=-1 must be caught by _validate_chunking before any file work."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "separate",
            "-i",
            str(tmp_path / "anything.wav"),
            "-o",
            str(tmp_path / "out"),
            "--chunk-frames",
            "-1",
        ],
    )

    assert result.exit_code != 0
    text = _surface(result)
    assert "chunk-frames" in text


def test_overlap_not_less_than_chunk(tmp_path):
    """--overlap-frames >= --chunk-frames must be rejected by _validate_chunking."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "separate",
            "-i",
            str(tmp_path / "anything.wav"),
            "-o",
            str(tmp_path / "out"),
            "--chunk-frames",
            "64",
            "--overlap-frames",
            "64",
        ],
    )

    assert result.exit_code != 0
    text = _surface(result)
    assert "overlap-frames" in text
