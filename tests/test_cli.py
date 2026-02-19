import re

from click.testing import CliRunner

import stemmy.tool.cli as cli_mod
from stemmy.tool.cli import cli, separate

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI color/style sequences for stable assertions in CI."""
    return ANSI_RE.sub("", text)


def test_separate_prints_expected_tree() -> None:
    """Verify the CLI stub prints the expected output preview tree."""
    runner = CliRunner()

    # Invoke the command as a user would from the shell.
    result = runner.invoke(separate, ["-i", "song.wav", "-o", "outdir", "--preview"])

    # Ensure the command succeeded.
    assert result.exit_code == 0

    # Strip ANSI codes to make the test stable across terminals/CI.
    output = strip_ansi(result.output)

    # Validate the core content of the preview tree.
    assert "Song Name:" in output
    assert "song.wav" in output
    assert "Expected Output:" in output
    assert "outdir" in output
    assert "song_drums.wav" in output
    assert "song_bass.wav" in output
    assert "song_vocals.wav" in output
    assert "song_other.wav" in output


def test_dev_fullsong_eval_masked_sets_default_env(monkeypatch) -> None:
    """Verify dev fullsong eval sets DATA/CKPT_DIR/EVAL_DIR defaults."""
    runner = CliRunner()

    monkeypatch.setattr(cli_mod, "dotenv_values", lambda _: {"TRAIN_DATA_ROOT": "/tmp/musdb"})
    monkeypatch.delenv("DATA", raising=False)
    monkeypatch.delenv("CKPT_DIR", raising=False)
    monkeypatch.delenv("EVAL_DIR", raising=False)

    captured: dict[str, str] = {}

    def _fake_eval_main() -> None:
        captured["DATA"] = cli_mod.os.environ.get("DATA", "")
        captured["CKPT_DIR"] = cli_mod.os.environ.get("CKPT_DIR", "")
        captured["EVAL_DIR"] = cli_mod.os.environ.get("EVAL_DIR", "")

    monkeypatch.setattr(cli_mod, "fullsong_eval_masked_main", _fake_eval_main)

    result = runner.invoke(cli, ["dev", "eval"])
    assert result.exit_code == 0
    assert captured["DATA"] == "/tmp/musdb"
    assert captured["CKPT_DIR"] == "checkpoints"
    assert captured["EVAL_DIR"] == "eval"


def test_dev_train_sets_default_args_and_forwards_extras(monkeypatch) -> None:
    """Verify dev train uses defaults, .env TRAIN_DATA_ROOT, and passthrough args."""
    runner = CliRunner()

    monkeypatch.setattr(cli_mod, "dotenv_values", lambda _: {"TRAIN_DATA_ROOT": "/tmp/musdb"})
    monkeypatch.delenv("TRAIN_DATA_ROOT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    captured: dict[str, object] = {}

    def _fake_train_main(argv) -> None:
        captured["argv"] = list(argv)
        captured["log_level"] = cli_mod.os.environ.get("LOG_LEVEL", "")

    monkeypatch.setattr(cli_mod, "train_main", _fake_train_main)

    result = runner.invoke(cli, ["dev", "train", "--lr", "0.001"])

    assert result.exit_code == 0
    assert captured["log_level"] == "ERROR"
    assert captured["argv"] == [
        "--data-root",
        "/tmp/musdb",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--time-frames",
        "256",
        "--checkpoint-dir",
        "checkpoints",
        "--export-ts",
        "--lr",
        "0.001",
    ]
