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


def test_dev_fullsong_eval_masked_potato_sets_info_and_disables_progress(monkeypatch) -> None:
    """Verify --potato forces INFO logs and disables progress output."""
    runner = CliRunner()

    monkeypatch.setattr(cli_mod, "dotenv_values", lambda _: {})
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("EVAL_PROGRESS", raising=False)
    monkeypatch.delenv("STEMMY_DISABLE_PROGRESS", raising=False)

    captured: dict[str, str] = {}

    def _fake_eval_main() -> None:
        captured["LOG_LEVEL"] = cli_mod.os.environ.get("LOG_LEVEL", "")
        captured["EVAL_PROGRESS"] = cli_mod.os.environ.get("EVAL_PROGRESS", "")
        captured["STEMMY_DISABLE_PROGRESS"] = cli_mod.os.environ.get("STEMMY_DISABLE_PROGRESS", "")

    monkeypatch.setattr(cli_mod, "fullsong_eval_masked_main", _fake_eval_main)

    result = runner.invoke(cli, ["dev", "eval", "--potato"])
    assert result.exit_code == 0
    assert captured["LOG_LEVEL"] == "INFO"
    assert captured["EVAL_PROGRESS"] == "0"
    assert captured["STEMMY_DISABLE_PROGRESS"] == "1"


def test_dev_train_help_lists_options() -> None:
    """Verify dev train exposes first-class Click options in --help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["dev", "train", "--help"])
    assert result.exit_code == 0
    assert "--data-root" in result.output
    assert "--epochs" in result.output
    assert "--potato" in result.output
    assert "--lr" in result.output
    assert "--batch-size" in result.output


def test_dev_train_env_var_sets_data_root() -> None:
    """Verify TRAIN_DATA_ROOT env var is picked up as the --data-root default."""
    runner = CliRunner()
    result = runner.invoke(cli, ["dev", "train"], env={"TRAIN_DATA_ROOT": "/nonexistent_path_xyz"})
    assert result.exit_code != 0
    assert "nonexistent_path_xyz" in result.output


def test_dev_train_potato_is_first_class_option() -> None:
    """Verify --potato is a proper Click option and not treated as unknown."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["dev", "train", "--potato", "--data-root", "/nonexistent_potato_test"]
    )
    assert "No such option: --potato" not in result.output
    assert result.exit_code != 0
    assert "nonexistent_potato_test" in result.output
