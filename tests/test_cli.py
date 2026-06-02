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
    import dotenv

    from stemmy.tool.dev import fullsong_eval_masked

    runner = CliRunner()

    monkeypatch.setattr(dotenv, "dotenv_values", lambda _: {"TRAIN_DATA_ROOT": "/tmp/musdb"})
    monkeypatch.delenv("DATA", raising=False)
    monkeypatch.delenv("CKPT_DIR", raising=False)
    monkeypatch.delenv("EVAL_DIR", raising=False)

    captured: dict[str, str] = {}

    def _fake_eval_main() -> None:
        captured["DATA"] = cli_mod.os.environ.get("DATA", "")
        captured["CKPT_DIR"] = cli_mod.os.environ.get("CKPT_DIR", "")
        captured["EVAL_DIR"] = cli_mod.os.environ.get("EVAL_DIR", "")

    monkeypatch.setattr(fullsong_eval_masked, "main", _fake_eval_main)

    result = runner.invoke(cli, ["dev", "eval"])
    assert result.exit_code == 0
    assert captured["DATA"] == "/tmp/musdb"
    assert captured["CKPT_DIR"] == "checkpoints"
    assert captured["EVAL_DIR"] == "eval"


def test_dev_fullsong_eval_masked_potato_sets_info_and_disables_progress(monkeypatch) -> None:
    """Verify --potato forces INFO logs and disables progress output."""
    import dotenv

    from stemmy.tool.dev import fullsong_eval_masked

    runner = CliRunner()

    monkeypatch.setattr(dotenv, "dotenv_values", lambda _: {})
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("EVAL_PROGRESS", raising=False)
    monkeypatch.delenv("STEMMY_DISABLE_PROGRESS", raising=False)

    captured: dict[str, str] = {}

    def _fake_eval_main() -> None:
        captured["LOG_LEVEL"] = cli_mod.os.environ.get("LOG_LEVEL", "")
        captured["EVAL_PROGRESS"] = cli_mod.os.environ.get("EVAL_PROGRESS", "")
        captured["STEMMY_DISABLE_PROGRESS"] = cli_mod.os.environ.get("STEMMY_DISABLE_PROGRESS", "")

    monkeypatch.setattr(fullsong_eval_masked, "main", _fake_eval_main)

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


def test_load_model_and_cfg_calls_get_default_model_when_no_args(monkeypatch) -> None:
    """Verify the HF auto-download path triggers when both --checkpoint and --torchscript are empty.

    We stub `get_default_model` and `load_pth_model` to avoid network and disk I/O;
    the assertion is that `_load_model_and_cfg` consults the hub helper and then
    loads the returned path via the .pth code path.
    """
    import stemmy.hub as hub_mod

    called: dict[str, object] = {}

    def _fake_get_default_model() -> str:
        called["hub"] = True
        return "/fake/default.pth"

    def _fake_load_pth_model(path: str, device: str, stems: int):
        called["path"] = path
        called["device"] = device
        called["stems"] = stems
        return object(), {"cfg": "fake"}

    def _fake_config_from_checkpoint(_obj):
        return cli_mod.InferenceConfig()

    monkeypatch.setattr(hub_mod, "get_default_model", _fake_get_default_model)
    monkeypatch.setattr(cli_mod, "load_pth_model", _fake_load_pth_model)
    monkeypatch.setattr(cli_mod, "config_from_checkpoint", _fake_config_from_checkpoint)

    # Bypass the existence check in _load_model_and_cfg — /fake/default.pth doesn't exist.
    monkeypatch.setattr(cli_mod.Path, "exists", lambda self: True)
    monkeypatch.setattr(cli_mod.Path, "is_file", lambda self: True)

    cli_mod._load_model_and_cfg(
        checkpoint=None, torchscript=None, device="cpu", stems=["drums", "bass", "vocals", "other"]
    )

    assert called.get("hub") is True, "get_default_model was not called"
    assert called.get("path", "").endswith("default.pth")
    assert called.get("stems") == 4


def test_register_dev_commands_no_op_when_training_missing(monkeypatch) -> None:
    """Verify _register_dev_commands does nothing when training modules are absent.

    Simulates the published-wheel environment where `stemmy.training.*` is excluded.
    The helper must swallow the ImportError and leave the parent group untouched.
    """
    import sys

    import click

    # Force `import stemmy.training.train` to raise ImportError regardless of install state.
    monkeypatch.setitem(sys.modules, "stemmy.training.train", None)

    parent = click.Group("root")
    cli_mod._register_dev_commands(parent)

    assert "dev" not in parent.commands


def test_register_dev_commands_attaches_dev_group_when_available() -> None:
    """Verify _register_dev_commands attaches dev group + subcommands when importable."""
    import click

    parent = click.Group("root")
    cli_mod._register_dev_commands(parent)

    assert "dev" in parent.commands
    dev_group = parent.commands["dev"]
    assert "train" in dev_group.commands
    assert "eval" in dev_group.commands


def test_spinner_updates_constants(monkeypatch) -> None:
    """Verify stemmy spinner triggers EQ frame regeneration/update."""
    runner = CliRunner()
    called: dict[str, bool] = {"value": False}

    def _fake_update_constants_eq_frames() -> bool:
        called["value"] = True
        return True

    monkeypatch.setattr(cli_mod, "update_constants_eq_frames", _fake_update_constants_eq_frames)

    result = runner.invoke(cli, ["spinner"])
    assert result.exit_code == 0
    assert called["value"] is True
    assert "Updated EQ_FRAMES in src/stemmy/constants.py" in result.output
