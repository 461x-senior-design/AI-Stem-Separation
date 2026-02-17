import re

from click.testing import CliRunner

from stemmy.tool.cli import separate

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI color/style sequences for stable assertions in CI."""
    return ANSI_RE.sub("", text)


def test_separate_prints_expected_tree() -> None:
    """Verify the CLI stub prints the expected output preview tree."""
    runner = CliRunner()

    # Invoke the command as a user would from the shell.
    result = runner.invoke(separate, ["-i", "song.wav", "-o", "outdir"])

    # Ensure the command succeeded.
    assert result.exit_code == 0

    # Strip ANSI codes to make the test stable across terminals/CI.
    output = strip_ansi(result.output)

    # Validate the core content of the preview tree.
    assert "Song Name:" in output
    assert "song.wav" in output
    assert "Expected Output:" in output
    assert "outdir" in output
    assert "drums-song.wav" in output
    assert "vocals-song.wav" in output
    assert "bass-song.wav" in output
    assert "other-song.wav" in output
