from pathlib import Path

import click
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.tree import Tree

BOLD_GREEN: str = "bold green"
BOLD_RED: str = "bold red"
CYAN: str = "cyan"
DIR: str = Path.cwd().name
STEMS: list[str] = ["drums-", "vocals-", "bass-", "other-"]


@click.command()
@click.option("--input-file", "-i", required=True, help="Name of input audio file.")
@click.option("--output-dir", "-o", default=DIR, help="Name of output directory.")
def separate(input_file: str, output_dir: str) -> None:
    """CLI wrapper for the separate command."""
    # TODO: Integrate functionality
    console = Console()
    display_input = input_file
    console.print("\nSong Name:", style=BOLD_RED, end=" ")
    console.print(display_input, style=CYAN)
    console.print("\nExpected Output:", style=BOLD_RED)
    tree = Tree(Text(output_dir, style=BOLD_GREEN))
    for stem in STEMS:
        tree.add(Text(f"{stem}{display_input}", style=CYAN))
    console.print(Padding(tree, (0, 0, 0, 2)))


if __name__ == "__main__":
    separate()
