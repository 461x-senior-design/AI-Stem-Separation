from pathlib import Path

import click
from rich.console import Console
from rich.padding import Padding
from rich.tree import Tree

DIR = Path.cwd().name
STEMS = ["drums-", "vocals-", "bass-", "other-"]


@click.command()
@click.option("--input-file", "-i", required=True, help="Name of input audio file.")
@click.option("--output-dir", "-o", default=DIR, help="Name of output directory.")
def separate(input_file: str, output_dir: str) -> None:
    console = Console()
    display_input = input_file
    console.print(f"\n[bold red]Song Name:[/bold red] [cyan]{display_input}[/cyan]")
    console.print("\n[bold red]Expected Output:[/bold red]")
    tree = Tree(f"[bold green]{output_dir}[/bold green][bold]/[/bold]")
    for stem in STEMS:
        tree.add(f"[cyan]{stem}{display_input}[/cyan]")
    console.print(Padding(tree, (0, 0, 0, 2)))


if __name__ == "__main__":
    separate()
