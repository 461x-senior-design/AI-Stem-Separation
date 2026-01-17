from pathlib import Path

import click

DIR = Path.cwd().name
STEMS = ["├── drums-", "├── vocals-", "├── bass-", "└── other-"]


@click.command()
@click.option("--input-file", "-i", help="Name of input audio file.")
@click.option("--output-dir", "-o", default=DIR, help="Name of output directory.")
def separate(input_file, output_dir):
    click.echo(f"Song Name: {input_file}")
    click.echo("Expected Output:")
    click.echo(f"  {output_dir}/")
    for i in STEMS:
        click.echo(f"  {i}{input_file}")


if __name__ == "__main__":
    separate()
