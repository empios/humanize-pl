from __future__ import annotations

import subprocess
import sys

import typer
from rich import print

app = typer.Typer(add_completion=False, help="Run local release checks for humanize-pl.")


@app.command()
def main(
    skip_lint: bool = typer.Option(False, "--skip-lint", help="Skip ruff check"),
    skip_benchmark: bool = typer.Option(False, "--skip-benchmark", help="Skip basic benchmark gate"),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip wheel build"),
) -> None:
    commands: list[list[str]] = [[sys.executable, "-m", "pytest", "-q"]]
    if not skip_lint:
        commands.append([sys.executable, "-m", "ruff", "check", "."])
    if not skip_benchmark:
        commands.append(
            [
                sys.executable,
                "-m",
                "humanize_pl.benchmark",
                "--engines",
                "basic",
                "--mode",
                "standard",
                "--allow-fallback",
                "--fail-on-status",
            ]
        )
    if not skip_build:
        commands.append([sys.executable, "-m", "build", "--wheel", "--no-isolation"])

    for command in commands:
        print(f"[bold]Running:[/bold] {' '.join(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise typer.Exit(completed.returncode)

    print("[green]Release check passed.[/green]")


if __name__ == "__main__":
    app()
