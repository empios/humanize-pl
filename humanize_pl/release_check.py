from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from rich import print

app = typer.Typer(add_completion=False, help="Run local release checks for humanize-pl.")


@app.command()
def main(
    skip_lint: bool = typer.Option(False, "--skip-lint", help="Skip ruff check"),
    skip_benchmark: bool = typer.Option(False, "--skip-benchmark", help="Skip basic benchmark gate"),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip wheel build"),
) -> None:
    with TemporaryDirectory(prefix="humanize-pl-release-", dir=Path.cwd()) as directory:
        _run_checks(Path(directory), skip_lint=skip_lint, skip_benchmark=skip_benchmark, skip_build=skip_build)


def _run_checks(directory: Path, *, skip_lint: bool, skip_benchmark: bool, skip_build: bool) -> None:
    commands: list[list[str]] = [[sys.executable, "-B", "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", "--basetemp", str(directory / "tests")]]
    # A lemma swap whose target has no inflection paradigm can never fire:
    # `_build_replacement` returns None and the rule is silently dead. The
    # audit has always detected this and nothing ran it, so two rules shipped
    # dead. It costs milliseconds, so it goes first.
    commands.append([sys.executable, "-B", "tools/rules_lemma_audit.py"])
    if not skip_lint:
        commands.append([sys.executable, "-B", "-m", "ruff", "check", "--no-cache", "."])
    if not skip_benchmark:
        commands.append(
            [
                sys.executable,
                "-B",
                "-m",
                "humanize_pl.benchmark",
                "--engines",
                "basic",
                "--mode",
                "standard",
                "--allow-fallback",
                "--fail-on-status",
                "--output", str(directory / "benchmark"),
            ]
        )
    if not skip_build:
        commands.append([sys.executable, "-B", "-m", "build", "--wheel", "--no-isolation", "--outdir", str(directory / "wheel")])

    for command in commands:
        print(f"[bold]Running:[/bold] {' '.join(command)}")
        completed = subprocess.run(command, check=False, env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        if completed.returncode != 0:
            raise typer.Exit(completed.returncode)

    print("[green]Release check passed.[/green]")


if __name__ == "__main__":
    app()
