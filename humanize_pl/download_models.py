from __future__ import annotations

import typer
from rich import print

from humanize_pl.nlp.semantic import DEFAULT_FLUENCY_MODEL, DEFAULT_SEMANTIC_MODEL

app = typer.Typer(add_completion=False)


@app.command()
def main(
    stanza_model: bool = typer.Option(False, "--stanza", help="Download Stanza Polish model"),
    transformers_model: bool = typer.Option(False, "--transformers", help="Download sentence-transformer model"),
    fluency_model: bool = typer.Option(False, "--fluency", help="Download masked-LM fluency model"),
    check_morfeusz: bool = typer.Option(
        False,
        "--morfeusz",
        help="Verify that morfeusz2 is importable and report install hints",
    ),
    semantic_model_name: str = typer.Option(
        DEFAULT_SEMANTIC_MODEL,
        "--semantic-model",
        help="Sentence-transformer model name",
    ),
    fluency_model_name: str = typer.Option(
        DEFAULT_FLUENCY_MODEL,
        "--fluency-model",
        help="Masked-LM model name",
    ),
) -> None:
    if not stanza_model and not transformers_model and not fluency_model and not check_morfeusz:
        stanza_model = True

    if stanza_model:
        print("[bold]Downloading Stanza Polish model...[/bold]")
        import stanza  # type: ignore

        stanza.download("pl")
        print("[green]Stanza model ready.[/green]")

    if transformers_model:
        print("[bold]Downloading Polish sentence-transformer model...[/bold]")
        from sentence_transformers import SentenceTransformer  # type: ignore

        SentenceTransformer(semantic_model_name)
        print(f"[green]Semantic model ready:[/green] {semantic_model_name}")

    if fluency_model:
        print("[bold]Downloading Polish masked-LM fluency model...[/bold]")
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore

        AutoTokenizer.from_pretrained(fluency_model_name)
        AutoModelForMaskedLM.from_pretrained(fluency_model_name)
        print(f"[green]Fluency model ready:[/green] {fluency_model_name}")

    if check_morfeusz:
        print("[bold]Checking Morfeusz2 availability...[/bold]")
        try:
            from humanize_pl.nlp.morfeusz import MorfeuszAnalyzer

            analyzer = MorfeuszAnalyzer()
            sample = analyzer.analyses("pracownikiem")
            print(f"[green]Morfeusz2 ready.[/green] Sample analyses for 'pracownikiem': {len(sample)}")
        except Exception as exc:
            print(f"[red]Morfeusz2 unavailable:[/red] {type(exc).__name__}: {exc}")
            print(
                "[yellow]Hint:[/yellow] Morfeusz2 bundles a native SGJP dictionary. "
                "On macOS try: brew install morfeusz2 && pip install -e '.[morfeusz]'. "
                "On Debian/Ubuntu add the morfeusz APT repo first. See sgjp.pl/morfeusz/."
            )


if __name__ == "__main__":
    app()
