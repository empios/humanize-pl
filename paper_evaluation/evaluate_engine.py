"""Evaluate the engine on real model output, separated by provenance.

What this replaces, and why it had to be replaced:

The previous version ran on paper_evaluation/dataset/ - forty documents of
four sentences each, assembled by `random.choice` from lists that are
literally the detector's own signal families. It also tested twelve of the
forty (`range(1, 4)`), and its baseline half was a chat session with Gemini
that no script could reproduce. Numbers from that setup do not support a
claim about the engine; they report how well the engine finds patterns that
were planted for it to find.

This version runs on docs_tests/ai_generated/, on every document, and splits
the results by `generated_by` from the manifest. That split is not a detail.
Measured after the reference profile was rebuilt, the eight hand-written
fixtures score 0.318-0.682 while the seven real model outputs score
0.192-0.335 - so a result quoted over the pooled set is carried substantially
by the documents that were written to be caught.

Usage:
    python paper_evaluation/evaluate_engine.py
    python paper_evaluation/evaluate_engine.py --engine nlp --mode standard
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from humanize_pl.config import Engine, Mode
from humanize_pl.core import humanize_text
from humanize_pl.detect import detect_document
from humanize_pl.detect.calibration import REVIEW_THRESHOLD, profile_for_family

CORPUS = Path("docs_tests/ai_generated")
OUTPUT = Path("paper_evaluation/engine_results.md")


def load_corpus() -> list[dict]:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest:
        path = CORPUS / entry["file"]
        if not path.exists():
            continue
        rows.append(
            {
                "id": entry["id"],
                "category": entry.get("category", "?"),
                # Anything without a recorded model is hand-written. Calling
                # that "unknown provenance" in the output rather than folding
                # it in with the rest is the entire point of this script.
                "handwritten": entry.get("generated_by", "unknown") == "unknown",
                "model": entry.get("generated_by", "unknown"),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return rows


def score(text: str) -> float | None:
    reference = profile_for_family("filing_official")
    diagnosis = detect_document(text, profile=reference, calibrate_against_default=False)
    return diagnosis.calibration.calibrated_score if diagnosis.calibration else None


def evaluate(rows: list[dict], *, mode: Mode, engine: Engine) -> list[dict]:
    results = []
    for row in rows:
        before = score(row["text"])
        outcome = humanize_text(
            row["text"], mode=mode, engine=engine, require_models=False, include_candidates=True
        )
        after = score(outcome.text)
        results.append(
            {
                **row,
                "signal_before": before,
                "signal_after": after,
                "delta": None if before is None or after is None else round(after - before, 4),
                "changes": len(outcome.changes),
                "rules": sorted({change.rule for change in outcome.changes}),
                "rejected": len(outcome.rejected),
                "engine_used": outcome.engine_used,
            }
        )
        print(f"  {row['id']}: {before} -> {after}  zmian {len(outcome.changes)}")
    return results


def summarise(results: list[dict], label: str) -> list[str]:
    if not results:
        return [f"### {label}\n\nBrak dokumentów.\n"]
    befores = [r["signal_before"] for r in results if r["signal_before"] is not None]
    afters = [r["signal_after"] for r in results if r["signal_after"] is not None]
    above = sum(1 for value in befores if value >= REVIEW_THRESHOLD)
    return [
        f"### {label} (n={len(results)})",
        "",
        (f"- sygnał przed: mediana {statistics.median(befores):.4f}, "
        f"zakres {min(befores):.4f}–{max(befores):.4f}"),
        (f"- sygnał po: mediana {statistics.median(afters):.4f}, "
        f"zakres {min(afters):.4f}–{max(afters):.4f}"),
        f"- powyżej progu {REVIEW_THRESHOLD} przed redakcją: {above}/{len(befores)}",
        f"- zastosowanych zmian łącznie: {sum(r['changes'] for r in results)}",
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--engine", default="basic", help="basic, nlp, hybrid")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    rows = load_corpus()
    if not rows:
        parser.error(f"Pusty korpus: {CORPUS}")

    print(f"Korpus: {len(rows)} dokumentów z {CORPUS}")
    results = evaluate(rows, mode=Mode(args.mode), engine=Engine(args.engine))

    real = [r for r in results if not r["handwritten"]]
    hand = [r for r in results if r["handwritten"]]

    lines = [
        "# Wyniki działania silnika Humanize-PL",
        "",
        (f"Tryb `{args.mode}`, silnik `{args.engine}`. Korpus: `{CORPUS}`, "
        f"{len(results)} dokumentów."),
        "",
        ("Wyniki są rozdzielone według proweniencji. Dokumenty pisane ręcznie "
        "powstały tak, by zawierać wykrywane wzorce, więc mierzą zdolność "
        "silnika do znalezienia tego, co w nich zasadzono — nie jego "
        "skuteczność na tekście modelu. Wniosku nie należy formułować dla "
        "obu grup łącznie."),
        "",
    ]
    lines += summarise(real, "Realne wyjście modelu")
    lines += summarise(hand, "Fixture'y pisane ręcznie (nie do wnioskowania)")

    lines += [
        "## Dokument po dokumencie",
        "",
        "| Dokument | Proweniencja | Kategoria | Sygnał przed | Sygnał po | Δ | Zmian | Reguły |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        provenance = "ręczny" if row["handwritten"] else row["model"]
        rules = ", ".join(f"`{rule}`" for rule in row["rules"]) or "—"
        lines.append(
            f"| {row['id']} | {provenance} | {row['category']} | "
            f"{row['signal_before']} | {row['signal_after']} | {row['delta']} | "
            f"{row['changes']} | {rules} |"
        )

    lines += [
        "",
        "## Ograniczenia",
        "",
        ("1. Strona AI jest mała i pochodzi z jednego modelu. Korpus pełnej "
        "wielkości buduje `tools/build_ai_corpus.py`."),
        ("2. Dokumenty AI są o rząd wielkości krótsze od uzasadnień, na których "
        "zbudowano profil ludzki, a każdy sygnał rodzinowy to częstość na "
        "1000 słów."),
        ("3. Obie strony różnią się też gatunkiem. Rozstrzygnąłby to profil "
        "ludzki w gatunku umów i opinii."),
        "",
    ]

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nZapisano: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
