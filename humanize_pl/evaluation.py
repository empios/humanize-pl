"""Measured quality of structure analysis and rewriting, held against a
recorded baseline.

The benchmark asks whether each output is safe. This asks whether the engine
got better or worse at its two jobs: telling what a document is and what it
owes its category, and editing AI-drafted legal text while leaving human
legal text mostly alone. Every gated metric has a direction; a change that
moves one the wrong way fails until the baseline is deliberately re-recorded.

Court rulings come from the local SAOS corpus, which is not committed. Without
it the ruling metrics are skipped, and said to be skipped, rather than guessed.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from humanize_pl import layout_ir
from humanize_pl.benchmark import safety_checks
from humanize_pl.blueprint import blueprint_for, check, is_heading, match_template, templates
from humanize_pl.categories import catalogue, classify_category
from humanize_pl.core import humanize_text
from humanize_pl.detect import detect_document
from humanize_pl.nlp.morfeusz import try_load_morfeusz

DEFAULT_BASELINE = Path("docs_tests/eval_baseline.json")
AI_DIR = Path("docs_tests/ai_generated")
RULINGS = Path("docs_tests/corpus/saos_holdout.jsonl")
# Enough rulings for a stable rate; each is cut to a few pages to keep the run short.
RULINGS_SAMPLE = 150
RULING_CHARS = 8000
# The opening lines hold the title, which names the category outright.
TITLE_LINES = 3
SAFETY_CHECKS = (
    "no_placeholder_leak",
    "numbers_preserved",
    "protected_fragments_preserved",
    "legal_sensitive_content_preserved",
)

# Gated metrics: +1 when higher is better, -1 when lower is better.
DIRECTION = {
    "category_exact": 1,
    "category_family": 1,
    "category_exact_untitled": 1,
    "category_wrong": -1,
    "category_wrong_untitled": -1,
    "ai_category_exact": 1,
    "rulings_given_category": -1,
    "checklist_false_blocks": -1,
    "checklist_notices_removal": 1,
    "template_self_match": 1,
    "template_match_gapped": 1,
    "edit_ratio_ai_to_rulings": 1,
    "fog_change_ai": -1,
    "unsafe_outputs": -1,
    "unsafe_outputs_rulings": -1,
}
LABELS = {
    "category_exact": "kategoria trafiona (wzory)",
    "category_family": "rodzina dokumentu trafiona (wzory)",
    "category_exact_untitled": "kategoria trafiona bez tytułu (wzory)",
    "category_wrong": "kategoria błędna, nie nieokreślona (wzory)",
    "category_wrong_untitled": "kategoria błędna bez tytułu (wzory)",
    "ai_category_exact": "kategoria trafiona (dokumenty AI)",
    "rulings_given_category": "orzeczenia uznane za dokument z katalogu",
    "checklist_false_blocks": "wzory blokowane przez listę kontrolną własnej kategorii",
    "checklist_notices_removal": "lista kontrolna zauważa usuniętą sekcję (wzory)",
    "template_self_match": "wzór rozpoznany we własnym tekście",
    "template_match_gapped": "wzór rozpoznany po usunięciu co piątego akapitu",
    "edits_per_1k_ai": "zmiany merytoryczne na 1000 słów (dokumenty AI)",
    "edits_per_1k_rulings": "zmiany merytoryczne na 1000 słów (orzeczenia)",
    "edit_ratio_ai_to_rulings": "ile razy więcej zmian w tekście AI niż w orzeczeniach",
    "fog_change_ai": "zmiana mglistości FOG-PL po redakcji (dokumenty AI; poniżej 0 = prościej)",
    "fog_change_rulings": "zmiana mglistości FOG-PL po redakcji (orzeczenia)",
    "unsafe_outputs": "wyniki naruszające bezpieczeństwo (wzory, dokumenty AI)",
    "unsafe_outputs_rulings": "wyniki naruszające bezpieczeństwo (orzeczenia)",
}
RULING_METRICS = {
    "rulings_given_category",
    "edits_per_1k_rulings",
    "edit_ratio_ai_to_rulings",
    "unsafe_outputs_rulings",
    "fog_change_rulings",
}

app = typer.Typer(
    add_completion=False,
    help="Pomiar jakości analizy struktury i redakcji względem zapisanej bazy.",
)


def _template_lines(ir: dict[str, Any]) -> list[str]:
    return [b["text"] for b in layout_ir.text_lines(ir)["blocks"] if b["text"].strip()]


def _ai_documents(root: Path) -> list[tuple[str, str]]:
    manifest = json.loads((root / AI_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [
        (row["category"], (root / AI_DIR / row["file"]).read_text(encoding="utf-8"))
        for row in manifest
    ]


def _rulings(root: Path) -> list[str] | None:
    path = root / RULINGS
    if not path.exists():
        return None
    rows = path.read_text(encoding="utf-8").splitlines()[:RULINGS_SAMPLE]
    return [json.loads(row)["text"][:RULING_CHARS] for row in rows]


def _matches(text: str, template_id: str) -> bool:
    found = match_template(text)
    return found is not None and found[0].id == template_id


def _section_spans(lines: list[str], blueprint) -> dict[str, tuple[int, int]]:
    """The lines holding each section the checklist finds: from a heading that
    names it up to the next heading, or, with no such heading, from the first
    line its phrases match up to the next found section."""
    lowered = [line.casefold() for line in lines]
    headings = [index for index, line in enumerate(lines) if is_heading(line)]
    starts: dict[str, tuple[int, bool]] = {}
    for section in blueprint.sections:
        hits = [i for i, line in enumerate(lowered) if any(p in line for p in section.matches)]
        if hits:
            heading = next((i for i in hits if i in headings), None)
            starts[section.id] = (hits[0], False) if heading is None else (heading, True)
    order = sorted({start for start, _ in starts.values()})
    spans: dict[str, tuple[int, int]] = {}
    for section_id, (start, by_heading) in starts.items():
        following = headings if by_heading else order
        spans[section_id] = (start, next((row for row in following if row > start), len(lines)))
    return spans


def structure_metrics(ai: list[tuple[str, str]], rulings: list[str] | None) -> dict[str, float]:
    families = {row.id: row.family for row in catalogue().values()}
    counts: Counter[str] = Counter()
    removals = noticed = 0
    rows = templates()
    for template in rows:
        lines = _template_lines(template.document)
        text = "\n".join(lines)
        # Each template is placed as a new document would be: its own entry is
        # left out of the template search, which would otherwise find itself.
        guess = classify_category(text, leave_out=template.id)
        counts["category_exact"] += guess.category.id == template.category
        counts["category_wrong"] += guess.specified and guess.category.id != template.category
        # "nieokreslony" has a family too; leaving a document unspecified is no
        # claim about its family.
        counts["category_family"] += (
            guess.specified and guess.category.family == families[template.category]
        )
        untitled = classify_category("\n".join(lines[TITLE_LINES:]), leave_out=template.id)
        counts["category_exact_untitled"] += untitled.category.id == template.category
        counts["category_wrong_untitled"] += (
            untitled.specified and untitled.category.id != template.category
        )
        blueprint = blueprint_for(template.category)
        if blueprint is not None:
            counts["checklist_false_blocks"] += check(text, blueprint).blocking
            # Would the checklist notice the section gone? A phrase that also
            # occurs elsewhere ("dnia" in "ustawy z dnia") keeps it "present".
            for section_id, (start, end) in _section_spans(lines, blueprint).items():
                removed = "\n".join(lines[:start] + lines[end:])
                removals += 1
                noticed += section_id not in check(removed, blueprint).present_sections
        counts["template_self_match"] += _matches(text, template.id)
        gapped = "\n".join(line for index, line in enumerate(lines) if index % 5 != 2)
        counts["template_match_gapped"] += _matches(gapped, template.id)
    metrics = {key: round(value / len(rows), 4) for key, value in counts.items()}
    metrics["checklist_notices_removal"] = round(noticed / removals, 4)
    metrics["ai_category_exact"] = round(
        sum(classify_category(text).category.id == category for category, text in ai) / len(ai), 4
    )
    if rulings is not None:
        metrics["rulings_given_category"] = round(
            sum(classify_category(text).specified for text in rulings) / len(rulings), 4
        )
    return metrics


def _rewrite(texts: list[str]) -> tuple[int, Counter[str], int, list[str]]:
    """Words, substantive edits by rule (spacing cleanup left out), unsafe
    outputs, and the outputs themselves."""
    words, rules, unsafe, outputs = 0, Counter(), 0, []
    for text in texts:
        result = humanize_text(text, mode="standard")
        words += len(text.split())
        rules.update(c.rule for c in result.changes if not c.rule.startswith("cleanup"))
        checks = safety_checks(text, result.text)
        unsafe += not all(checks[name] for name in SAFETY_CHECKS)
        outputs.append(result.text)
    return words, rules, unsafe, outputs


def _fog_change(before: list[str], after: list[str]) -> float:
    """Mean FOG-PL after rewriting minus before: below zero, easier to read."""
    def mean_fog(texts: list[str]) -> float:
        values = [detect_document(text, calibrate_against_default=False).metrics.get("fog_pl", 0.0)
                  for text in texts]
        return sum(values) / len(values)

    return round(mean_fog(after) - mean_fog(before), 2)


def rewrite_metrics(
    ai: list[tuple[str, str]], rulings: list[str] | None
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Edit rates on AI text and on rulings, plus edits per 1000 words by rule."""
    template_texts = ["\n".join(_template_lines(t.document)) for t in templates()]
    _, _, unsafe_templates, _ = _rewrite(template_texts)
    ai_texts = [text for _category, text in ai]
    ai_words, ai_rules, unsafe_ai, ai_outputs = _rewrite(ai_texts)
    metrics = {
        "edits_per_1k_ai": round(1000 * sum(ai_rules.values()) / ai_words, 2),
        "fog_change_ai": _fog_change(ai_texts, ai_outputs),
        "unsafe_outputs": unsafe_templates + unsafe_ai,
    }
    by_rule = {"ai": {rule: 1000 * n / ai_words for rule, n in ai_rules.items()}}
    if rulings is not None:
        words, rules, unsafe, outputs = _rewrite(rulings)
        metrics["fog_change_rulings"] = _fog_change(rulings, outputs)
        per_1k = 1000 * sum(rules.values()) / words
        metrics["edits_per_1k_rulings"] = round(per_1k, 2)
        metrics["edit_ratio_ai_to_rulings"] = round(
            metrics["edits_per_1k_ai"] / max(per_1k, 0.01), 2
        )
        metrics["unsafe_outputs_rulings"] = unsafe
        by_rule["rulings"] = {rule: 1000 * n / words for rule, n in rules.items()}
    return metrics, by_rule


def measure(root: Path = Path(".")) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    ai = _ai_documents(root)
    rulings = _rulings(root)
    rewrite, by_rule = rewrite_metrics(ai, rulings)
    current = {
        "morfeusz": try_load_morfeusz() is not None,
        "metrics": {**structure_metrics(ai, rulings), **rewrite},
    }
    return current, by_rule


def regressions(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Gated metrics that moved the wrong way, in Polish; measured on both sides only."""
    if current["morfeusz"] != baseline["morfeusz"]:
        recorded = "z Morfeuszem2" if baseline["morfeusz"] else "bez Morfeusza2"
        return [f"bazę zmierzono {recorded}; ten pomiar nie jest z nią porównywalny"]
    rows = []
    for key, better in DIRECTION.items():
        if key not in current["metrics"] or key not in baseline["metrics"]:
            continue
        before, now = baseline["metrics"][key], current["metrics"][key]
        if (now - before) * better < 0:
            rows.append(f"{LABELS[key]}: {before} → {now}")
    return rows


@app.command()
def main(
    baseline: Path = typer.Option(DEFAULT_BASELINE, "--baseline", help="Plik bazy (JSON)"),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Zapisz bieżący pomiar jako nową bazę"
    ),
    rules: bool = typer.Option(
        False, "--rules", help="Pokaż zmiany na 1000 słów według reguł: tekst AI i orzeczenia"
    ),
) -> None:
    current, by_rule = measure()
    recorded = json.loads(baseline.read_text(encoding="utf-8")) if baseline.exists() else None
    for key, value in current["metrics"].items():
        before = "" if recorded is None else recorded["metrics"].get(key, "—")
        gated = "" if key in DIRECTION else "  (informacyjnie)"
        print(f"{LABELS[key]}: {value}  [baza: {before}]{gated}")
    skipped = sorted(RULING_METRICS - current["metrics"].keys())
    if skipped:
        print(f"[yellow]Pominięto (brak lokalnego korpusu {RULINGS}):[/yellow] {', '.join(skipped)}")
    if rules:
        rows = sorted(set(by_rule["ai"]) | set(by_rule.get("rulings", {})))
        print("\nreguła | AI na 1000 słów | orzeczenia na 1000 słów")
        for rule in rows:
            ruling = by_rule.get("rulings", {}).get(rule)
            ruling_rate = "—" if ruling is None else f"{ruling:.2f}"
            print(f"{rule} | {by_rule['ai'].get(rule, 0.0):.2f} | {ruling_rate}")

    if update_baseline:
        if skipped:
            print("[red]Baza wymaga lokalnego korpusu orzeczeń; nie zapisano.[/red]")
            raise typer.Exit(1)
        baseline.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"[green]Zapisano bazę:[/green] {baseline}")
        return
    if recorded is None:
        print(f"[red]Brak bazy {baseline}. Zapisz ją flagą --update-baseline.[/red]")
        raise typer.Exit(1)
    rows = regressions(current, recorded)
    if rows:
        print("[red]Regresje względem bazy:[/red]")
        for row in rows:
            print(f"  - {row}")
        raise typer.Exit(1)
    print("[green]Brak regresji względem bazy.[/green]")


if __name__ == "__main__":
    app()
