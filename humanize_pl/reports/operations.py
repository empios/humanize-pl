"""Scope and results of editing, completeness checks and drafting."""

from collections import Counter
from typing import Any

LABELS = {
    "disabled": "wyłączona", "completed": "wykonana", "unavailable": "niedostępna",
    "incomplete": "niepełna", "failed": "błąd", "added": "dopisano sekcje",
    "not_needed": "nie wykryto wymaganych braków", "not_added": "nic nie dopisano",
    "not_saved": "wycofano zapis",
}


def operation_lines(items: list[dict[str, Any]]) -> list[str]:
    """Summarise recorded operations; do not infer scope for older reports."""
    records = [row["operations"] for row in items if row.get("operations")]
    if not records:
        return []
    lines = []
    for key, title in (
        ("editing", "Redakcja językowa"),
        ("completeness", "Kontrola kompletności"),
        ("drafting", "Dopisywanie sekcji"),
    ):
        states = Counter(record[key]["status"] for record in records)
        labels = LABELS if key != "drafting" else {**LABELS, "disabled": "wyłączone", "unavailable": "niedostępne"}
        status = "; ".join(
            labels.get(state, state) + (f" ({count})" if len(records) > 1 else "")
            for state, count in states.items()
        )
        line = f"{title}: {status}."
        if key == "editing":
            line += f" Zmiany istniejącego tekstu: {sum(r[key]['changes'] for r in records)}."
        elif key == "completeness" and any(r[key]["requested"] for r in records):
            line += " Wynik dotyczy wybranego szkieletu; nie potwierdza poprawności prawnej."
        elif key == "drafting":
            line += f" Sekcje dodane: {sum(r[key]['sections_added'] for r in records)}."
            if any(r[key]["requires_review"] for r in records):
                line += " Pełna treść dopisków jest w rejestrze; wymaga przeglądu prawnika."
        lines.append(line)
    return lines
