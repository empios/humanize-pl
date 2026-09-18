"""What changed, axis by axis, computed once for every surface.

The PDF had this table and nothing else did, so a batch could not be
aggregated from flow-report.json and a spreadsheet user never saw it. Three
surfaces computing it separately would disagree the first time one of them
was edited, so they all read these rows.

An axis with nothing behind it carries `before is None` and a reason. "No
office profile", "no skeleton for this category" and "checked, clean" are
three different facts, and printing 0 for the first two tells the reader a
check passed that never ran.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AxisRow:
    key: str
    axis: str
    measure: str
    before: float | None
    after: float | None
    # Why the axis does not apply; empty when it does.
    reason: str = ""
    # Digits after the decimal point when shown.
    digits: int = 0

    @property
    def applicable(self) -> bool:
        return self.before is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "axis": self.axis,
            "measure": self.measure,
            "before": self.before,
            "after": self.after,
            "applicable": self.applicable,
            "reason": self.reason,
        }

    def shown(self, value: float | None) -> str:
        if value is None:
            return "nie dotyczy"
        return f"{value:.{self.digits}f}".replace(".", ",")


def _summed(
    items: list[dict[str, Any]], key: str, inner: Callable[[dict[str, Any]], int]
) -> tuple[int, int] | None:
    """Total across items, or None when no item carried the layer."""
    before = after = 0
    seen = False
    for item in items:
        left, right = item.get(f"{key}_before"), item.get(f"{key}_after")
        if not isinstance(right, dict):
            continue
        seen = True
        after += inner(right)
        before += inner(left) if isinstance(left, dict) else inner(right)
    return (before, after) if seen else None


def _style_issues(payload: dict[str, Any]) -> int:
    return len(payload.get("issues") or [])


def _tone_deviations(payload: dict[str, Any]) -> int:
    return len(payload.get("deviations") or [])


def _missing_sections(payload: dict[str, Any]) -> int:
    if not payload.get("checked"):
        return 0
    return len(payload.get("missing_required") or []) + len(payload.get("empty_sections") or [])


def _traces(payload: dict[str, Any]) -> int:
    counts = payload.get("counts") or {}
    return sum(count for kind, count in counts.items() if kind != "placeholder")


def _fields(payload: dict[str, Any]) -> int:
    return int(payload.get("fields") or 0)


def axis_rows(items: list[dict[str, Any]]) -> list[AxisRow]:
    rows: list[AxisRow] = []

    # House style. Two measures share a row because a lawyer reads them as
    # one question: does this still sound like us.
    style = _summed(items, "style_compliance", _style_issues)
    tone = _summed(items, "tone", _tone_deviations)
    checked_tone = any(
        isinstance(item.get("tone_after"), dict) and item["tone_after"].get("checked")
        for item in items
    )
    # Without an office profile the check still runs, on the phrases banned
    # for the kind of document - which is not the office's style, and a row
    # titled "Styl kancelarii" showing 0 -> 0 would claim a check that did
    # not happen.
    office = checked_tone or any(
        isinstance(item.get("style_compliance_after"), dict)
        and item["style_compliance_after"].get("profile")
        for item in items
    )
    if style is None and not checked_tone:
        rows.append(
            AxisRow("house_style", "Styl kancelarii", "odstępstwa od wzorca",
                    None, None, reason="brak profilu kancelarii")
        )
    elif not office:
        rows.append(
            AxisRow(
                "house_style", "Styl",
                "zwroty zakazane dla rodzaju dokumentu (bez profilu kancelarii)",
                style[0] if style else 0,
                style[1] if style else 0,
            )
        )
    else:
        rows.append(
            AxisRow(
                "house_style", "Styl kancelarii", "zwroty zakazane i odstępstwa od wzorca",
                (style[0] if style else 0) + (tone[0] if tone else 0),
                (style[1] if style else 0) + (tone[1] if tone else 0),
            )
        )

    # Words. Density, not raw counts: a 2700-word document and a 300-word
    # one are not comparable on totals.
    words = sum(int(item.get("words") or 0) for item in items)
    found_before = sum(int(item.get("findings_before") or 0) for item in items)
    found_after = sum(int(item.get("findings_after") or 0) for item in items)
    if words:
        rows.append(
            AxisRow("words", "Słowa", "znaleziska na 1000 słów",
                    round(found_before * 1000 / words, 1),
                    round(found_after * 1000 / words, 1), digits=1)
        )
    else:
        rows.append(AxisRow("words", "Słowa", "znaleziska na 1000 słów", found_before, found_after))

    # Traces of the tool. Zero here is a result, not an empty check - human
    # documents almost never carry them - so the rows show whenever the
    # layer ran.
    measured = [
        item for item in items
        if isinstance(item.get("artifacts_after"), dict) and item["artifacts_after"]
    ]
    if measured:
        rows.append(
            AxisRow(
                "tool_traces", "Ślady czatbota",
                "markdown (**, #, ---, tabele) i zwroty do użytkownika",
                sum(_traces(item.get("artifacts_before") or {}) for item in measured),
                sum(_traces(item["artifacts_after"]) for item in measured),
            )
        )
        rows.append(
            AxisRow(
                "fields", "Gotowość do wysłania", "pola do uzupełnienia ([data], ……)",
                sum(_fields(item.get("artifacts_before") or {}) for item in measured),
                sum(_fields(item["artifacts_after"]) for item in measured),
            )
        )

    # Structure, counted only over items that had a skeleton.
    checked = [
        item for item in items
        if isinstance(item.get("blueprint_after"), dict) and item["blueprint_after"].get("checked")
    ]
    if not checked:
        rows.append(
            AxisRow("structure", "Struktura dokumentu", "brakujące sekcje wymagane",
                    None, None, reason="brak szkieletu dla tej kategorii")
        )
    else:
        structure = _summed(items, "blueprint", _missing_sections)
        before, after = structure if structure else (0, 0)
        measure = f"brakujące sekcje wymagane (sprawdzono {len(checked)} z {len(items)})"
        # A gap closed by a clause the model wrote is not a gap a lawyer
        # closed, and the row must not read as if it were.
        drafted = sum(
            1
            for item in items
            for row in item.get("drafted_sections") or []
            if row.get("inserted", True)
        )
        if drafted:
            measure += f"; {drafted} dopisał model, zob. 1.2"
        rows.append(AxisRow("structure", "Struktura dokumentu", measure, before, after))
    return rows
