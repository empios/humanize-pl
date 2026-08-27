"""Rebuild a report from a run that already happened.

Runs are expensive and already done: a folder processed last month must not
have to be processed again just because the reporting improved. So the PDF is
regenerated from the saved payload instead.

The catch is that older payloads predate the per-family and per-metric
before/after numbers, and those cannot be invented. Where the documents are
still on disk they are re-diagnosed — detection needs no models and no rewrite,
so this is cheap. Where they are gone, the missing sections stay missing and
the report says so rather than printing zeros that would read as "nothing
found".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from difflib import SequenceMatcher

from humanize_pl.detect import detect_document
from humanize_pl.sentence_splitter import split_sentences

REPORT_NAME = "flow-report.json"

# Fields added after the first flows shipped. A payload carrying all of them
# needs no backfill.
BACKFILLED_FIELDS = (
    "family_counts_before",
    "family_counts_after",
    "metrics_before",
    "metrics_after",
    "findings_after",
    "examples",
)


def reconstruct_examples(
    before_text: str, after_text: str, *, limit: int = 12
) -> list[dict[str, str]]:
    """Recover the before/after pairs by comparing the two texts.

    An older run recorded how many rewrites it applied but not which sentences
    they touched. Both versions of the text are still on disk, though, so the
    pairs can be read straight off the difference between them. Sentences that
    were only added or only removed are skipped: without the run's own record
    there is no honest way to say what they were paired with.
    """
    if not before_text or before_text == after_text:
        return []

    before = _sentences(before_text)
    after = _sentences(after_text)
    pairs: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        a=before, b=after, autojunk=False
    ).get_opcodes():
        if tag != "replace":
            continue
        if i2 - i1 == j2 - j1:
            pairs.extend(
                {"before": before[i1 + step], "after": after[j1 + step], "issue": ""}
                for step in range(i2 - i1)
            )
        else:
            pairs.append(
                {
                    "before": " ".join(before[i1:i2]),
                    "after": " ".join(after[j1:j2]),
                    "issue": "",
                }
            )
    return pairs[:limit]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
        for sentence in split_sentences(paragraph)
        if sentence.strip()
    ]


def payload_from_workbook(
    path: str | Path,
    *,
    column: str,
    sheet_name: str | None = None,
    header_row: int | None = 1,
) -> dict[str, Any]:
    """Rebuild a payload from a finished workbook, without rerunning anything.

    The .xlsx flow only started writing a JSON report recently, so a sheet
    processed earlier has no payload at all — but it has everything needed to
    make one. The source column still holds the original text (the flow never
    overwrites it) and `tekst po redakcji` holds the result, so both sides can
    simply be measured again.

    What cannot be recovered is what the run *did*: how many rewrites were
    applied, and which sentences they touched. Those are marked unknown rather
    than guessed, and the report says so.
    """
    from .base import ItemOutcome, summarise
    from .xlsx_flow import REWRITE_COLUMN, resolve_column, _require_openpyxl

    openpyxl = _require_openpyxl()
    path = Path(path)
    workbook = openpyxl.load_workbook(str(path), data_only=True)
    sheet = workbook[sheet_name] if sheet_name else workbook.active
    column_index = resolve_column(sheet, column, header_row=header_row)
    rewrite_index = _header_column(sheet, REWRITE_COLUMN, header_row)
    review_index = _header_column(sheet, "do przeglądu", header_row)
    constraints_index = _header_column(sheet, "ograniczenia do regeneracji", header_row)

    outcomes: list[ItemOutcome] = []
    start_row = (header_row + 1) if header_row else 1
    for row_index in range(start_row, sheet.max_row + 1):
        before_text = _cell_text(sheet, row_index, column_index)
        if not before_text:
            continue
        after_text = _cell_text(sheet, row_index, rewrite_index) or before_text
        outcomes.append(
            _rebuild_outcome(
                name=f"wiersz {row_index}",
                before_text=before_text,
                after_text=after_text,
                review=_cell_text(sheet, row_index, review_index),
                constraints=_cell_text(sheet, row_index, constraints_index),
            )
        )

    if not outcomes:
        raise ValueError(
            f"Kolumna „{column}” w arkuszu „{sheet.title}” nie ma tekstu od wiersza "
            f"{start_row}. Wskaż kolumnę z odpowiedziami AI przez --column."
        )

    return {
        "flow": "xlsx",
        "rebuilt": True,
        "input_path": str(path),
        "output_path": str(path),
        "sheet": sheet.title,
        "source_column": column,
        "source_column_index": column_index,
        "settings": {},
        "layers": {},
        "summary": summarise(outcomes),
        "rows": [outcome.to_json() for outcome in outcomes],
    }


def _rebuild_outcome(
    *, name: str, before_text: str, after_text: str, review: str, constraints: str
):
    from .base import ItemOutcome
    from humanize_pl.gate import review_response

    before = detect_document(before_text)
    after = detect_document(after_text) if after_text != before_text else before
    outcome = ItemOutcome(
        name=name,
        words=before.word_count,
        signal_before=_score(before),
        signal_after=_score(after),
        findings_before=len(before.findings),
        findings_after=len(after.findings),
        findings_rewritable=before.rewritable_count,
        families=[row.family for row in before.families],
        family_counts_before={row.family: row.count for row in before.families},
        family_counts_after={row.family: row.count for row in after.families},
        metrics_before=dict(before.metrics),
        metrics_after=dict(after.metrics),
        examples=reconstruct_examples(before_text, after_text),
    )
    # The sheet already carries the verdict the run reached. Reusing it beats
    # recomputing under settings we would have to guess at.
    if review:
        outcome.needs_review = review.strip().upper() == "TAK"
        outcome.constraints = [
            line.lstrip("•").strip() for line in constraints.splitlines() if line.strip()
        ]
    else:
        verdict = review_response(after_text)
        outcome.needs_review = verdict.needs_revision
        outcome.constraints = verdict.prompt_constraints
    return outcome


def _score(diagnosis) -> float:
    calibration = diagnosis.calibration
    return calibration.calibrated_score if calibration else diagnosis.ai_signal_score


def _header_column(sheet, header: str, header_row: int | None) -> int | None:
    if not header_row:
        return None
    wanted = header.casefold()
    for cell in sheet[header_row]:
        if cell.value is not None and str(cell.value).strip().casefold() == wanted:
            return cell.column
    return None


def _cell_text(sheet, row_index: int, column_index: int | None) -> str:
    if column_index is None:
        return ""
    value = sheet.cell(row=row_index, column=column_index).value
    return str(value).strip() if value is not None else ""


def load_payload(source: str | Path) -> dict[str, Any]:
    """Read a flow payload from a run directory or a report file."""
    source = Path(source)
    path = source / REPORT_NAME if source.is_dir() else source
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono raportu przebiegu: {path}. Wskaż folder wyjściowy "
            f"z poprzedniego uruchomienia albo plik {REPORT_NAME}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "summary" not in payload:
        raise ValueError(f"{path} nie wygląda na raport przebiegu humanize-pl.")
    return payload


def needs_backfill(payload: dict[str, Any]) -> bool:
    items = payload.get("documents") or payload.get("rows") or []
    return any(
        item.get("status") == "ok" and any(field not in item for field in BACKFILLED_FIELDS)
        for item in items
    )


def backfill_payload(payload: dict[str, Any]) -> int:
    """Fill in the newer measurements by re-diagnosing documents on disk.

    Returns the number of items completed. Only missing fields are written:
    the numbers the original run reported stay exactly as they were reported.
    """
    if payload.get("flow") != "docx":
        return 0

    from humanize_pl.io.docx_io import docx_text

    input_directory = Path(str(payload.get("input_directory", "")))
    output_directory = Path(str(payload.get("output_directory", "")))
    completed = 0

    for item in payload.get("documents", []):
        if item.get("status") != "ok" or not any(
            field not in item for field in BACKFILLED_FIELDS
        ):
            continue
        source = input_directory / str(item.get("name", ""))
        if not source.is_file():
            continue
        try:
            before_text = docx_text(source)
            rewritten = output_directory / f"{Path(str(item['name'])).stem}_humanized.docx"
            after_text = docx_text(rewritten) if rewritten.is_file() else before_text
        except Exception:  # a document deleted or replaced since the run
            continue

        before = detect_document(before_text)
        after = detect_document(after_text) if after_text != before_text else before
        item.setdefault(
            "family_counts_before", {row.family: row.count for row in before.families}
        )
        item.setdefault(
            "family_counts_after", {row.family: row.count for row in after.families}
        )
        item.setdefault("metrics_before", dict(before.metrics))
        item.setdefault("metrics_after", dict(after.metrics))
        item.setdefault("findings_after", len(after.findings))
        item.setdefault("examples", reconstruct_examples(before_text, after_text))
        completed += 1

    return completed
