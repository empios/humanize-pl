"""A folder of DOCX files in, pseudonymized copies, a key and a report out.

The report carries counts, labels' kinds and where data remained, never a
value: it is the file people forward, while the key stays with its owner.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from humanize_pl.nlp.morfeusz import try_load_morfeusz

from .detect import MorfeuszRequired
from .docx import DocumentResult, pseudonymize_docx
from .labels import Pseudonymizer, load_key, save_key

REPORT_NAME = "pseudonimizacja.json"
KEY_NAME = "klucz.json"
SUFFIX = "_pseudonimizowany"


def pseudonymize_folder(
    folder: Path, output: Path, *, key_path: Path | None = None
) -> dict[str, Any]:
    from humanize_pl.flows.docx_flow import docx_files

    if try_load_morfeusz() is None:
        raise MorfeuszRequired(
            "Pseudonimizacja wymaga Morfeusza2; bez niego osoby nie zostałyby znalezione. "
            "Zainstaluj: pip install '.[morfeusz]'."
        )
    key_path = key_path or output / KEY_NAME
    pseudonymizer = Pseudonymizer(load_key(key_path) if key_path.exists() else None)
    results: list[DocumentResult] = []
    for source in docx_files(folder):
        target = output / f"{source.stem}{SUFFIX}.docx"
        try:
            results.append(pseudonymize_docx(source, target, pseudonymizer))
        except MorfeuszRequired:
            raise
        except Exception as exc:  # one unreadable file must not stop the batch
            # The exception's message may quote the document, so only its type.
            results.append(
                DocumentResult(source.name, "failed", leaks=[f"błąd: {type(exc).__name__}"])
            )
    if pseudonymizer.key()["etykiety"]:
        save_key(pseudonymizer.key(), key_path)
    statuses = Counter(result.status for result in results)
    replaced: Counter[str] = Counter()
    for result in results:
        replaced.update(result.counts)
    report = {
        "wersja": 1,
        "dokumenty": [result.to_json() for result in results],
        "podsumowanie": {
            "documents": len(results),
            "complete": statuses["complete"],
            "incomplete": statuses["incomplete"],
            "failed": statuses["failed"],
            "replaced": dict(replaced),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
