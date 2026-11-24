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
from .docx import DocumentResult, package_texts, pseudonymize_docx
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
    folder, output = folder.resolve(), output.resolve()
    if output == folder or folder in output.parents:
        raise ValueError(
            "Katalog wyjściowy nie może leżeć w katalogu z dokumentami: kopie trafiłyby "
            "do następnego przebiegu."
        )
    key_path = key_path or output / KEY_NAME
    try:
        pseudonymizer = Pseudonymizer(load_key(key_path) if key_path.exists() else None)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Nie można użyć pliku klucza {key_path.name}: {exc}") from exc
    sources = docx_files(folder)
    # Everyone named anywhere in the run is known before any copy is made, so a
    # surname met in one file before its full name appears in another is found.
    for source in sources:
        try:
            pseudonymizer.learn([text for _, text in package_texts(source)])
        except Exception:  # an unreadable file is reported in the main pass
            continue
    results: list[DocumentResult] = []
    for source in sources:
        alias = pseudonymizer.file_alias(source.name)
        target = output / f"{alias}{SUFFIX}.docx"
        try:
            results.append(pseudonymize_docx(source, target, pseudonymizer, name=alias))
        except MorfeuszRequired:
            raise
        except Exception as exc:  # one unreadable file must not stop the batch
            # The exception's message may quote the document, so only its type.
            results.append(
                DocumentResult(alias, "failed", leaks=[f"błąd: {type(exc).__name__}"])
            )
    if results:
        save_key(pseudonymizer.key(), key_path)
    statuses = Counter(result.status for result in results)
    replaced: Counter[str] = Counter()
    for result in results:
        replaced.update(result.counts)
    report = {
        "wersja": 1,
        # Only the file name: the report travels, a local path would not stay local.
        "klucz": key_path.name if results else None,
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
