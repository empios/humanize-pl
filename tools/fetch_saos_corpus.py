"""Download a human Polish legal-writing reference corpus from SAOS.

SAOS (saos.org.pl) publishes Polish court judgments under an open API. Court
reasoning ("uzasadnienie") is running human legal prose, which is what the
detection layer needs as a baseline — its thresholds are meaningless without
something human to compare against.

Judgments are official documents and fall outside copyright protection under
art. 4 of the Polish copyright act. Only derived statistics are committed to
the repository; the raw corpus stays local and gitignored.

Usage:
    python tools/fetch_saos_corpus.py --pages 40 --start-date 2018-01-01 \
        --court-type COMMON --out docs_tests/corpus/saos.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from humanize_pl.corpus.normalize import is_usable, normalize_judgment

API = "https://www.saos.org.pl/api/dump/judgments"
USER_AGENT = "humanize-pl reference-corpus builder (research use)"
# SAOS publishes no rate limit; this is deliberate politeness, not a requirement.
DELAY_SECONDS = 1.0


def fetch_page(
    page_number: int,
    *,
    page_size: int,
    start_date: str | None,
    end_date: str | None,
    timeout: int,
) -> list[dict]:
    params: dict[str, str] = {
        "pageSize": str(page_size),
        "pageNumber": str(page_number),
    }
    if start_date:
        params["judgmentStartDate"] = start_date
    if end_date:
        params["judgmentEndDate"] = end_date

    request = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("items") or []


def existing_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    found: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                found.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs_tests/corpus/saos.jsonl"))
    parser.add_argument("--pages", type=int, default=20, help="Number of API pages to fetch")
    parser.add_argument("--page-size", type=int, default=100, help="10-100, SAOS limit")
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--start-date", default="2018-01-01", help="yyyy-MM-dd or empty")
    parser.add_argument("--end-date", default=None, help="yyyy-MM-dd")
    parser.add_argument(
        "--court-type",
        default="COMMON",
        help="Filter client-side: COMMON, SUPREME, ADMINISTRATIVE, ... or ALL",
    )
    parser.add_argument("--min-words", type=int, default=150)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_ids(args.out)
    if seen:
        print(f"Wznowienie: {len(seen)} dokumentów już pobranych", file=sys.stderr)

    written = 0
    skipped = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for offset in range(args.pages):
            page_number = args.start_page + offset
            try:
                items = fetch_page(
                    page_number,
                    page_size=args.page_size,
                    start_date=args.start_date or None,
                    end_date=args.end_date,
                    timeout=args.timeout,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"Strona {page_number}: {type(exc).__name__}: {exc}", file=sys.stderr)
                break

            if not items:
                print(f"Strona {page_number}: brak wyników, koniec", file=sys.stderr)
                break

            for item in items:
                judgment_id = item.get("id")
                if judgment_id is None or judgment_id in seen:
                    continue
                if args.court_type != "ALL" and item.get("courtType") != args.court_type:
                    continue

                text = normalize_judgment(item.get("textContent") or "")
                if not is_usable(text, min_words=args.min_words):
                    skipped += 1
                    continue

                seen.add(judgment_id)
                handle.write(
                    json.dumps(
                        {
                            "id": judgment_id,
                            "court_type": item.get("courtType"),
                            "judgment_type": item.get("judgmentType"),
                            "judgment_date": item.get("judgmentDate"),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

            handle.flush()
            print(
                f"Strona {page_number}: zapisane {written}, odrzucone {skipped}",
                file=sys.stderr,
            )
            time.sleep(args.delay)

    print(f"Zapisano {written} dokumentów do {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
