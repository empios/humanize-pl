"""Download real Polish customer reviews as human text for the general route.

The general route was measured only on emails Claude wrote; these reviews are
people's own writing, typos and all. The texts come from the KLEJ benchmark's
Allegro reviews (huggingface.co/datasets/legacy-datasets/allegro_reviews),
licensed CC BY-SA 4.0. The raw corpus stays local and gitignored.

Usage:
    python tools/fetch_allegro_reviews.py --split test --min-words 40 \
        --out docs_tests/corpus/general/allegro_reviews.jsonl
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

API = "https://datasets-server.huggingface.co/rows"
DATASET = "legacy-datasets/allegro_reviews"
USER_AGENT = "humanize-pl reference-corpus builder (research use)"
PAGE = 100  # the rows API's maximum


def fetch_rows(split: str, offset: int, timeout: int) -> tuple[list[dict], int]:
    params = {"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": PAGE}
    request = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    # The rows API answers 429 under load; it says how long to wait
    for _ in range(4):
        try:
            return _rows(request, timeout)
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            time.sleep(int(error.headers.get("Retry-After") or 30))
    return _rows(request, timeout)


def _rows(request: urllib.request.Request, timeout: int) -> tuple[list[dict], int]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [row["row"] for row in payload["rows"]], payload["num_rows_total"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("docs_tests/corpus/general/allegro_reviews.jsonl")
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = offset = 0
    total = None
    with args.out.open("w", encoding="utf-8") as handle:
        while total is None or offset < total:
            rows, total = fetch_rows(args.split, offset, args.timeout)
            for index, row in enumerate(rows, start=offset):
                text = " ".join(row["text"].split())
                if len(text.split()) < args.min_words:
                    continue
                record = {"id": f"allegro-{args.split}-{index}", "text": text}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            offset += PAGE
            time.sleep(args.delay)
    print(f"{kept} reviews of {total} kept in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
