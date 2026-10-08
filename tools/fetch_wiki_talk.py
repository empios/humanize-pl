"""Download messages Polish Wikipedia users wrote to each other, as real emails.

A user talk page ("Dyskusja wikipedysty") is a mailbox: each section opens with
one person's message to another ("Witaj, … Pozdrawiam"), signed and dated.
Only that opening message is kept, without wiki markup, signature or date, and
records carry the page id rather than a username. Text from pl.wikipedia.org
is licensed CC BY-SA 4.0. The raw corpus stays local and gitignored.

Usage:
    python tools/fetch_wiki_talk.py --messages 150 --min-words 40 \
        --out docs_tests/corpus/general/wiki_talk.jsonl
    # a larger set for training, without the mailboxes held out above
    python tools/fetch_wiki_talk.py --messages 6000 --pages-per-letter 80 \
        --exclude docs_tests/corpus/general/wiki_talk.jsonl \
        --out docs_tests/corpus/general/wiki_talk_train.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import regex as re

API = "https://pl.wikipedia.org/w/api.php"
USER_AGENT = "humanize-pl reference-corpus builder (research use)"
USER_TALK = 3
PER_PAGE = 3  # messages from one mailbox, so a few talkative users do not dominate

_SECTION = re.compile(r"^==+[^=\n].*?==+\s*$", re.MULTILINE)
# Where a message ends: a signature link or the date "~~~~" expands to.
_SIGNATURE = re.compile(
    r"\[\[(?:Wikipedysta|Wikipedystka|Dyskusja wikipedysty|Dyskusja wikipedystki|User|"
    r"Specjalna:Wkład)[:/]|\d{1,2}:\d{2}, \d{1,2} \p{L}+ \d{4} \(CES?T\)",
    re.IGNORECASE,
)
# Form letters (welcome, missing file licence, a moved draft) are templates, or
# templates substituted into text that addresses anyone as "wysłałeś(-aś)".
_FORM_LETTER = re.compile(r"\{\{|\(-?a?ś\)|\(-a\)")


def call(params: dict[str, str | int], timeout: int) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    # Wikimedia answers 429 when asked too fast; it says how long to wait
    for _ in range(4):
        try:
            return _read(request, timeout)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(int(error.headers.get("Retry-After") or 30))
    return _read(request, timeout)


def _read(request: urllib.request.Request, timeout: int) -> dict:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def plain(wikitext: str) -> str:
    """Wiki markup to running text; empty for a form letter or markup the cleanup
    cannot read."""
    if _FORM_LETTER.search(wikitext):
        return ""
    text = re.sub(r"<!--.*?-->|<ref[^>]*?/>|<ref.*?</ref>", "", wikitext, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:Plik|File|Grafika|Image|Kategoria|Category):[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[:[^\]|]*:([^\]|]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    if re.search(r"[{}|\[\]<>]", text):
        return ""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.lstrip(":;*# ").strip() for line in block.splitlines()]
        paragraph = " ".join(line for line in lines if line)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).rstrip(" -–—")


def opening_messages(wikitext: str) -> list[str]:
    """The first message of each section, up to its author's signature."""
    messages = []
    for section in _SECTION.split(wikitext)[1:]:
        signature = _SIGNATURE.search(section)
        if signature:
            messages.append(plain(section[: signature.start()]))
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs_tests/corpus/general/wiki_talk.jsonl"))
    parser.add_argument("--messages", type=int, default=150)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--min-page-bytes", type=int, default=20000)
    parser.add_argument("--pages-per-letter", type=int, default=6, help="up to 500")
    parser.add_argument(
        "--exclude", type=Path, help="an earlier output whose mailboxes are skipped (a held-out set)"
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    held_out = set()
    if args.exclude:
        for line in args.exclude.open(encoding="utf-8"):
            held_out.add(int(json.loads(line)["id"].split("-")[1]))
    kept = 0
    with args.out.open("w", encoding="utf-8") as handle:
        # Pages from across the alphabet, not one letter's users
        for letter in string.ascii_uppercase:
            if kept >= args.messages:
                break
            pages = call(
                {"action": "query", "list": "allpages", "apnamespace": USER_TALK, "apfrom": letter,
                 "apminsize": args.min_page_bytes, "aplimit": args.pages_per_letter,
                 "apfilterredir": "nonredirects"},
                args.timeout,
            )["query"]["allpages"]
            time.sleep(args.delay)
            for page in pages:
                if page["pageid"] in held_out or kept >= args.messages:
                    continue
                content = call(
                    {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                     "pageids": page["pageid"]},
                    args.timeout,
                )["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
                time.sleep(args.delay)
                usable = [m for m in opening_messages(content) if len(m.split()) >= args.min_words]
                for index, message in enumerate(usable[:PER_PAGE]):
                    record = {"id": f"wikitalk-{page['pageid']}-{index}", "text": message}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
    print(f"{kept} messages kept in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
