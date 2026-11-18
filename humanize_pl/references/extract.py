"""Statute references and case numbers as they are written in a document.

Pure text work: nothing here touches the network. A reference is kept only
when its act can be named, either by a code abbreviation right after the
unit ("art. 6 § 1 k.c.") or by a Dz.U. citation later in the same sentence;
"art. 5 ustawy o zmianie" alone names no act anyone could check.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex as re
import yaml

from humanize_pl.sentence_splitter import split_sentences

ACTS_PATH = Path(__file__).resolve().parent.parent / "data" / "legal_acts.yaml"

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_NUMBERS = r"\d+[a-z]?(?:\s*(?:,|i|oraz)\s*\d+[a-z]?)*"
_ARTICLE_RE = re.compile(
    r"\b[Aa]rt\.\s*(?P<article>\d+)(?P<letter>[a-z])?(?![\d])"
    r"(?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹]+|\^\d+|\(\d+\))?"
    rf"(?:\s*§\s*(?P<paragraphs>{_NUMBERS}))?"
    rf"(?:\s*ust\.\s*(?P<passages>{_NUMBERS}))?"
    rf"(?:\s*pkt\.?\s*(?P<points>{_NUMBERS}))?"
)
_DZ_U_RE = re.compile(
    r"Dz\.\s*U\.\s*(?:z\s*)?(?P<year>\d{4})(?:\s*r\.)?\s*,?\s*"
    r"(?:Nr\s*\d+\s*,?\s*)?poz\.\s*(?P<position>\d+)"
)
# A case number has a division in Roman numerals before its repertory symbol,
# except at the Constitutional Tribunal and the KIO, whose symbols are a
# closed list. Any other capitals before a number are an invoice, contract or
# resolution number: "FV 12/2024", "NR 123/2020", "ZP 12/2024". Those are the
# document's own content and must never be sent to SAOS.
_UNDIVIDED_SYMBOLS = ("KIO", "Kpt", "SK", "Kp", "Pp", "Tw", "Ts", "K", "P", "U")
_CASE_RE = re.compile(
    r"(?<![\p{L}\d/])"
    r"(?:[IVX]{1,6}\s+[A-Z][A-Za-z]{0,4}|(?:" + "|".join(_UNDIVIDED_SYMBOLS) + r"))"
    r"\s+\d{1,6}/(?:\d{4}|\d{2})(?![\d/])"
)
# An article and a Dz.U. citation later in the sentence belong together only
# when no other article or other act stands between them.
# A full word and a period right before the citation end the sentence there
# ("art. 5 ustawy. Dz. U. …"); a one-letter abbreviation ("r.", "t.j.") does not.
_SENTENCE_END_RE = re.compile(r"\p{Ll}{3,}\.\s*$")
_OTHER_ACT_RE = re.compile(
    r"rozporządzeni|dyrektyw|Konstytucj|\bRODO\b|\bTFUE\b|\bTUE\b", re.IGNORECASE
)


@dataclass(frozen=True)
class LegalAct:
    key: str
    name: str
    eli: str
    forms: tuple[str, ...]


@dataclass(frozen=True)
class StatuteRef:
    text: str
    article: str
    paragraphs: tuple[str, ...]
    passages: tuple[str, ...]
    points: tuple[str, ...]
    act: str
    flattened_superscript: bool
    start: int
    end: int


@dataclass(frozen=True)
class CaseRef:
    text: str
    signature: str
    start: int
    end: int


@lru_cache(maxsize=None)
def _acts(path: Path) -> dict[str, LegalAct]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        row["key"]: LegalAct(row["key"], row["name"], row["eli"], tuple(row["forms"]))
        for row in raw.get("acts") or []
    }


def load_acts(path: Path = ACTS_PATH) -> dict[str, LegalAct]:
    return _acts(path)


@lru_cache(maxsize=None)
def _act_form_re(path: Path) -> tuple[re.Pattern, re.Pattern, dict[str, str]]:
    """Code forms, longest first so "k.p.c." is not read as "k.p."."""
    by_form: dict[str, str] = {}
    for act in _acts(path).values():
        for form in act.forms:
            by_form[form.casefold()] = act.key
    alternatives = sorted(by_form, key=len, reverse=True)
    body = "|".join(re.escape(form).replace(r"\.", r"\.\s*") for form in alternatives)
    pattern = re.compile(rf"\s*,?\s*(?P<form>{body})(?!\p{{L}})", re.IGNORECASE)
    anywhere = re.compile(rf"(?<!\p{{L}})(?:{body})(?!\p{{L}})", re.IGNORECASE)
    return pattern, anywhere, by_form


def extract_references(text: str, *, acts_path: Path = ACTS_PATH) -> list[StatuteRef | CaseRef]:
    found: list[StatuteRef | CaseRef] = []
    for start, sentence in _sentences(text):
        found.extend(_statutes(sentence, start, acts_path))
        found.extend(
            CaseRef(
                text=match.group(0),
                signature=_signature(match.group(0)),
                start=start + match.start(),
                end=start + match.end(),
            )
            for match in _CASE_RE.finditer(sentence)
        )
    return sorted(found, key=lambda ref: ref.start)


def _sentences(text: str) -> list[tuple[int, str]]:
    """Sentences with their offsets, cut from the original text.

    The splitter reads "Dz. U." and "r. Nr" as sentence ends, so citations
    are masked with lowercase letters of the same length (a capital would
    end the sentence at "t.j." or "r.") before splitting: offsets
    stay put, and the sentences are sliced from the unmasked text.
    """
    masked = _DZ_U_RE.sub(lambda match: "x" * len(match.group(0)), text)
    out: list[tuple[int, str]] = []
    offset = 0
    for paragraph in masked.split("\n"):
        cursor = 0
        for sentence in split_sentences(paragraph):
            position = paragraph.find(sentence, cursor)
            if position < 0:
                continue
            start = offset + position
            out.append((start, text[start:start + len(sentence)]))
            cursor = position + len(sentence)
        offset += len(paragraph) + 1
    return out


def _statutes(sentence: str, start: int, acts_path: Path) -> list[StatuteRef]:
    form_re, code_anywhere, by_form = _act_form_re(acts_path)
    refs: list[StatuteRef] = []
    for match in _ARTICLE_RE.finditer(sentence):
        end = match.end()
        code = form_re.match(sentence, end)
        if code:
            act = by_form[" ".join(code.group("form").split()).casefold().replace(". ", ".")]
            end = code.end()
        else:
            citation = _DZ_U_RE.search(sentence, match.end())
            if citation is None:
                continue
            between = sentence[match.end():citation.start()]
            if (
                _ARTICLE_RE.search(between)
                or _SENTENCE_END_RE.search(between)
                or _OTHER_ACT_RE.search(between)
                or code_anywhere.search(between)
            ):
                continue
            act = f"DU/{citation.group('year')}/{citation.group('position')}"
        article, flattened = _article(match)
        refs.append(
            StatuteRef(
                text=sentence[match.start():end].strip(),
                article=article,
                paragraphs=_numbers(match.group("paragraphs")),
                passages=_numbers(match.group("passages")),
                points=_numbers(match.group("points")),
                act=act,
                flattened_superscript=flattened,
                start=start + match.start(),
                end=start + end,
            )
        )
    return refs


def _article(match: re.Match) -> tuple[str, bool]:
    """ELI's form of the article number, and whether a superscript may have
    been written flat ("4491" for 449¹)."""
    digits = match.group("article")
    letter = match.group("letter") or ""
    superscript = match.group("sup")
    if superscript:
        index = superscript.translate(_SUPERSCRIPTS).strip("^()")
        return f"{digits}{letter}_{index}", False
    return f"{digits}{letter}", not letter and len(digits) >= 2


def _signature(written: str) -> str:
    """Single spaces and a two-digit year, the form SAOS stores."""
    return re.sub(r"/(?:19|20)(\d{2})$", r"/\1", " ".join(written.split()))


def _numbers(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(re.findall(r"\d+[a-z]?", value))
