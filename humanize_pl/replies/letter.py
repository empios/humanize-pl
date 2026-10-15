"""Reading an incoming letter: its head, its instruction and its facts.

Every official letter has the same build whoever sends it: the sender and the
case reference on top, the title of the act, the ruling or the demand, the
legal basis, and an instruction ("pouczenie") naming the remedy, its term and
where to file it. These functions find those parts by their form; which act or
remedy they are is decided against the catalogue in `analysis.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import regex as re

from humanize_pl.replies.catalogue import Term

# The head is where the sender and the title stand; past it a court or an
# authority is only being quoted.
HEAD_LINES = 15
TITLE_MAX = 90

NUMBERS = {
    "jednego": 1,
    "dwóch": 2,
    "dwu": 2,
    "trzech": 3,
    "czterech": 4,
    "pięciu": 5,
    "siedmiu": 7,
    "dziesięciu": 10,
    "czternastu": 14,
    "dwudziestu jeden": 21,
    "trzydziestu": 30,
}
# "w terminie tygodniowym", "w terminie 14-dniowym": the number and the unit in one adjective.
ADJECTIVES = {
    "tygodniowym": (1, "week"),
    "dwutygodniowym": (2, "week"),
    "miesięcznym": (1, "month"),
    "siedmiodniowym": (7, "day"),
    "czternastodniowym": (14, "day"),
    "trzydziestodniowym": (30, "day"),
}
UNIT_WORDS = {
    "dni": "day",
    "dnia": "day",
    "tygodni": "week",
    "tygodnia": "week",
    "tygodnie": "week",
    "miesięcy": "month",
    "miesiąca": "month",
    "miesiące": "month",
}
MONTHS = (
    "stycznia",
    "lutego",
    "marca",
    "kwietnia",
    "maja",
    "czerwca",
    "lipca",
    "sierpnia",
    "września",
    "października",
    "listopada",
    "grudnia",
)

_INSTRUCTION = re.compile(r"(?im)^\s*pouczeni[ea]\b")
_REMEDY_VERB = re.compile(
    r"(?i)przysługuje|można wnieść|może wnieść|można złożyć|może złożyć|wnosi się|należy wnieść"
)
_TERM = re.compile(
    r"(?i)(?:w\s+terminie|w\s+ciągu|w\s+okresie)\s+"
    r"(?:(?P<adj>\w+(?:owym|cznym)|\d+-dniowym)"
    r"|(?:(?P<num>\d+|dwudziestu\s+jeden|\p{L}+)\s+)?"
    r"(?P<unit>dni|dnia|tygodni|tygodnia|tygodnie|miesięcy|miesiąca|miesiące)\b)"
)
_START = re.compile(
    r"(?i)\bod\s+(?:dnia\s+)?(?:(?:jej|jego|ich|niniejszej|niniejszego)\s+)?(?P<start>"
    r"(?:doręczenia|otrzymania|ogłoszenia|przyjęcia|uprawomocnienia|zawiadomienia|dokonania)"
    r"\b[^,.;:]{0,60})"
)
_PLACE_WORDS = r"(?:[\p{Lu}][\p{L}.-]*|w|we|i|ds\.|do|spraw|z|nr|\d+)(?!\p{L})"
_ADDRESSEE = re.compile(r"\bdo\s+(?P<name>[\p{Lu}][\p{L}.-]*(?:\s+" + _PLACE_WORDS + r"){0,9})")
_VIA = re.compile(
    r"\b[Zz]a\s+pośrednictwem\s+(?P<name>[\p{Lu}][\p{L}.-]*(?:\s+" + _PLACE_WORDS + r"){0,9})"
)

# Court reference ("I C 123/24", "XII GC 45/23") and an authority's file mark after its label.
_COURT_REFERENCE = re.compile(
    r"\b[IVXLC]{1,5}\s+[\p{Lu}][\p{L}]{0,4}(?:-[\p{L}]+)?\s+\d{1,6}/\d{2,4}\b"
)
_FILE_MARK = re.compile(
    r"(?i)(?:sygn\.\s*akt|sygnatura(?:\s+akt)?|znak(?:\s+sprawy|\s+pisma)?|nasz\s+znak|"
    r"nr\s+sprawy|numer\s+sprawy|sprawa\s+nr|nr\s+decyzji|numer\s+decyzji)"
    r"\s*:?\s*(?P<mark>[\p{L}\d][\w./\-–]*\d[\w./\-–]*)"
)
_NUMERIC_DATE = re.compile(r"\b(?P<d>\d{1,2})[.\-/](?P<m>\d{1,2})[.\-/](?P<y>\d{4})\b")
_WORD_DATE = re.compile(r"(?i)\b(?P<d>\d{1,2})\s+(?P<m>" + "|".join(MONTHS) + r")\s+(?P<y>\d{4})\b")
_LETTER_DATE = re.compile(
    r"(?im)^\s*[\p{Lu}][\p{L} .-]{1,40},\s*(?:dnia\s+|dn\.\s*)?"
    r"(?P<date>\d{1,2}(?:[.\-/]\d{1,2}[.\-/]|\s+\p{L}+\s+)\d{4})"
)
_AMOUNT = re.compile(r"\b\d{1,3}(?:[ .]\d{3})*(?:,\d{2})?\s*(?:zł|PLN)\b")
_IDENTIFIERS = {
    "NIP": re.compile(
        r"(?i)\bNIP\s*:?\s*(\d{3}-?\d{3}-?\d{2}-?\d{2}|\d{3}-?\d{2}-?\d{2}-?\d{3})\b"
    ),
    "PESEL": re.compile(r"(?i)\bPESEL\s*:?\s*(\d{11})\b"),
    "KRS": re.compile(r"(?i)\bKRS\s*:?\s*(\d{10})\b"),
    "REGON": re.compile(r"(?i)\bREGON\s*:?\s*(\d{9}|\d{14})\b"),
}
_ARTICLE = re.compile(
    r"(?i)\bart\.\s*\d+[\p{L}\d⁰¹²³⁴⁵⁶⁷⁸⁹]*(?:\s*§\s*\d+[\p{L}]?)?"
    r"(?:\s*(?:pkt|ust\.)\s*\d+[\p{L}]?)?"
)


@dataclass(frozen=True)
class Remedy:
    """One remedy the instruction names, with what it says about it."""

    phrase: str
    term: Term | None = None
    addressee: str = ""
    via: str = ""
    quote: str = ""


@dataclass(frozen=True)
class Facts:
    references: tuple[str, ...] = ()
    letter_date: str = ""
    dates: tuple[str, ...] = ()
    amounts: tuple[str, ...] = ()
    identifiers: tuple[tuple[str, str], ...] = ()
    articles: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "references": list(self.references),
            "letter_date": self.letter_date,
            "dates": list(self.dates),
            "amounts": list(self.amounts),
            "identifiers": dict(self.identifiers),
            "articles": list(self.articles),
        }


@dataclass(frozen=True)
class Letter:
    text: str
    lowered: str
    head: str
    titles: tuple[str, ...]
    instruction: str
    facts: Facts = field(default_factory=Facts)


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(" ".join(v.split()) for v in values))


def head_of(text: str, headers: tuple[str, ...]) -> str:
    """The lines above the act's title: there the sender names itself. Below it,
    in the reasons and the instruction, other offices and courts are only named."""
    lines = [line for line in text.splitlines() if line.strip()][:HEAD_LINES]
    for i, line in enumerate(lines):
        if is_title(" ".join(line.split()).casefold(), headers):
            return "\n".join(lines[:i])
    return "\n".join(lines)


def is_title(line: str, headers: tuple[str, ...]) -> bool:
    return len(line) <= TITLE_MAX and any(
        re.match(re.escape(h) + r"(?!\p{L})", line) for h in headers
    )


def titles_of(text: str) -> tuple[str, ...]:
    """Short lines that can be a title ("DECYZJA nr 12/2024", "Wezwanie do zapłaty")."""
    return tuple(
        " ".join(line.split()).casefold()
        for line in text.splitlines()[: HEAD_LINES * 3]
        if line.strip() and len(line.strip()) <= TITLE_MAX
    )


def instruction_of(text: str) -> str:
    """The instruction section; without one, the sentences that grant a remedy."""
    found = _INSTRUCTION.search(text)
    if found:
        return text[found.start() :]
    sentences = re.split(r"(?<=[.!?])\s+(?=[\p{Lu}])", text)
    return " ".join(s for s in sentences if _REMEDY_VERB.search(s))


def term_in(window: str) -> Term | None:
    """The first time limit a passage states, with the event it runs from."""
    found = _TERM.search(window)
    if not found:
        return None
    start = _START.search(window, found.end())
    begins = " ".join(start["start"].split()) if start and start.start() - found.end() < 40 else ""
    if found["adj"]:
        adjective = found["adj"].casefold()
        if adjective.endswith("-dniowym"):
            value, unit = int(adjective.split("-")[0]), "day"
        elif adjective in ADJECTIVES:
            value, unit = ADJECTIVES[adjective]
        else:
            return None
    else:
        unit = UNIT_WORDS[found["unit"].casefold()]
        number = (found["num"] or "").casefold()
        number = " ".join(number.split())
        if not number:
            value = 1
        elif number.isdigit():
            value = int(number)
        elif number in NUMBERS:
            value = NUMBERS[number]
        else:
            return None
    return Term(value, unit, begins or "doręczenia")


def _name(found) -> str:
    """An office's name without the connecting word the match stopped on."""
    words = found["name"].split() if found else []
    while words and words[-1].islower():
        words.pop()
    # A sentence's full stop, not an abbreviation's ("ds.").
    if words and len(words[-1]) > 4:
        words[-1] = words[-1].rstrip(".")
    return " ".join(words)


def remedies_in(instruction: str, phrases: tuple[str, ...]) -> list[Remedy]:
    """Each remedy phrase the instruction uses, the longest where they overlap,
    read up to the next one: its term, addressee and the body that forwards it."""
    lowered = instruction.casefold()
    spans = []
    for phrase in sorted(set(phrases), key=len, reverse=True):
        for found in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lowered):
            if not any(s <= found.start() and found.end() <= e for _, s, e in spans):
                spans.append((phrase, found.start(), found.end()))
    spans.sort(key=lambda span: span[1])
    remedies = []
    for i, (phrase, start, _) in enumerate(spans):
        stop = spans[i + 1][1] if i + 1 < len(spans) else len(instruction)
        window = instruction[start : min(stop, start + 400)]
        addressee, via = _ADDRESSEE.search(window), _VIA.search(window)
        remedies.append(
            Remedy(
                phrase=phrase,
                term=term_in(window),
                addressee=_name(addressee),
                via=_name(via),
                quote=" ".join(window.split())[:300],
            )
        )
    return remedies


def facts_of(text: str) -> Facts:
    """What a reply must quote unchanged: references, dates, amounts, identifiers."""
    letter_date = _LETTER_DATE.search(text)
    return Facts(
        references=_unique(
            [m.group(0) for m in _COURT_REFERENCE.finditer(text)]
            + [m["mark"].rstrip(".,") for m in _FILE_MARK.finditer(text)]
        ),
        letter_date=" ".join(letter_date["date"].split()) if letter_date else "",
        dates=_unique(
            [m.group(0) for m in _NUMERIC_DATE.finditer(text)]
            + [m.group(0) for m in _WORD_DATE.finditer(text)]
        ),
        amounts=_unique(m.group(0) for m in _AMOUNT.finditer(text)),
        identifiers=tuple(
            (name, m.group(1))
            for name, pattern in _IDENTIFIERS.items()
            if (m := pattern.search(text))
        ),
        articles=_unique(m.group(0) for m in _ARTICLE.finditer(text)),
    )


def read(text: str, headers: tuple[str, ...] = ()) -> Letter:
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    return Letter(
        text=text,
        lowered=text.casefold(),
        head=head_of(text, headers).casefold(),
        titles=titles_of(text),
        instruction=instruction_of(text),
        facts=facts_of(text),
    )
