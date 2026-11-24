"""Personal data in a text: people by Morfeusz name categories, identifiers by
their shape and checksums.

Morfeusz marks "Kraków" and "Bank" as possible surnames too, so a capitalised
word alone is never a person. A person is a first name next to a surname, a
surname after a title or a procedural role ("Panu Nowakowi", "powód Nowak"),
or a surname already known to belong to someone. Identifiers come first when
spans overlap: "ul. Kowalskiego 5" is an address, not Mr Kowalski.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

from humanize_pl.nlp.morfeusz import MorfeuszAnalyzer, try_load_morfeusz


class MorfeuszRequired(RuntimeError):
    """People cannot be found without Morfeusz, and a run that silently
    skipped them would hand out copies that still name everyone."""


@dataclass(frozen=True)
class Entity:
    kind: str  # person | pesel | id_card | passport | iban | phone | email | address | kw
    start: int
    end: int
    key: str  # who or what it is: "jan|kowalski", "|kowalski", digits, address


_PESEL_WEIGHTS = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
_PESEL_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_PESEL_LABEL_RE = re.compile(r"PESEL\W{0,15}$", re.IGNORECASE)
_ID_CARD_RE = re.compile(
    r"(?i:dow[oó]d\w*\s+osobist\w*|seri[aię]\s+i\s+num\w*)[^\n]{0,30}?"
    r"(?P<value>(?<![A-Z])[A-Z]{3}\s?\d{6}(?!\d))"
)
_PASSPORT_RE = re.compile(r"(?i:paszport\w*)[^\n]{0,30}?(?P<value>(?<![A-Z])[A-Z]{2}\s?\d{7}(?!\d))")
_IBAN_RE = re.compile(r"(?<![\dA-Z])(?:PL\s?)?\d{2}(?:\s?\d{4}){6}(?!\d)")
# Three groups of three digits are also how amounts are written ("100 000 000
# zł"), so such a match is not a phone when a digit group comes before it or a
# currency, a decimal part or a share count after it.
_NOT_AN_AMOUNT = (
    r"(?<!\d[ \u00a0.])(?![ \u00a0]\d{3})"
    r"(?!\s?(?:,\d|zł|PLN|EUR|USD|akcj|udział|%|gr\b))"
)
_PHONE_RE = re.compile(
    r"(?<![\d+])\+48\s?\d{3}[ -]?\d{3}[ -]?\d{3}(?!\d)"
    r"|(?<![\d+])(?<!\d[ \u00a0.])\d{3}[ -]\d{3}[ -]\d{3}(?!\d)" + _NOT_AN_AMOUNT
    + r"|(?<![\d(])(?:\(\d{2}\)|\d{2})\s?\d{3}[ -]\d{2}[ -]\d{2}(?!\d)"
    r"|(?i:\btel\.?|\btelefon\w*)\s*(?:nr\.?\s*)?:?\s*(?P<value>\d{9})(?!\d)"
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HOUSE = r"\d+[A-Za-z]?(?:\s*/\s*\d+[A-Za-z]?)?(?:\s*(?:m\.|lok\.)\s*\d+[A-Za-z]?)?"
# "zam." marks a residence without being part of it, so it stays in the copy.
_ADDRESS_RE = re.compile(
    r"(?:(?i:\b(?:ul|al|pl|os)\.|\bulic[ay]\b|\balej[ai]\b)\s*|(?<=(?i:\bzam\.)\s{0,3}))"
    r"(?:(?:\p{Lu}[\p{L}\-]*\.?|[IVXL]+|\d+(?=\s+\p{L}))\s+){1,5}?"
    + _HOUSE
    + r"(?!\d)"
)
# A postal code is personal data even without the town after it.
_POSTAL_RE = re.compile(
    r"(?<![\d-])\d{2}-\d{3}(?![\d-])(?:\s+\p{Lu}[\p{L}\-]+(?:\s+\p{Lu}[\p{L}\-]+)?)?"
)
# Format alone: a wrong check-digit rule would leave real numbers in place.
_KW_RE = re.compile(r"\b[A-Z]{2}\d[A-Z]/\d{8}/\d\b")

_WORD_RE = re.compile(r"\p{L}+(?:-\p{L}+)*")
# Titles: "Pan" has a first-name reading, so it is never read as one.
_TITLES = frozenset(
    {
        "pan", "pana", "panu", "panem", "panie", "pani", "panią", "p",
        "adw", "mec", "pr", "dr", "prof", "mgr", "inż",
    }
)
# What may stand between a title or role and the surname: "adw. Kowalski".
_AFTER_TITLE_RE = re.compile(r"\.?\s+")
# "Uniwersytet im. Adama Mickiewicza": a patron is part of an institution's name.
_PATRON_RE = re.compile(r"\bim\.\s*$", re.IGNORECASE)
# "w Krakowie Jan Kowalski": a place after a preposition never starts a name,
# though Morfeusz also reads "Krakowie" as a first name.
_PLACE_PREPOSITIONS = frozenset(
    {"w", "we", "z", "ze", "do", "na", "przy", "pod", "od", "koło", "spod", "nad"}
)
# Surname before first name only in listings: "Kowalski Jan, zam. …".
_LISTING_END_RE = re.compile(r"\s*(?:[,;(\n]|$)")
# Procedural and contractual roles a surname may follow ("powód Nowak").
_ROLE_LEMMAS = frozenset(
    {
        "powód", "powódka", "pozwany", "pozwana", "pozwać", "świadek", "oskarżony",
        "oskarżona", "oskarżyć", "wnioskodawca", "wnioskodawczyni", "uczestnik",
        "uczestniczka", "pełnomocnik", "adwokat", "radca", "sędzia", "notariusz",
        "komornik", "dłużnik", "dłużniczka", "wierzyciel", "wierzycielka",
    }
)


@dataclass(frozen=True)
class _Word:
    text: str
    start: int
    end: int
    capital: bool
    dotted: bool  # an initial written "J."


def find_entities(text: str, *, known_people: frozenset[str] = frozenset()) -> list[Entity]:
    """Every piece of personal data, left to right, without overlaps."""
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        raise MorfeuszRequired("Pseudonimizacja wymaga Morfeusza2 (pip install '.[morfeusz]').")
    accepted: list[Entity] = []
    for candidate in [*_identifiers(text), *_people(text, morfeusz, known_people)]:
        if all(candidate.end <= taken.start or candidate.start >= taken.end for taken in accepted):
            accepted.append(candidate)
    return sorted(accepted, key=lambda entity: entity.start)


def _identifiers(text: str) -> list[Entity]:
    found: list[Entity] = []
    for match in _KW_RE.finditer(text):
        found.append(Entity("kw", match.start(), match.end(), match.group(0)))
    for match in _PESEL_RE.finditer(text):
        labelled = _PESEL_LABEL_RE.search(text[max(0, match.start() - 20):match.start()])
        if labelled or _pesel_valid(match.group(0)):
            found.append(Entity("pesel", match.start(), match.end(), match.group(0)))
    for kind, pattern in (("id_card", _ID_CARD_RE), ("passport", _PASSPORT_RE)):
        for match in pattern.finditer(text):
            value = match.group("value")
            found.append(
                Entity(kind, match.start("value"), match.end("value"), value.replace(" ", ""))
            )
    for match in _IBAN_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        found.append(Entity("iban", match.start(), match.end(), digits))
    for match in _PHONE_RE.finditer(text):
        start, end = (match.start("value"), match.end("value")) if match.group("value") else (
            match.start(), match.end()
        )
        found.append(Entity("phone", start, end, re.sub(r"\D", "", text[start:end])[-9:]))
    for match in _EMAIL_RE.finditer(text):
        found.append(Entity("email", match.start(), match.end(), match.group(0).lower()))
    for pattern in (_ADDRESS_RE, _POSTAL_RE):
        for match in pattern.finditer(text):
            key = " ".join(match.group(0).split()).lower()
            found.append(Entity("address", match.start(), match.end(), key))
    return found


def _pesel_valid(digits: str) -> bool:
    total = sum(int(digit) * weight for digit, weight in zip(digits, _PESEL_WEIGHTS))
    return (10 - total % 10) % 10 == int(digits[10])


def _people(text: str, morfeusz: MorfeuszAnalyzer, known: frozenset[str]) -> list[Entity]:
    words = [
        _Word(
            match.group(0),
            match.start(),
            match.end(),
            match.group(0)[0].isupper(),
            text[match.end():match.end() + 1] == "." and len(match.group(0)) == 1,
        )
        for match in _WORD_RE.finditer(text)
    ]
    surnames = set(known)
    found: list[Entity] = []
    index = 0
    while index < len(words):
        word = words[index]
        if not word.capital or _PATRON_RE.search(text[:word.start]):
            index += 1
            continue
        taken = _name(text, words, index, morfeusz)
        if taken is not None:
            end_index, key = taken
            found.append(Entity("person", word.start, words[end_index].end, key))
            surnames.add(key.split("|", 1)[1])
            index = end_index + 1
            continue
        if not _is_title(word):
            last = _lemmas(morfeusz, word.text, "nazwisko")
            unknown = not _analyses(morfeusz, word.text)
            previous = words[index - 1] if index else None
            after_role = (
                previous is not None
                and _AFTER_TITLE_RE.fullmatch(text[previous.end:word.start]) is not None
                and _is_role(morfeusz, previous)
                and (last or unknown)
            )
            known_surname = (
                bool(set(last) & surnames)
                or word.text.lower() in surnames
                or (unknown and _same_stem(word.text.lower(), surnames))
            )
            if after_role or known_surname:
                surname = last[0] if last else word.text.lower()
                found.append(Entity("person", word.start, word.end, f"|{surname}"))
                surnames.add(surname)
        index += 1
    return found


def _name(
    text: str, words: list[_Word], index: int, morfeusz: MorfeuszAnalyzer
) -> tuple[int, str] | None:
    """(index of the last word, key) when a name starts at `index`."""
    word = words[index]
    if _is_role(morfeusz, word):
        return None
    previous = words[index - 1] if index else None
    if (
        previous is not None
        and previous.text.lower() in _PLACE_PREPOSITIONS
        and _lemmas(morfeusz, word.text, "nazwa_geograficzna")
    ):
        return None

    def next_capital(position: int) -> _Word | None:
        if position + 1 >= len(words):
            return None
        candidate = words[position + 1]
        if candidate.capital and _adjacent(text, words[position], candidate):
            return candidate
        return None

    def surname_like(candidate: _Word) -> bool:
        return bool(_lemmas(morfeusz, candidate.text, "nazwisko")) or not _analyses(
            morfeusz, candidate.text
        )

    def surname_of(candidate: _Word) -> str:
        last = _lemmas(morfeusz, candidate.text, "nazwisko")
        return last[0] if last else candidate.text.lower()

    first = _lemmas(morfeusz, word.text, "imię")
    second = next_capital(index)
    if second is None:
        return None
    if first or word.dotted:
        given = first[0] if first else f"{word.text.lower()}."
        third = next_capital(index + 1)
        if third is not None and _lemmas(morfeusz, second.text, "imię") and surname_like(third):
            return index + 2, f"{given}|{surname_of(third)}"
        if surname_like(second):
            return index + 1, f"{given}|{surname_of(second)}"
    reversed_first = _lemmas(morfeusz, second.text, "imię")
    reversed_last = _lemmas(morfeusz, word.text, "nazwisko")
    listing = word.text.isupper() or _LISTING_END_RE.match(text, second.end) is not None
    if reversed_last and reversed_first and not first and listing:
        return index + 1, f"{reversed_first[0]}|{reversed_last[0]}"
    return None


def _same_stem(word: str, surnames: set[str]) -> bool:
    """An inflected form of a surname Morfeusz does not know: "Kowalczykowej"
    after "Kowalczykowa"."""
    return any(
        len(surname) >= 6 and word[: len(surname) - 2] == surname[: len(surname) - 2]
        for surname in surnames
    )


def _adjacent(text: str, left: _Word, right: _Word) -> bool:
    between = text[left.end:right.start]
    return between.strip(".") == "" or between.isspace() or (
        left.dotted and between.startswith(".") and between[1:].isspace()
    )


def _analyses(morfeusz: MorfeuszAnalyzer, word: str) -> list:
    """Readings of a word, with capitals read as a capitalised name."""
    if word.isupper() and len(word) > 1:
        word = word.capitalize()
    return morfeusz.analyses(word)


def _lemmas(morfeusz: MorfeuszAnalyzer, word: str, category: str) -> list[str]:
    return sorted(
        {
            reading.lemma.lower()
            for reading in _analyses(morfeusz, word)
            if category in reading.names
        }
    )


def _is_title(word: _Word) -> bool:
    return word.text.lower() in _TITLES


def _is_role(morfeusz: MorfeuszAnalyzer, word: _Word) -> bool:
    if _is_title(word):
        return True
    lowered = word.text.lower()
    return lowered in _ROLE_LEMMAS or any(
        reading.lemma.lower() in _ROLE_LEMMAS for reading in morfeusz.analyses(lowered)
    )
