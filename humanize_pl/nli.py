"""Klauzula po klauzuli: czy dokument mówi to, czego szkielet od niego wymaga.

`blueprint.check` odpowiada na jedno pytanie — czy sekcja tu jest — i robi to
przez dopasowanie nagłówka. Na tym jej wiedza się kończy. Sekcja
„Wynagrodzenie”, pod którą stoi wyłącznie zdanie o terminie dostawy, przechodzi
ten test bez zastrzeżeń: nagłówek jest, słów starczy, numeracja się zgadza.
Struktura bywa kompletna wtedy, gdy treści nie ma.

Ten moduł zadaje pytanie o poziom niżej: czy dla każdej klauzuli, którą szkielet
zapowiada, w odnalezionej sekcji naprawdę stoi zdanie, które ją wyraża albo z
którego ona wynika. Odpowiada model, bo to pytanie o znaczenie, a nie o wzorzec.

**Ostrożność jest tu regułą, nie ustawieniem.** Model, który waha się między
„entailed” a „partial”, zwraca „entailed” — i tak samo rozstrzyga każda linia
tego kodu: nieznany werdykt, brakujący wpis w odpowiedzi, uszkodzony JSON, błąd
endpointu. Powód jest praktyczny, nie estetyczny. Narzędzie, które podnosi alarm
nad poprawną polszczyzną, zostaje wyłączone po trzecim takim alarmie i nie
złapie już nigdy niczego. Przeoczony brak kosztuje jedno znalezisko; fałszywy
alarm kosztuje całe narzędzie. Dlatego raport z tego modułu czyta się jako
„to na pewno warto sprawdzić”, nigdy jako „reszta jest w porządku”.

Jedno wywołanie modelu na sekcję, nie na klauzulę: klauzule jednej sekcji dzielą
ten sam kontekst, a sto osobnych rozmów o jednej umowie to sto szans na to, że
któraś się nie powiedzie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import regex as re

from humanize_pl.blueprint import DocumentBlueprint, Section, is_heading
from humanize_pl.blueprint_learning import (
    _UNIT_PATTERNS,
    document_headings,
    normalise_heading,
)
from humanize_pl.llm import (
    LlmEndpointError,
    LlmSettings,
    OpenAICompatibleRewriter,
    _safe_error,
)
from humanize_pl.safety.protectors import protect_text
from humanize_pl.sentence_splitter import split_sentences

ENTAILED = "entailed"
PARTIAL = "partial"
MISSING = "missing"
ABSENT = "absent"

# Kolejność ciężaru. Werdykt sekcji to najgorszy werdykt jej klauzul, a werdykt
# dokumentu — najgorszy werdykt jego sekcji; brak całej sekcji bije wszystko.
_RANK = {ENTAILED: 0, PARTIAL: 1, MISSING: 2, ABSENT: 3}

# Model odpowiada po polsku albo po angielsku, zależnie od tego, co akurat
# przeważyło w jego treningu. Cokolwiek spoza tej tabeli znaczy „nie wiem”, a
# „nie wiem” znaczy `entailed` — tak samo jak brak odpowiedzi.
_ALIASES = {
    "entailed": ENTAILED,
    "entails": ENTAILED,
    "entailment": ENTAILED,
    "covered": ENTAILED,
    "yes": ENTAILED,
    "tak": ENTAILED,
    "pokryta": ENTAILED,
    "pokryte": ENTAILED,
    "spełniona": ENTAILED,
    "partial": PARTIAL,
    "partially": PARTIAL,
    "częściowo": PARTIAL,
    "czesciowo": PARTIAL,
    "częściowa": PARTIAL,
    "missing": MISSING,
    "absent": MISSING,
    "no": MISSING,
    "nie": MISSING,
    "brak": MISSING,
    "brakuje": MISSING,
}

# Budżet odpowiedzi. Jeden wpis to mniej więcej piętnaście tokenów, więc
# dwanaście klauzul w jednym pytaniu mieści się w tysiącu z dużym zapasem —
# a sekcja z większą liczbą wymagań idzie w kolejnych paczkach, nadal po jednym
# wywołaniu na paczkę, nie na klauzulę.
MAX_COMPLETION_TOKENS = 1000
MAX_EXPECTED_CLAUSES = 12

# Bezpiecznik rozmiaru zapytania, nie kryterium oceny. Obcięcie przesłanki może
# tylko zabrać modelowi dowód pokrycia, czyli pchnąć werdykt w stronę ostrą,
# więc progi są wysokie: sekcja dłuższa niż czterdzieści klauzul praktycznie nie
# istnieje, a klauzula dłuższa niż czterysta znaków to już cały akapit.
MAX_DOCUMENT_CLAUSES = 40
MAX_CLAUSE_CHARS = 400

# Fragment krótszy niż to nie jest samodzielną klauzulą, tylko ogonem
# poprzedniej („; oraz”, „a także”), i wraca tam, skąd go odcięto.
_MIN_CLAUSE_WORDS = 3

# Znaczniki wyliczenia na początku wiersza: „1.”, „2)”, „a)”, „—”.
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|\p{Ll}[.)]|[-–—•*])\s+")
_LAST_TOKEN = re.compile(r"(\p{L}+)\.$")
_HAS_LETTER = re.compile(r"\p{L}")

# Skróty, po których kropka nie kończy zdania. Bez nich „w dniu 3.04.2026 r.
# Strony ustalają…” rozpada się na dwie klauzule w miejscu, w którym nie ma
# granicy zdania.
_ABBREVIATIONS = frozenset(
    (
        "art", "ust", "pkt", "lit", "poz", "nr", "r", "tj", "tzn", "np",
        "m", "in", "itp", "itd", "ok", "godz", "zł", "tys", "mln", "mld",
        "sp", "o", "ww", "ds", "al", "ul", "woj", "str", "rys", "tab",
        "zob", "por",
    )
)


@dataclass(frozen=True)
class ClauseCheck:
    """Jedna klauzula ze szkieletu i to, co dokument z nią zrobił."""

    clause: str
    verdict: str


@dataclass(frozen=True)
class SectionCheck:
    section: str
    heading: str | None
    verdict: str
    clauses: tuple[ClauseCheck, ...] = ()


@dataclass(frozen=True)
class LocatedSection:
    """Kawałek dokumentu, w którym rozpoznano sekcję szkieletu."""

    heading: str
    raw_heading: str
    body: str


@dataclass
class NliReport:
    """Co szkielet zapowiadał i czego dokument pod tym nagłówkiem nie powiedział."""

    category: str
    blueprint: str | None = None
    sections: list[SectionCheck] = field(default_factory=list)
    # Model, który nie odpowiedział, zostawia sekcję ocenioną łagodnie — i ślad
    # tutaj. Bez tego raport bez zastrzeżeń wygląda identycznie, niezależnie od
    # tego, czy model potwierdził pokrycie, czy w ogóle się nie odezwał.
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return max(
            (row.verdict for row in self.sections),
            key=lambda value: _RANK.get(value, 0),
            default=ENTAILED,
        )

    @property
    def issues(self) -> list[str]:
        """To, co warto sprawdzić, po polsku, od najcięższego."""
        rows: list[str] = []
        for section in sorted(self.sections, key=lambda row: -_RANK.get(row.verdict, 0)):
            if section.verdict == ABSENT:
                rows.append(f"brak sekcji: {section.section}")
                continue
            for clause in section.clauses:
                if clause.verdict == MISSING:
                    rows.append(f"{section.section}: brak treści „{clause.clause}”")
                elif clause.verdict == PARTIAL:
                    rows.append(f"{section.section}: treść „{clause.clause}” pokryta częściowo")
        return rows

    def to_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "blueprint": self.blueprint,
            "verdict": self.verdict,
            "warnings": list(self.warnings),
            "sections": [
                {
                    "section": row.section,
                    "heading": row.heading,
                    "verdict": row.verdict,
                    "clauses": [
                        {"clause": item.clause, "verdict": item.verdict}
                        for item in row.clauses
                    ],
                }
                for row in self.sections
            ],
        }


def _ends_with_abbreviation(fragment: str) -> bool:
    match = _LAST_TOKEN.search(fragment.rstrip())
    if match is None:
        return False
    token = match.group(1)
    # Inicjał („J. Kowalski”) dzieli los skrótu: kropka po nim nie kończy zdania.
    if len(token) == 1 and token.isupper():
        return True
    return token.casefold() in _ABBREVIATIONS


def _merge_abbreviated(parts: list[str]) -> list[str]:
    merged: list[str] = []
    for part in parts:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
            continue
        merged.append(part)
    return merged


def _split_enumeration(sentence: str) -> list[str]:
    """Rozetnij wyliczenie na średnikach, bo tak zapisuje je polska legislacja."""
    if ";" not in sentence:
        return [sentence]
    merged: list[str] = []
    pending = ""
    for raw in sentence.split(";"):
        piece = raw.strip()
        if not piece:
            continue
        piece = f"{pending}; {piece}" if pending else piece
        pending = ""
        if len(piece.split()) < _MIN_CLAUSE_WORDS:
            if merged:
                merged[-1] = f"{merged[-1]}; {piece}"
            else:
                # Zapowiedź wyliczenia („Wykonawca zobowiązuje się:”) nie stoi
                # sama — dokleja się do pierwszego punktu, który opisuje.
                pending = piece
            continue
        merged.append(piece)
    if pending:
        merged.append(pending)
    return merged or [sentence]


def split_clauses(text: str) -> list[str]:
    """Potnij tekst na klauzule: wiersz, zdanie, punkt wyliczenia.

    Zdania dzieli wspólny `split_sentences`, żeby poprawka tam działała i tutaj.
    Nad nim stoją trzy rzeczy, których tamten nie wie, a które w piśmie prawnym
    decydują: wiersz jest granicą (wyliczenie bywa bez kropek), średnik też, a
    kropka po skrócie — „art.”, „ust.”, „r.” — granicą nie jest.
    """
    clauses: list[str] = []
    for raw_line in text.split("\n"):
        line = _LIST_MARKER.sub("", raw_line.strip())
        if not line:
            continue
        for sentence in _merge_abbreviated(split_sentences(line)):
            clauses.extend(_split_enumeration(sentence))
    return [
        row for row in (clause.strip(" ;,") for clause in clauses) if _HAS_LETTER.search(row)
    ]


def expected_clauses(section: Section) -> list[str]:
    """Czego szkielet oczekuje po tej sekcji, klauzula po klauzuli.

    `expects` w YAML-u jest opcjonalne i w żadnym z dostarczonych szkieletów go
    nie ma — `matches` odpowiada na pytanie „czy sekcja tu jest”, a nie „co ma
    mówić”. Bez niego zostaje jedno pytanie, warte zadania: czy ta sekcja w
    ogóle traktuje o tym, co obiecuje jej nazwa. Reszta `matches` do tej roli
    się nie nadaje — to synonimy nagłówka, więc pytanie o każdy z nich osobno
    zgłosiłoby brak wszędzie tam, gdzie dokument użył innego z nich.
    """
    declared = getattr(section, "expects", ()) or ()
    if not declared:
        return [section.label_pl]
    clauses: list[str] = []
    for row in declared:
        clauses.extend(split_clauses(row))
    return clauses


@dataclass(frozen=True)
class _Heading:
    index: int
    raw: str
    text: str


def _headings_with_positions(text: str) -> tuple[list[str], list[_Heading]]:
    lines = [line for line in text.split("\n") if line.strip()]
    indices = [index for index, line in enumerate(lines) if is_heading(line)]
    # `document_headings` zna jedną rzecz, której `is_heading` nie zna: tytuł
    # dokumentu jest nagłówkiem i nie jest sekcją. Odrzuca go z początku listy,
    # więc ogony obu list są zgodne i różnica długości mówi, ile odpadło.
    titles = document_headings(text)
    offset = len(indices) - len(titles)
    if offset < 0:
        return lines, [
            _Heading(index, lines[index], normalise_heading(lines[index]))
            for index in indices
        ]
    return lines, [
        _Heading(indices[position + offset], lines[indices[position + offset]], heading)
        for position, heading in enumerate(titles)
    ]


def _is_unit_heading(raw: str, numbering: str | None) -> bool:
    """Czy ten nagłówek otwiera jednostkę najwyższego rzędu, a nie podpunkt.

    „§ 4.” i „III.” w polskim piśmie prawnym zawsze ją otwierają. „1.” bywa
    jednym i drugim, a rozstrzyga interpunkcja: podpunkt jest zdaniem i kończy
    się kropką albo średnikiem, nagłówek nie kończy się niczym.
    """
    stripped = raw.strip()
    patterns = (
        [_UNIT_PATTERNS[numbering]]
        if numbering in _UNIT_PATTERNS
        else list(_UNIT_PATTERNS.values())
    )
    if not any(pattern.match(stripped) for pattern in patterns):
        return False
    return not (
        _UNIT_PATTERNS["arabic"].match(stripped) and stripped.endswith((".", ";", ","))
    )


def _match_heading(section: Section, headings: list[_Heading], taken: set[int]) -> int | None:
    for position, heading in enumerate(headings):
        if position in taken:
            continue
        if section.found_in(heading.text.casefold()):
            return position
    # Druga szansa dla wzorców, które niosą własny znacznik jednostki
    # („§ 4 wynagrodzenie”): z `heading.text` znacznik już zniknął.
    for position, heading in enumerate(headings):
        if position not in taken and section.found_in(heading.raw.casefold()):
            return position
    return None


def locate_sections(
    doc_text: str, blueprint: DocumentBlueprint
) -> list[tuple[Section, LocatedSection | None]]:
    """Przypisz każdej sekcji szkieletu kawałek dokumentu — albo nic.

    Sekcja kończy się na pierwszym z dwóch: następnym nagłówku, który szkielet
    rozpoznał jako swoją sekcję, albo następnej jednostce najwyższego rzędu
    („§ 3.”). Nie na następnym nagłówku w ogóle — `is_heading` uznaje za
    nagłówek także krótki punkt wyliczenia („1. Wykonawca dostarczy
    dokumentację.”), a cięcie na nim ucinałoby sekcję na jej własnym pierwszym
    podpunkcie i pytało model o pokrycie, pokazawszy mu jedno zdanie z
    dziesięciu.
    """
    lines, headings = _headings_with_positions(doc_text)
    claimed: dict[int, str] = {}
    matched: list[int | None] = []
    for section in blueprint.sections:
        position = _match_heading(section, headings, set(claimed))
        matched.append(position)
        if position is not None:
            claimed[position] = section.id

    boundaries = sorted(claimed)
    units = [
        position
        for position, heading in enumerate(headings)
        if _is_unit_heading(heading.raw, blueprint.numbering)
    ]
    located: list[tuple[Section, LocatedSection | None]] = []
    for section, position in zip(blueprint.sections, matched):
        if position is None:
            located.append((section, None))
            continue
        start = headings[position].index
        stop = min(
            (
                next((headings[row].index for row in rows if row > position), len(lines))
                for rows in (boundaries, units)
            ),
            default=len(lines),
        )
        located.append(
            (
                section,
                LocatedSection(
                    heading=headings[position].text,
                    raw_heading=headings[position].raw,
                    body="\n".join(lines[start + 1 : stop]),
                ),
            )
        )
    return located


def normalise_verdict(value: Any) -> str:
    """Każde „nie wiem” — cudze i własne — kończy się na `entailed`."""
    if not isinstance(value, str):
        return ENTAILED
    return _ALIASES.get(value.strip().casefold(), ENTAILED)


_SYSTEM_PROMPT = (
    "Jesteś asystentem kancelarii prawnej. Sprawdzasz WYŁĄCZNIE, czy treść "
    "sekcji dokumentu pokrywa wymaganą treść. Nie oceniasz stylu, języka, "
    "poprawności prawnej ani jakości redakcji.\n"
    "Dla każdej wymaganej treści zwróć jedną ocenę:\n"
    '  "entailed" — sekcja wyraża ją wprost albo ona z sekcji wynika,\n'
    '  "partial" — sekcja dotyka tematu, ale pomija jego istotny element,\n'
    '  "missing" — w sekcji nie ma nic na ten temat.\n'
    "Rozstrzyganie wątpliwości: jeśli wahasz się między dwiema ocenami, wybierz "
    'łagodniejszą; jeśli nie masz pewności, zwróć "entailed". Lepiej przeoczyć '
    "brak niż zgłosić poprawny zapis jako wadliwy.\n"
    "Ciągi postaci __PROTECTED_0000__ to usunięte dane (kwoty, daty, nazwy, "
    "numery). Traktuj je jako obecne i poprawne.\n"
    'Zwróć wyłącznie obiekt JSON postaci {"oceny": [{"nr": 1, "ocena": '
    '"entailed"}]}, po jednym wpisie na każdą wymaganą treść, w tej samej '
    "kolejności. Bez komentarza i bez bloku kodu."
)


def build_messages(
    heading: str, document_clauses: list[str], expected: list[str]
) -> list[dict[str, str]]:
    """Zapytanie o jedną sekcję: przesłanki, wymagania, format odpowiedzi."""
    premises = "\n".join(
        f"{number}. {clause}" for number, clause in enumerate(document_clauses, 1)
    )
    hypotheses = "\n".join(f"{number}. {clause}" for number, clause in enumerate(expected, 1))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Sekcja dokumentu: „{heading}”\n\n"
                f"Treść sekcji:\n{premises}\n\n"
                f"Wymagane treści:\n{hypotheses}\n\n"
                f"Zwróć dokładnie {len(expected)} ocen, po jednej na każdą "
                "wymaganą treść, w polu „oceny”."
            ),
        },
    ]


def read_verdicts(payload: dict[str, Any], count: int) -> list[str]:
    """Przeczytaj odpowiedź modelu.

    Czego w niej nie ma albo czego nie da się przeczytać — `entailed`.
    """
    verdicts = [ENTAILED] * count
    rows = payload.get("oceny")
    if not isinstance(rows, list):
        return verdicts
    for position, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        try:
            number = int(row.get("nr", position))
        except (TypeError, ValueError):
            continue
        if 1 <= number <= count:
            verdicts[number - 1] = normalise_verdict(row.get("ocena"))
    return verdicts


class ClauseJudge(Protocol):
    """Kto odpowiada na pytanie o pokrycie jednej sekcji.

    Zwraca po jednym werdykcie na każdą wymaganą treść, w tej samej kolejności.
    Krótsza lista jest dopuszczalna — brakujące pozycje czyta się łagodnie.
    Opcjonalne pole `warnings` trafia do raportu.
    """

    def judge_section(
        self, *, heading: str, document_clauses: list[str], expected_clauses: list[str]
    ) -> list[str]: ...


class LlmClauseJudge:
    """Werdykty z modelu — jedna rozmowa na sekcję, przez klienta z `llm.py`."""

    def __init__(
        self,
        client: OpenAICompatibleRewriter,
        *,
        max_completion_tokens: int = MAX_COMPLETION_TOKENS,
        batch: int = MAX_EXPECTED_CLAUSES,
    ) -> None:
        self._client = client
        self._max_completion_tokens = max_completion_tokens
        self._batch = max(1, batch)
        self.warnings: list[str] = []

    @classmethod
    def from_environment(cls, env_file: str | Path | None = None) -> LlmClauseJudge:
        return cls(OpenAICompatibleRewriter(LlmSettings.from_environment(env_file)))

    def judge_section(
        self, *, heading: str, document_clauses: list[str], expected_clauses: list[str]
    ) -> list[str]:
        verdicts: list[str] = []
        for start in range(0, len(expected_clauses), self._batch):
            chunk = expected_clauses[start : start + self._batch]
            verdicts.extend(self._ask(heading, document_clauses, chunk))
        return verdicts

    def _ask(self, heading: str, document_clauses: list[str], expected: list[str]) -> list[str]:
        messages = build_messages(heading, document_clauses, expected)
        try:
            payload = self._client.complete_json(
                messages, max_tokens=self._max_completion_tokens
            )
        except LlmEndpointError as exc:
            # Milczenie endpointu nie jest dowodem braku klauzuli. Sekcja
            # dostaje ocenę łagodną, a raport — zdanie o tym, że jej nie
            # sprawdzono.
            self.warnings.append(f"sekcja „{heading}”: {_safe_error(exc)}")
            return [ENTAILED] * len(expected)
        return read_verdicts(payload, len(expected))


def _prompt_clauses(body: str) -> list[str]:
    """Klauzule sekcji gotowe do wysłania: zredagowane i przycięte do budżetu.

    `protect_text` jest tu z tego samego powodu, dla którego jest w ścieżce
    redakcyjnej: poza kancelarię wychodzi tekst bez nazwisk, numerów i kwot.
    Zamiana idzie na całej sekcji naraz, nie na pojedynczych klauzulach —
    inaczej licznik zaczynałby się od zera w każdej z nich i ten sam
    `__PROTECTED_0000__` znaczyłby w jednym zdaniu datę, a w następnym nazwisko.
    """
    protected = protect_text(body, include_sensitive=True)
    clauses = split_clauses(protected.text)[:MAX_DOCUMENT_CLAUSES]
    return [
        clause if len(clause) <= MAX_CLAUSE_CHARS else f"{clause[:MAX_CLAUSE_CHARS]}…"
        for clause in clauses
    ]


def check_document_against_blueprint(
    doc_text: str,
    blueprint: DocumentBlueprint,
    *,
    judge: ClauseJudge | None = None,
) -> NliReport:
    """Sprawdź klauzula po klauzuli, czy dokument pokrywa treść ze szkieletu.

    Bez `judge` klient modelu powstaje z konfiguracji (`.env`) tak samo jak w
    całej reszcie narzędzia; w testach i w trybie offline podaje się własnego.
    """
    if judge is None:
        judge = LlmClauseJudge.from_environment()

    report = NliReport(category=blueprint.category, blueprint=blueprint.label_pl)
    for section, part in locate_sections(doc_text, blueprint):
        expected = expected_clauses(section)
        if part is None:
            # Werdykty klauzul nie są tu zgadywanką modelu, tylko następstwem:
            # sekcji nie ma, więc nie ma w niej żadnej z nich. Wypisane, bo
            # raport ma pokazywać, czego dokument jest winien, a nie tylko to,
            # że czegoś brakuje.
            report.sections.append(
                SectionCheck(
                    section=section.id,
                    heading=None,
                    verdict=ABSENT,
                    clauses=tuple(ClauseCheck(clause, MISSING) for clause in expected),
                )
            )
            continue

        document_clauses = _prompt_clauses(part.body)
        if not expected:
            report.sections.append(SectionCheck(section.id, part.heading, ENTAILED, ()))
            continue
        if not document_clauses:
            # Nagłówek, pod którym nie ma ani jednego zdania. Modelu nie ma tu
            # o co pytać: pusty fragment nie pokrywa niczego. To ta sama
            # obserwacja, którą `blueprint.check` zgłasza jako „sekcja bez
            # treści”, więc zgodność obu raportów jest spójna, a nie jest nowym
            # źródłem fałszywych alarmów.
            report.sections.append(
                SectionCheck(
                    section=section.id,
                    heading=part.heading,
                    verdict=MISSING,
                    clauses=tuple(ClauseCheck(clause, MISSING) for clause in expected),
                )
            )
            continue

        verdicts = judge.judge_section(
            heading=part.heading,
            document_clauses=document_clauses,
            expected_clauses=list(expected),
        )
        clauses = tuple(
            ClauseCheck(
                clause,
                normalise_verdict(verdicts[position])
                if position < len(verdicts)
                else ENTAILED,
            )
            for position, clause in enumerate(expected)
        )
        report.sections.append(
            SectionCheck(
                section=section.id,
                heading=part.heading,
                verdict=max(
                    (row.verdict for row in clauses),
                    key=lambda value: _RANK.get(value, 0),
                    default=ENTAILED,
                ),
                clauses=clauses,
            )
        )

    report.warnings.extend(getattr(judge, "warnings", []) or [])
    return report
