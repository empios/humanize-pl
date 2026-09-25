"""Supply a section the document owes its category.

Everything else in this engine edits what is already there. This writes
something that is not, which is a different kind of act and carries a
different risk, so the module is deliberately small and says no in more
places than it says yes.

What makes it possible at all is `expects`: the blueprint already records, in
Polish, what each section has to say - "Umowa określa wysokość wynagrodzenia
albo sposób jego obliczenia." The model is not asked to invent an obligation;
it is asked to phrase one the skeleton already named, in the register of the
surrounding document.

What the model is never asked to do is supply facts. A missing payment clause
cannot be drafted with an amount in it, because there is no amount to know.
The prompt asks for a blank ("…") wherever a fact belongs, and the check below
enforces the other half: a draft carrying a figure, a date, a legal reference
or an identifier that the rest of the document does not already carry is
refused, not cleaned up. A blank is the honest form of "not known here" - it
is visibly unfinished, where a descriptive stand-in ("w terminie uzgodnionym
przez Strony") reads as agreed while agreeing nothing.

The clause enters the document unmarked, by explicit decision of the owner,
and the report is therefore the only place that says a machine wrote it. Two
consequences follow and are implemented rather than assumed: a document with
a drafted clause never reports as ready, and the report carries a section of
its own listing every draft in full.

Where the document numbers its units with "§", a drafted section gets a unit
of its own, numbered the way Polish drafting technique adds a unit between two
existing ones: "§ 2a" after "§ 2". Renumbering everything below would be the
tidier result and the more dangerous one - every "zgodnie z § 5" elsewhere in
the document, and in any annex, would silently point at the wrong clause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import regex as re

from humanize_pl.blueprint import DocumentBlueprint, Section
from humanize_pl.llm import LlmEndpointError, OpenAICompatibleRewriter, _safe_error
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import legal_sensitive_inventory

# Room for a clause, not for an essay. A model asked for one paragraph of
# contract language and given four thousand tokens will write four thousand
# tokens of it.
#
# But a reasoning model (qwen-local, the first endpoint used) spends its
# budget in `reasoning_content` before writing a word of the answer, so the
# budget has to cover the thinking as well; the cap on the clause itself is
# `MAX_WORDS`, checked after the fact. It also has to fit the context window:
# Bielik on llama.cpp has 8192 tokens for prompt and answer together.
MAX_TOKENS = 4000

# A drafted clause longer than this is not a clause.
MAX_WORDS = 160

# Ustępy of a clause. More than this is a chapter.
MAX_PARAGRAPHS = 6
# Lines of a block outside the numbering - a signature block is a dozen
# short lines by nature ("Zleceniobiorca:", a dotted line, "(podpis)", twice
# over, then place and date), and six refused the one Bielik wrote.
MAX_BLOCK_LINES = 14

# The first answer and one corrected one. A model that fails twice with the
# reason in front of it is not going to write this clause.
ATTEMPTS = 2

# Concrete particulars the model has no way of knowing. Any of these in a
# draft means it invented something.
_AMOUNT = re.compile(r"\d+(?:[ .]\d{3})*(?:[,.]\d+)?\s*(?:zł|PLN|EUR|USD|%)", re.IGNORECASE)
_DATE = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b|\b\d{1,2}\s+\p{L}+\s+\d{4}\b")
# A period or a deadline: "7 dni", "12 miesięcy", "do 5. dnia każdego
# miesiąca". The prompt forbids them and qwen-local complied; Bielik wrote
# four in one clause, and nothing here checked - a fourteen-day term nobody
# agreed to would have gone into the contract.
_PERIOD = re.compile(
    r"\b\d+\s*(?:dni|dnia|tygodni|tygodnia|tygodnie|miesięcy|miesiąca|miesiące|"
    r"lat|lata|roku|godzin|godziny|godzinę)\b|\b\d+\.\s*(?:dnia|dniu|dzień)\b",
    re.IGNORECASE,
)
# An attachment the document may not have: Bielik put a schedule "w
# załączniku nr 1" and reports "w załączniku nr 2" into a contract with no
# attachments at all - a reference to a document nobody wrote.
_ATTACHMENT = re.compile(r"\bzałącznik\w*\s+(?:nr\s*)?\d+", re.IGNORECASE)
# A hint for whoever fills a blank, "… (np. 15. dzień każdego miesiąca)": an
# instruction to the drafter, not contract text, and a place for an invented
# term to hide.
_BLANK_HINT = re.compile(r"(…+|\.{3,})\s*\(np\.[^)]*\)", re.IGNORECASE)
# The engine's own placeholders, echoed back by a model and not restored.
_PLACEHOLDER_ECHO = re.compile(r"__PROTECTED_\d+__")
_LEGAL_REF = re.compile(
    r"\bart\.\s*\d|§\s*\d|\bust\.\s*\d|\bpkt\s*\d|\bDz\.\s*U\.", re.IGNORECASE
)

# The inventory the rewrite gate protects, reused so "a fact the document
# does not have" means the same thing here as everywhere else. Party names
# and quoted terms are left out on purpose: a draft saying "Strony Umowy"
# where the document only ever wrote "Strony" is wording, not an invented
# party, and refusing it would refuse most contract clauses.
_FACT_CATEGORIES = {
    "amounts": "kwota",
    "dates": "data",
    "legal_references": "przepis",
    "identifiers": "identyfikator",
}

# A blank the model left where a fact belongs.
_BLANK = re.compile(r"…+|\.{3,}")

# A "§" unit heading: prefix, number, letter suffix, separator, title.
UNIT_HEADING = re.compile(r"^(\s*§\s*)(\d+)(\p{Ll}?)(\s*\.?\s*)(.*)$")


@dataclass(frozen=True)
class DraftedSection:
    """One clause written for a section the document did not have."""

    section_id: str
    label_pl: str
    text: str
    expects: tuple[str, ...]
    after_line: int
    # Position in the skeleton. Two gaps that anchor on the same existing
    # section share `after_line`, and only this says which comes first.
    rank: int = 0
    # "§ 2a. Odpowiedzialność" when the document numbers its units with "§",
    # empty otherwise. A line of its own, so a clause does not end up inside
    # the unit before it - a liability clause sitting in "§ 2. Wynagrodzenie"
    # would read as part of the payment terms.
    heading: str = ""

    @property
    def lines(self) -> tuple[str, ...]:
        """One entry per paragraph the draft becomes: heading, then ustępy."""
        body = tuple(self.text.split("\n"))
        return (self.heading, *body) if self.heading else body

    @property
    def blanks(self) -> int:
        return len(_BLANK.findall(self.text))


@dataclass
class DraftingResult:
    drafts: list[DraftedSection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def any_drafted(self) -> bool:
        return bool(self.drafts)


# Longer than this, a document is not sent whole to ask whether a section is
# in it: cut down, a "no" could only mean the section sat past the cut.
PRESENCE_MAX_CHARS = 30000
PRESENCE_QUOTE_MIN_CHARS = 12


@dataclass(frozen=True)
class SectionPresence:
    """Whether the model finds a section the patterns missed.

    `state` is "absent" (drafting may go ahead), "present" (with `quote`, a
    passage verified to be in the document) or "unclear" (with `reason`).
    Anything short of a verified answer counts as unclear, and unclear
    never leads to drafting.
    """

    state: str
    quote: str = ""
    reason: str = ""


def section_presence(
    document: str, section: Section, *, client: OpenAICompatibleRewriter
) -> SectionPresence:
    """Ask whether `document` already has `section`, in words the patterns lack.

    The structure check looks for phrases, and the firm writes the same
    clause in its own words: on its complete documents the check called 14
    of 20 incomplete. A section is drafted only when the model says it is
    not there; a "yes" must come with a passage that really is in the
    document, so an invented quote cannot stop - or start - anything.
    """
    protected = protect_text(document, include_sensitive=True)
    if len(protected.text) > PRESENCE_MAX_CHARS:
        return SectionPresence("unclear", reason="dokument za długi, by sprawdzić go w całości")
    clauses = " ".join(section.expects) or f"Sekcja dotyczy: {section.label_pl}."
    messages = [
        {
            "role": "system",
            "content": (
                "Sprawdzasz, czy dokument zawiera wskazaną część, choćby opisaną innymi "
                "słowami. Odpowiadasz wyłącznie obiektem JSON z polami: obecna (true albo "
                "false) i cytat (tekst). Jeżeli część jest w dokumencie, w polu cytat "
                "przepisz z dokumentu dosłownie, znak w znak, jedno zdanie, które ją "
                "stanowi. Jeżeli jej nie ma, cytat zostaw pusty. Bez komentarza."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Część: {section.label_pl}.\nCo powinna zawierać: {clauses}\n\n"
                f"Dokument:\n{protected.text}"
            ),
        },
    ]
    try:
        answer = client.complete_json(messages, max_tokens=400)
    except (LlmEndpointError, OSError, ValueError) as exc:
        return SectionPresence("unclear", reason=_safe_error(exc))
    if not isinstance(answer, dict):
        return SectionPresence("unclear", reason="odpowiedź modelu nie jest obiektem JSON")
    present = answer.get("obecna")
    if present is False:
        if answer.get("cytat") not in (None, ""):
            return SectionPresence("unclear", reason="model jednocześnie wskazał brak sekcji i cytat")
        return SectionPresence("absent")
    if present is not True:
        return SectionPresence("unclear", reason="odpowiedź modelu bez rozstrzygnięcia")
    quote = str(answer.get("cytat") or "").strip().strip("„”\"")

    def squash(value: str) -> str:
        return " ".join(value.split()).casefold()

    if len(quote) >= PRESENCE_QUOTE_MIN_CHARS:
        # The model reads the document with names and figures masked, so its
        # quote carries the placeholders; restored, it is the document's own.
        if squash(quote) in squash(protected.text):
            return SectionPresence("present", quote=protected.restore(quote))
        if squash(quote) in squash(document):
            return SectionPresence("present", quote=quote)
    return SectionPresence("unclear", reason="model wskazał cytat, którego nie ma w dokumencie")


def invented_particulars(draft: str, document: str) -> list[str]:
    """Figures, dates and legal references the draft has and the document does not.

    Checked rather than trusted. A model told not to invent an amount will
    usually comply and occasionally write "w terminie 14 dni od dnia
    otrzymania faktury" because that is what such clauses say - and a
    fourteen-day deadline nobody agreed to is a worse defect than the missing
    clause it replaced.
    """
    found: list[str] = []
    for label, pattern in (
        ("kwota", _AMOUNT),
        ("data", _DATE),
        ("termin", _PERIOD),
        ("przepis", _LEGAL_REF),
        ("załącznik", _ATTACHMENT),
    ):
        for match in pattern.finditer(draft):
            value = match.group(0).strip()
            if value and value not in document:
                found.append(f"{label}: {value}")
    known = legal_sensitive_inventory(document)
    for category, values in legal_sensitive_inventory(draft).items():
        label = _FACT_CATEGORIES.get(category)
        if label is None:
            continue
        for value in values:
            entry = f"{label}: {value}"
            if value not in known[category] and not any(
                value in row.casefold() for row in found
            ):
                found.append(entry)
    return found


def _messages(
    section: Section, blueprint: DocumentBlueprint, neighbours: str
) -> list[dict[str, str]]:
    clauses = " ".join(section.expects) or f"Sekcja dotyczy: {section.label_pl}."
    phrases = ", ".join(f"„{phrase}”" for phrase in section.matches)
    # Two corrections measured on the configured model. Told not to give
    # "przepisów", it wrote "stosuje się odpowiednie przepisy" where every
    # contract names the Civil Code - the ban is on article numbers, not on
    # naming an act. And handed the requirements as "Sekcja ma powiedzieć",
    # it restated one ("Umowa określa sposób i skutek, z jakim może nastąpić
    # jej rozwiązanie") instead of writing the clause that meets it.
    system = (
        "Jesteś polskim prawnikiem redagującym dokumenty. Piszesz wyłącznie "
        "treść jednej brakującej jednostki redakcyjnej, w rejestrze dokumentu, "
        "który dostajesz jako kontekst. "
        "Piszesz postanowienia: kto co może albo musi zrobić. Nie opisujesz, "
        "co dokument reguluje, i nie powtarzasz wymogów, które dostajesz. "
        "Nie podawaj kwot, dat, terminów liczbowych, numerów rachunków, "
        "danych stron ani numerów artykułów i paragrafów aktów prawnych. "
        "Nie znasz ich i nie wolno ci ich wymyślać. W miejscu każdej takiej "
        "wartości zostaw wykropkowanie „…”, które prawnik uzupełni. Ustawę "
        "możesz powołać z nazwy. Nie odsyłaj do innych paragrafów ani punktów "
        "dokumentu i nie używaj wypunktowań. "
        "Jeżeli sekcja ma kilka ustępów, każdy zaczynasz od nowej linii. "
        "Odpowiadasz samą treścią, bez nagłówka, bez numeracji ustępów, "
        "bez komentarza i bez bloku kodu."
    )
    user = "\n".join(
        [
            f"Rodzaj dokumentu: {blueprint.label_pl}.",
            f"Brakująca sekcja: {section.label_pl}.",
            f"Wymogi, które sekcja ma spełnić (nie przepisuj ich): {clauses}",
            f"Użyj w treści jednego ze sformułowań: {phrases}.",
            f"Fragment dokumentu dla rejestru i terminologii:\n{neighbours}",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _opening(text: str) -> str:
    return " ".join(re.findall(r"\p{L}+", text.casefold())[:2])


def restated_requirement(draft: str, section: Section) -> str | None:
    """A sentence of `draft` that restates a requirement instead of meeting it.

    Every requirement in the shipped skeletons is phrased as a description of
    the document - "Umowa określa…", "Regulamin wskazuje…", "Wezwanie
    podaje…" - and a clause is phrased as an obligation or a right of someone
    in it. A drafted sentence opening the way a requirement opens is the
    requirement copied into the document.

    Similarity would not do: the one real restatement seen scored 0.65
    against its requirement, while a proper confidentiality clause ("Wykonawca
    zobowiązuje się zachować w poufności…") scored 0.85 against its own.
    """
    openings = {_opening(line) for line in section.expects} - {""}
    for sentence in re.split(r"(?<=[.!?])\s+|\n", draft):
        if sentence.strip() and _opening(sentence) in openings:
            return sentence.strip()
    return None


def draft_missing_sections(
    text: str,
    blueprint: DocumentBlueprint,
    missing_labels: list[str],
    *,
    client: OpenAICompatibleRewriter,
) -> DraftingResult:
    """Write the sections named in `missing_labels`, or say why not.

    `missing_labels` comes from `BlueprintReport.missing_required`, so the
    caller has already decided which gaps are worth filling and this does not
    second-guess it.
    """
    result = DraftingResult()
    by_label = {section.label_pl: section for section in blueprint.sections}
    order = {section.id: index for index, section in enumerate(blueprint.sections)}
    lines = text.split("\n")

    # Register and terminology come from the document itself, redacted. The
    # model needs to see how this document writes, not what it is about.
    protected = protect_text(text, include_sensitive=True)
    neighbours = "\n".join(
        line for line in protected.text.split("\n") if line.strip()
    )[:1500]

    units = _unit_headings(lines)
    accepted: list[tuple[Section, str, int]] = []
    for label in missing_labels:
        section = by_label.get(label)
        if section is None:
            continue
        after_line = _insertion_line(lines, blueprint, section, order)
        # The heading allocated later carries the section's name when the
        # unit it follows carries a title, and the recognition check has to
        # see what the document will see.
        governing = [match for index, match in units if index <= after_line]
        titled = section.unit and bool(governing) and bool(governing[-1].group(5).strip())
        messages = _messages(section, blueprint, neighbours)

        draft: str | None = None
        reason = ""
        for _attempt in range(ATTEMPTS):
            try:
                answer = client.complete_text(messages, max_tokens=MAX_TOKENS)
            except LlmEndpointError as exc:
                reason = f"nie powstała: {_safe_error(exc).rstrip('.')}"
                break
            candidate = _clean_answer(protected.restore(answer), section.label_pl)
            reason = _rejection(
                candidate, section, text, section.label_pl if titled else ""
            ) or ""
            if not reason:
                draft = candidate
                break
            # One more try, told what was wrong. A model that invented a
            # deadline usually writes the blank when asked again; refusing
            # on the first slip would leave the gap for a reason the model
            # could have fixed.
            messages = [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        f"Tej propozycji nie można użyć: {reason}. "
                        "Napisz tę sekcję jeszcze raz, zgodnie z poleceniem."
                    ),
                },
            ]
        if draft is None:
            result.warnings.append(
                f"Propozycja sekcji „{label}” {reason}; sekcji nie dopisano."
            )
            continue
        accepted.append((section, draft, after_line))

    # Headings are allocated in document order, after every draft is known:
    # two sections landing after "§ 2" become "§ 2a" and "§ 2b" only once
    # both exist, and a refused one must not leave a gap in the letters.
    taken = {(int(match.group(2)), match.group(3)) for _, match in units}
    for section, draft, after_line in sorted(
        accepted, key=lambda row: (row[2], order[row[0].id])
    ):
        heading, key = (
            _heading_for(lines, units, after_line, section.label_pl, taken)
            if section.unit
            else ("", None)
        )
        if key is not None:
            taken.add(key)
        result.drafts.append(
            DraftedSection(
                section_id=section.id,
                label_pl=section.label_pl,
                text=draft,
                expects=section.expects,
                after_line=after_line,
                rank=order[section.id],
                heading=heading,
            )
        )
    result.drafts.sort(key=lambda row: row.rank)
    return result


def _rejection(draft: str, section: Section, document: str, heading: str) -> str | None:
    """Why `draft` cannot go into the document, or None when it can.

    Phrased to be read twice: by the model on its second attempt, and by the
    lawyer in the report when the second attempt failed too.
    """
    if not draft:
        return "jest pusta"
    words = len(draft.split())
    if words > MAX_WORDS:
        return f"jest za długa ({words} słów, najwyżej {MAX_WORDS})"
    paragraphs = len(draft.split("\n"))
    limit = MAX_PARAGRAPHS if section.unit else MAX_BLOCK_LINES
    if paragraphs > limit:
        noun = "ustępów" if section.unit else "linii"
        return f"ma za dużo {noun} ({paragraphs}, najwyżej {limit})"
    # A placeholder the restore did not recognise - the model altered it -
    # would go into the document as "__PROTECTED_0031__".
    echoed = _PLACEHOLDER_ECHO.search(draft)
    if echoed:
        return f"zawiera znacznik techniczny ({echoed.group(0)}) zamiast treści"
    invented = invented_particulars(draft, document)
    if invented:
        # Refused rather than stripped: a clause with its figure removed is a
        # clause with a hole in it, and the hole is harder to notice than the
        # absence of the whole section.
        return (
            "zawiera dane, których nie ma w dokumencie "
            f"({'; '.join(invented[:3])}), zamiast wykropkowania „…”"
        )
    restated = restated_requirement(draft, section)
    if restated:
        return f"powtarza wymóg zamiast go spełnić („{restated}”)"
    # The structure check finds a section by its words. A clause it would not
    # recognise leaves the document reported as missing the very section just
    # written into it - the worst of both outcomes.
    if section.found_in(f"{heading}\n{draft}".casefold()) is None:
        phrases = ", ".join(f"„{phrase}”" for phrase in section.matches)
        return (
            "nie zawiera żadnego sformułowania, po którym szkielet rozpoznaje "
            f"tę sekcję ({phrases})"
        )
    return None


def _clean_answer(draft: str, label: str) -> str:
    """The answer without the wrapping models add anyway, one line per ustęp.

    Lines are kept because a section of several ustępy is normal drafting and
    a signature block squeezed onto one line is not; each line becomes its
    own paragraph in a DOCX. What goes is a code fence, blank lines, markdown
    markers, and a heading line repeating the section's name despite the
    instruction - the heading is the engine's to write.

    Markup has to go here as well as on the input: the input is stripped
    before the rules run, and a clause the model writes afterwards would
    carry its "**" straight into the document.
    """
    from humanize_pl.artifacts import strip_markup

    rows = [
        # The space keeps the blank countable as a field: "….", glued to the
        # full stop, is no longer one.
        " ".join(_BLANK_HINT.sub(r"\1 ", line).split())
        for line in strip_markup(draft.strip().strip("`"))[0].split("\n")
    ]
    rows = [row for row in rows if row and not row.startswith("```")]
    title = re.sub(r"\s*\(.*?\)", "", label).strip().casefold()
    while rows and (
        UNIT_HEADING.match(rows[0]) or rows[0].casefold().rstrip(".:") == title
    ):
        rows.pop(0)
    # A number alone on its line belongs to the line after it: Bielik wrote a
    # signature block as "2.", "W imieniu Zleceniobiorcy:", "3.", ... and the
    # bare numbers went into the contract as paragraphs of their own.
    numbered: list[str] = []
    pending = ""
    for row in rows:
        if re.fullmatch(r"(?:\d+|[a-z])[.)]", row):
            pending = f"{pending}{row} "
            continue
        numbered.append(f"{pending}{row}")
        pending = ""
    rows = numbered
    # A line that stops mid-sentence and is continued in lower case is a
    # wrapped line, not an ustęp. Both conditions: the configured model
    # answered a signature block with "Zamawiający:\npodpis\n\nWykonawca:\n
    # podpis", and joining on the missing full stop alone produced
    # "podpis Wykonawca:".
    joined: list[str] = []
    for row in rows:
        if (
            joined
            and not joined[-1].endswith((".", ";", ":", "!", "?", "…"))
            and row[:1].islower()
        ):
            joined[-1] = f"{joined[-1]} {row}"
        else:
            joined.append(row)
    return "\n".join(joined)


def _unit_headings(lines: list[str]) -> list[tuple[int, Any]]:
    return [
        (index, match)
        for index, line in enumerate(lines)
        if (match := UNIT_HEADING.match(line))
    ]


def _heading_for(
    lines: list[str],
    units: list[tuple[int, Any]],
    after_line: int,
    label: str,
    taken: set[tuple[int, str]],
) -> tuple[str, tuple[int, str] | None]:
    """The "§" heading for a clause inserted after `after_line`.

    Modelled on the unit it follows, so "§ 2. Wynagrodzenie" gives
    "§ 2a. Odpowiedzialność" and a bare "§ 2" gives a bare "§ 2a". After the
    last unit there is nothing below to renumber, so the clause simply takes
    the next number: "§ 6" after "§ 5". Empty when the document does not
    number its units with "§", or when the clause lands before the first one
    - there is no unit to add it after.
    """
    governing = [match for index, match in units if index <= after_line]
    if not governing:
        return "", None
    model = governing[-1]
    if any(index > after_line for index, _ in units):
        number = int(model.group(2))
        letter = next(
            chr(code)
            for code in range(ord("a"), ord("z") + 1)
            if (number, chr(code)) not in taken
        )
    else:
        number = max(value for value, _ in taken) + 1
        letter = ""
    title = re.sub(r"\s*\(.*?\)", "", label).strip()
    title = title[:1].upper() + title[1:]
    if model.group(5).strip().isupper():
        title = title.upper()
    prefix, separator = model.group(1), model.group(4)
    if model.group(5).strip():
        heading = f"{prefix}{number}{letter}{separator}{title}"
    else:
        heading = f"{prefix}{number}{letter}{separator}".rstrip()
    return heading.strip(), (number, letter)


def _insertion_line(
    lines: list[str],
    blueprint: DocumentBlueprint,
    missing: Section,
    order: dict[str, int],
) -> int:
    """Line index to insert after, from the skeleton's own ordering.

    A clause belongs where the skeleton says it belongs, not at the end: a
    termination clause after the signature block would be a worse document
    than one without it. The anchor is the last section that is present and
    that the skeleton puts before this one.
    """
    lowered = [line.casefold() for line in lines]
    target_rank = order[missing.id]
    heading_rows = [index for index, _ in _unit_headings(lines)]

    # Where each later section starts, so the insertion stops before it.
    # Walking forward to the next blank line instead would have worked only
    # on documents that separate their clauses that way; a contract written
    # as consecutive lines would take the clause past two more sections.
    next_section_at = len(lines)
    # Located on the whole text, so a section recognised by a pattern that
    # spans lines - a signature block - bounds the insertion like any other;
    # a line-by-line phrase test missed it, and a final clause would have
    # gone in after the signatures.
    joined = "\n".join(lowered)
    for section in blueprint.sections:
        if order[section.id] <= target_rank:
            continue
        located = section.locate(joined)
        if located is not None:
            next_section_at = min(next_section_at, joined.count("\n", 0, located[1]))

    anchor = -1
    for section in blueprint.sections:
        if order[section.id] >= target_rank:
            continue
        for index, line in enumerate(lowered):
            if index < next_section_at and any(
                phrase in line for phrase in section.matches
            ):
                anchor = max(anchor, index)

    boundary = next_section_at
    if boundary < len(lines):
        # A section is found by its words, which may sit in the body under a
        # bare "§ 7." - the clause goes before that heading, not between it
        # and its body.
        opening = [row for row in heading_rows if row <= boundary]
        if opening and opening[-1] > anchor:
            boundary = opening[-1]

    if anchor < 0:
        # No earlier section found: sit just before the first later one, or
        # at the end when there is none.
        return boundary - 1

    if heading_rows:
        # A numbered document: after the whole unit the anchor sits in, so
        # the new "§ Na" follows "§ N" rather than splitting it.
        end = min([row for row in heading_rows if row > anchor] + [boundary])
        at = end - 1
        while at > anchor and not lines[at].strip():
            at -= 1
        return at

    # Carry to the end of the anchoring section, but never past the section
    # the skeleton puts after this one.
    while anchor + 1 < boundary and lines[anchor + 1].strip():
        anchor += 1
    return anchor


def insert_drafts(text: str, drafts: list[DraftedSection]) -> str:
    """Put each draft where it belongs, later positions first.

    Back to front so an earlier insertion does not shift the line numbers the
    later ones were computed against.
    """
    lines = text.split("\n")
    # Each insertion goes directly after its anchor, so of two drafts sharing
    # one the later-ranked must go in first to end up second. Measured on the
    # first version: "rozwiązanie" (skeleton position 6) landed before
    # "odpowiedzialność" (position 4) because a tie in `after_line` fell back
    # on list order.
    for draft in sorted(drafts, key=lambda row: (-row.after_line, -row.rank)):
        at = min(max(draft.after_line, -1), len(lines) - 1)
        lines[at + 1 : at + 1] = list(draft.lines)
    return "\n".join(lines)


def inserted_line_indices(drafts: list[DraftedSection]) -> list[int]:
    """Where the lines `insert_drafts` added sit in its result.

    A DOCX has one paragraph per line of the text the engine sees, and the
    writer needs to tell the paragraphs it rewrites from the ones it has to
    create.
    """
    rows: list[int] = []
    shift = 0
    for draft in sorted(drafts, key=lambda row: (row.after_line, row.rank)):
        start = draft.after_line + 1 + shift
        rows.extend(range(start, start + len(draft.lines)))
        shift += len(draft.lines)
    return rows


def drafted_payload(result: DraftingResult) -> list[dict[str, Any]]:
    return [
        {
            "section_id": draft.section_id,
            "label_pl": draft.label_pl,
            "heading": draft.heading,
            "text": draft.text,
            "expects": list(draft.expects),
            "after_line": draft.after_line,
            "rank": draft.rank,
            "blanks": draft.blanks,
            # False once a writer had to give up on the insertion - the
            # proposal is still reported, as a proposal.
            "inserted": True,
        }
        for draft in result.drafts
    ]


def drafts_from_payload(rows: list[dict[str, Any]]) -> list[DraftedSection]:
    return [
        DraftedSection(
            section_id=row["section_id"],
            label_pl=row["label_pl"],
            text=row["text"],
            expects=tuple(row.get("expects") or ()),
            after_line=int(row["after_line"]),
            rank=int(row.get("rank", 0)),
            heading=row.get("heading", ""),
        )
        for row in rows
    ]
