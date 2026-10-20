"""Deciding what an incoming letter is and how it can be answered.

Three findings, each from the letter's own form: the act (its title and the
words by which it rules or demands), the sender (whose names stand in the head)
and the procedure (the codes it cites, else the sender's). The replies follow
from those, and the instruction in the letter comes first: a remedy it names
is offered ahead of the catalogue, with the letter's term, and a term that
differs from the statute is reported rather than silently replaced.

Given the delivery date, each term becomes a last day to file. When one has
passed, a request to restore it leads the list.

Nothing here writes or files anything. The user picks the reply; a letter
nothing recognises still gets the general reply, never a guessed one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

from humanize_pl.replies.catalogue import Catalogue, Reply, Signals, Term, catalogue
from humanize_pl.replies.deadlines import AFTER_TERM, Deadline, compute_deadline
from humanize_pl.replies.letter import (
    Facts,
    Letter,
    Remedy,
    head_of,
    is_title,
    read,
    remedies_in,
    term_in,
)

# A title names the act outright; a ruling verb nearly does; a word only nudges.
HEADER_WEIGHT = 5
STRONG_WEIGHT = 3
SIGNAL_WEIGHT = 1
# A name in the head is the sender's; further down it is only quoted.
HEAD_FACTOR = 2
# Below this a finding is a couple of common words, not a finding.
MIN_SCORE = 3
# Offered first once a term has passed.
RESTORE_TERM = "wniosek_o_przywrocenie_terminu"


@dataclass(frozen=True)
class Guess:
    id: str | None
    label_pl: str
    score: int = 0
    evidence: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label_pl": self.label_pl,
            "score": self.score,
            "evidence": list(self.evidence),
        }


NOT_FOUND = Guess(None, "nierozpoznane")


@dataclass(frozen=True)
class Candidate:
    reply: Reply
    goal_pl: str
    from_instruction: bool
    term: Term | None
    # "pouczenie", "pismo" or "przepis": where the term comes from.
    term_source: str
    basis: str
    addressee: str
    via: str
    warnings: tuple[str, ...]
    conflict: str = ""
    deadline: Deadline | None = None

    def to_json(self) -> dict:
        return {
            "id": self.reply.id,
            "label_pl": self.reply.label_pl,
            "category": self.reply.category,
            "goal": self.reply.goal,
            "goal_pl": self.goal_pl,
            "from_instruction": self.from_instruction,
            "term": self.term.to_json() if self.term else None,
            "term_source": self.term_source,
            "basis": self.basis,
            "addressee": self.addressee,
            "via": self.via,
            "warnings": list(self.warnings),
            "conflict": self.conflict,
            "deadline": self.deadline.to_json() if self.deadline else None,
        }


@dataclass(frozen=True)
class IncomingAnalysis:
    act: Guess
    sender: Guess
    procedure: Guess
    remedies: tuple[Remedy, ...]
    facts: Facts
    candidates: tuple[Candidate, ...]
    # "wysoka": the title and the instruction agree; "średnia": the act is known;
    # "niska": only the general reply is safe.
    confidence: str
    # The head line in which the sender names itself, as the letter writes it:
    # whom a reply goes back to.
    sender_name: str = ""

    def to_json(self) -> dict:
        return {
            "act": self.act.to_json(),
            "sender": self.sender.to_json(),
            "procedure": self.procedure.to_json(),
            "remedies": [
                {
                    "phrase": r.phrase,
                    "term": r.term.to_json() if r.term else None,
                    "addressee": r.addressee,
                    "via": r.via,
                    "quote": r.quote,
                }
                for r in self.remedies
            ],
            "facts": self.facts.to_json(),
            "candidates": [c.to_json() for c in self.candidates],
            "confidence": self.confidence,
            "sender_name": self.sender_name,
        }


def _signal_score(signals: Signals, letter: Letter, head_factor: int = 1) -> tuple[int, list[str]]:
    total, evidence = 0, []
    for weight, patterns in ((STRONG_WEIGHT, signals.strong), (SIGNAL_WEIGHT, signals.weak)):
        for phrase, pattern in patterns:
            if pattern.search(letter.head):
                total += weight * head_factor
            elif pattern.search(letter.lowered):
                total += weight
            else:
                continue
            evidence.append(phrase)
    return total, evidence


def _best(scored: dict[str, tuple[int, list[str]]], labels: dict[str, str]) -> Guess:
    """The highest score above the floor; a tie goes to the catalogue's order."""
    ranked = sorted(
        ((score, i, key) for i, (key, (score, _)) in enumerate(scored.items()) if score),
        key=lambda r: (-r[0], r[1]),
    )
    if not ranked or ranked[0][0] < MIN_SCORE:
        return NOT_FOUND
    score, _, key = ranked[0]
    return Guess(key, labels[key], score, tuple(scored[key][1][:5]))


def find_act(letter: Letter, cat: Catalogue) -> Guess:
    scored = {act.id: _signal_score(act.signals, letter) for act in cat.acts.values()}
    # The longest header a title begins with names the act: "wezwanie do zapłaty"
    # is not a summons by an authority, "wyrok zaoczny" not a plain judgment.
    titled = set()
    for title in letter.titles:
        matches = [
            (len(header), act.id)
            for act in cat.acts.values()
            for header in act.header
            if is_title(title, (header,))
        ]
        if matches:
            _, act_id = max(matches)
            if act_id not in titled:
                titled.add(act_id)
                score, evidence = scored[act_id]
                scored[act_id] = (score + HEADER_WEIGHT, [f"tytuł: {title}", *evidence])
    return _best(scored, {a.id: a.label_pl for a in cat.acts.values()})


def find_sender(letter: Letter, cat: Catalogue) -> Guess:
    scored = {s.id: _signal_score(s.signals, letter, HEAD_FACTOR) for s in cat.senders.values()}
    return _best(scored, {s.id: s.label_pl for s in cat.senders.values()})


# A court's division names its court on the line above ("Sąd Rejonowy…" /
# "I Wydział Cywilny"); the division alone does not say where to write.
DIVISION = re.compile(r"(?:[ivxlc]+ )?wydział(?!\w)")
COURT = re.compile(r"(?<!\w)sąd(?!\w)")


def sender_name(text: str, headers: tuple[str, ...], cat: Catalogue, sender: Guess) -> str:
    """The first head line carrying one of the sender's own names ("Naczelnik
    Urzędu Skarbowego…"); words that only suggest it ("podatnik") name no one."""
    if not sender.id:
        return ""
    names = cat.senders[sender.id].signals.strong
    lines = [" ".join(line.split()) for line in head_of(text, headers).splitlines()]
    for i, line in enumerate(lines):
        if any(pattern.search(line.casefold()) for _, pattern in names):
            if i and DIVISION.match(line.casefold()) and COURT.search(lines[i - 1].casefold()):
                return f"{lines[i - 1]}, {line}"
            return line
    return ""


def find_procedure(letter: Letter, cat: Catalogue, sender: Guess) -> Guess:
    """The sender's procedure when the letter cites it, else the most cited one,
    else the sender's: a decision quotes the code it was issued under."""
    cited = Counter(
        {
            p.id: sum(len(pattern.findall(letter.lowered)) for _, pattern in p.signals)
            for p in cat.procedures.values()
        }
    )
    cited = +cited
    own = cat.senders[sender.id].procedure if sender.id else None
    if own and own in cited:
        chosen, why = own, "cytowane przepisy i nadawca"
    elif cited:
        chosen, why = cited.most_common(1)[0][0], "cytowane przepisy"
    elif own:
        chosen, why = own, "nadawca"
    else:
        return NOT_FOUND
    return Guess(chosen, cat.procedures[chosen].label_pl, cited.get(chosen, 0), (why,))


def _days(term: Term) -> tuple[str, int]:
    return (
        ("month", term.value)
        if term.unit == "month"
        else ("day", term.value * (7 if term.unit == "week" else 1))
    )


def _holiday_rule(cat: Catalogue, basis: str, procedure: str | None) -> str:
    """The rule of the code the term's basis cites (a complaint to the
    administrative court counts under p.p.s.a. even against a k.p.a. decision),
    else of the letter's procedure."""
    cited = basis.casefold()
    for p in cat.procedures.values():
        if any(pattern.search(cited) for _, pattern in p.signals):
            return p.holiday_rule
    return cat.procedures[procedure].holiday_rule if procedure else ""


def _candidate(
    reply: Reply,
    cat: Catalogue,
    procedure: str | None,
    letter: Letter,
    remedy: Remedy | None,
    delivered: date | None,
    today: date | None,
) -> Candidate:
    statute, basis = reply.term_for(procedure)
    term, source, conflict = statute, "przepis" if statute else "", ""
    if reply.term_from_letter:
        term = term_in(letter.text)
        source = "pismo" if term else ""
    if remedy and remedy.term:
        if statute and _days(statute) != _days(remedy.term):
            conflict = (
                f"Pouczenie podaje {remedy.term.label_pl()}, a przepis ({basis or 'katalog'}) "
                f"{statute.label_pl()}. Sprawdź termin przed złożeniem pisma."
            )
        term, source = remedy.term, "pouczenie"
    return Candidate(
        reply=reply,
        goal_pl=cat.goals[reply.goal],
        from_instruction=remedy is not None,
        term=term,
        term_source=source,
        basis=basis,
        addressee=remedy.addressee if remedy else "",
        via=(remedy.via if remedy and remedy.via else reply.filed_via),
        warnings=tuple(cat.warnings[w] for w in reply.warnings),
        conflict=conflict,
        deadline=compute_deadline(term, delivered, _holiday_rule(cat, basis, procedure), today),
    )


def analyze(
    text: str, delivered: date | None = None, *, today: date | None = None
) -> IncomingAnalysis:
    """What the letter is, who sent it, under which procedure, and the replies
    it allows: those its instruction names first, then the catalogue's, then
    the ones that always apply. With the delivery date each term gets its last
    day, counted against `today` (the current date by default)."""
    cat = catalogue()
    headers = tuple(h for act in cat.acts.values() for h in act.header)
    letter = read(text, headers)
    act = find_act(letter, cat)
    sender = find_sender(letter, cat)
    procedure = find_procedure(letter, cat, sender)
    remedies = remedies_in(
        letter.instruction, tuple(p for r in cat.replies.values() for p in r.remedy)
    )

    def offer(reply: Reply, remedy: Remedy | None = None) -> Candidate:
        return _candidate(reply, cat, procedure.id, letter, remedy, delivered, today)

    chosen: dict[str, Candidate] = {}
    for remedy in remedies:
        owners = [r for r in cat.replies.values() if remedy.phrase in r.remedy]
        # The instruction wins: a remedy it names is offered even where the
        # catalogue would not, preferring the owner that fits the letter.
        fitting = (
            [r for r in owners if r.fits(act.id, sender.id, procedure.id)]
            or [r for r in owners if r.answers(act.id)]
            or owners
        )
        for reply in fitting:
            chosen.setdefault(reply.id, offer(reply, remedy))
    general = [r for r in cat.replies.values() if not (r.acts or r.senders or r.procedures)]
    if act.id:
        for reply in cat.replies.values():
            if (
                reply not in general
                and not reply.instruction_only
                and reply.fits(act.id, sender.id, procedure.id)
            ):
                chosen.setdefault(reply.id, offer(reply))
    for reply in general:
        chosen.setdefault(reply.id, offer(reply))

    if any(c.deadline and c.deadline.status == AFTER_TERM for c in chosen.values()):
        chosen = {RESTORE_TERM: chosen.pop(RESTORE_TERM), **chosen}

    named = any(c.from_instruction and c.reply.answers(act.id) for c in chosen.values())
    titled = any(e.startswith("tytuł:") for e in act.evidence)
    confidence = "wysoka" if act.id and titled and named else "średnia" if act.id else "niska"
    return IncomingAnalysis(
        act=act,
        sender=sender,
        procedure=procedure,
        remedies=tuple(remedies),
        facts=letter.facts,
        candidates=tuple(chosen.values()),
        confidence=confidence,
        sender_name=sender_name(letter.text, headers, cat, sender),
    )
