from __future__ import annotations

from humanize_pl.config import Mode
from .base import Candidate
from .finite_verbs import has_finite_verb


def _strip_final_punct(text: str) -> tuple[str, str]:
    text = text.strip()
    if text and text[-1] in ".!?":
        return text[:-1].rstrip(), text[-1]
    return text, "."


def _safe_split_candidate(sentence: str, marker: str, transition: str) -> Candidate | None:
    body, punct = _strip_final_punct(sentence)
    lower = body.lower()
    idx = lower.find(marker)
    if idx <= 0:
        return None

    left = body[:idx].strip(" ,;")
    right = body[idx + len(marker):].strip(" ,;")

    # Do not split enumerations or short complements such as
    # "w miejscu i czasie oraz za wynagrodzeniem".
    if len(left.split()) < 12 or len(right.split()) < 8:
        return None
    if not has_finite_verb(left) or not has_finite_verb(right):
        return None
    if right.lower().startswith(("za ", "w ", "na ", "do ", "od ", "przy ", "bez ", "pod ")):
        return None

    candidate = f"{left}. {transition} {right}{punct}"
    return Candidate(candidate, "split_long_sentence", 0.45)


def sentence_flow_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    out: list[Candidate] = []
    words = sentence.split()
    if len(words) < 32:
        return out

    # Never split on "oraz". In legal/formal Polish it commonly joins list items.
    markers = [
        (" natomiast ", "Z kolei"),
        (" jednak ", "Jednak"),
        (" a także ", "Ponadto"),
    ]
    if mode == Mode.strong:
        markers.append((" ponieważ ", "Wynika to z tego, że"))

    for marker, transition in markers:
        cand = _safe_split_candidate(sentence, marker, transition)
        if cand:
            out.append(cand)
            break
    return out
