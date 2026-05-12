from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from .base import Candidate
from .finite_verbs import has_finite_verb


_LEGAL_REF_RE = re.compile(
    r"\bart\.\b|§|\bust\.\b|\bpkt\b|\bDz\.\s*U\.|\bKodeksu\b|\bKonstytucji\b"
    r"|__PROTECTED_\d+__",
    re.IGNORECASE,
)


def _strip_final_punct(text: str) -> tuple[str, str]:
    text = text.strip()
    if text and text[-1] in ".!?":
        return text[:-1].rstrip(), text[-1]
    return text, "."


def _legal_ref_near_split(left: str, right: str, window: int = 6) -> bool:
    """Return True if a legal reference appears within *window* words of the split point."""
    left_tail = " ".join(left.split()[-window:])
    right_head = " ".join(right.split()[:window])
    return bool(_LEGAL_REF_RE.search(left_tail + " " + right_head))


def _safe_split_candidate(
    sentence: str,
    marker: str,
    transition: str,
    rule: str = "split_long_sentence",
    base_score: float = 0.45,
    *,
    min_left: int = 12,
    min_right: int = 8,
) -> Candidate | None:
    body, punct = _strip_final_punct(sentence)
    lower = body.lower()
    idx = lower.find(marker)
    if idx <= 0:
        return None

    left = body[:idx].strip(" ,;")
    right = body[idx + len(marker):].strip(" ,;")

    if len(left.split()) < min_left or len(right.split()) < min_right:
        return None
    if not has_finite_verb(left) or not has_finite_verb(right):
        return None
    # Do not split list complements starting with a bare preposition.
    if right.lower().startswith(("za ", "w ", "na ", "do ", "od ", "przy ", "bez ", "pod ")):
        return None
    # Do not split if the left half introduces a list (ends with a colon).
    if ":" in left:
        return None
    # Do not split near legal references — too risky to alter article/section context.
    if _legal_ref_near_split(left, right):
        return None

    text = f"{left}. {transition} {right}{punct}"
    return Candidate(text, rule, base_score)


def sentence_flow_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    out: list[Candidate] = []
    words = sentence.split()
    if len(words) < 32:
        return out

    # Ordered by preference: contrastive/additive first, causal last.
    # "oraz" is intentionally absent — it joins list items in legal Polish.
    markers: list[tuple[str, str, str, float]] = [
        (" natomiast ", "Z kolei", "split_long_sentence", 0.45),
        (" jednak ", "Jednak", "split_long_sentence", 0.45),
        (" przy czym ", "Przy czym", "split_przy_czym", 0.45),
        (" a także ", "Ponadto", "split_long_sentence", 0.45),
    ]

    if mode in {Mode.standard, Mode.strong}:
        markers.append((" ponieważ ", "Wynika to z tego, że", "split_causal", 0.42))

    for marker, transition, rule, score in markers:
        cand = _safe_split_candidate(sentence, marker, transition, rule, score)
        if cand:
            out.append(cand)
            break
    return out
