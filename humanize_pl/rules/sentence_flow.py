from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from humanize_pl.nlp.morfeusz import try_load_morfeusz
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


_WORD_RE = re.compile(r"\p{L}+")
_NOT_ONLY_A_CONJUNCTION = {"subst", "adj", "prep", "pred"}


def _opens_subordinate_clause(text: str) -> bool:
    """True if *text* opens a subordinate clause ("że", "skoro", "ponieważ", a
    relative "który") or Morfeusz is missing to tell.

    A connector after such a clause may continue it: in "wskazał, że A,
    ponieważ B" the reason is part of what was said, and a split would turn it
    into the writer's own claim; "Skoro A, a także B, to C" would lose its
    consequence. A word that can also be a pronoun or a preposition ("to",
    "co do") is not taken for one.
    """
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return True
    for word in _WORD_RE.findall(text):
        analyses = morfeusz.analyses(word)
        heads = {row.tag.split(":")[0] for row in analyses}
        if "comp" in heads and not heads & _NOT_ONLY_A_CONJUNCTION:
            return True
        if any(row.lemma == "który" for row in analyses):
            return True
    return False


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
    # A connector opening a clause follows a comma (", jednak"); without one it
    # belongs to a phrase ("ale jednak") and the split would leave "…, ale.".
    if not body[:idx].rstrip().endswith(","):
        return None

    left = body[:idx].strip(" ,;")
    right = body[idx + len(marker):].strip(" ,;")

    if len(left.split()) < min_left or len(right.split()) < min_right:
        return None
    # The new sentence needs its own predicate before its first comma: in
    # ", a także o zmianach, które mogą ..." the only verb is in a relative
    # clause and the split leaves a fragment.
    if not has_finite_verb(left) or not has_finite_verb(right.split(",", 1)[0]):
        return None
    # Do not split list complements starting with a bare preposition.
    if right.lower().startswith(("za ", "w ", "na ", "do ", "od ", "przy ", "bez ", "pod ")):
        return None
    if _opens_subordinate_clause(left):
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
