"""The edits the rhythm layer is allowed to make, and the ones it refuses.

Two kinds, both boundary moves: joining or splitting sentences, and joining or
splitting paragraphs. Nothing here rewrites words. That is the property the
guards enforce and the reason the existing safety gates can validate these
candidates unchanged - the content on both sides of a merge is the same
content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re

from humanize_pl.blueprint import is_heading
from humanize_pl.config import Mode
from humanize_pl.rules.sentence_flow import sentence_flow_candidates
from humanize_pl.safety.validators import SENTENCE_TRANSITIONS

# A sentence opening that `sentence_flow` produced, mapped back to the marker
# it replaced. Merging along this list is the exact inverse of the split the
# engine already performs, which is why it is the first choice: the shape it
# restores is one the engine was willing to create.
_TRANSITION_TO_MARKER: dict[str, str] = {
    "Z kolei": "natomiast",
    "Jednak": "jednak",
    "Przy czym": "przy czym",
    "Ponadto": "a także",
}

# A legal reference this close to the join makes the join unsafe, mirroring
# `sentence_flow._legal_ref_near_split`.
_LEGAL_REF_RE = re.compile(
    r"\bart\.\b|§|\bust\.\b|\bpkt\b|\bDz\.\s*U\.|\bKodeksu\b|\bKonstytucji\b"
    r"|__PROTECTED_\d+__",
    re.IGNORECASE,
)

# Deontic verbs, split by force. Merging across the boundary leaves one
# sentence carrying two different obligations, and the existing gate cannot
# see it: `deontic_modality_preserved` compares totals, and a merge preserves
# every total exactly.
_PERMISSION = re.compile(r"\b(?:może|mogą|jest uprawnion\p{L}*|ma prawo)\b", re.IGNORECASE)
_OBLIGATION = re.compile(
    r"\b(?:musi|muszą|zobowiązuj\p{L}*|jest zobowiązan\p{L}*|należy|powinien|powinna|powinny)\b",
    re.IGNORECASE,
)

# A paragraph that opens on one of these is tied to the one before it, so a
# split here would strand the reference.
_ANAPHORIC_OPENER = re.compile(
    r"^\s*(?:W\s+konsekwencji|W\s+takim\s+przypadku|Dlatego|Z\s+tego\s+względu"
    r"|Tym\s+samym|W\s+związku\s+z\s+powyższym|Jednocześnie|Ponadto"
    r"|Ten|Ta|To|Taki|Taka|Powyższy|Powyższa|Niniejszy|Przedmiotowy)\b",
    re.IGNORECASE,
)

# A numbered or lettered item, and a unit marker. Neither is prose, and
# joining them changes the structure of a norm rather than its rhythm.
_ENUMERATION = re.compile(r"^\s*(?:\d+[.)]|[a-ząćęłńóśźż][.)]|[ivxlcdm]+[.)]|§)", re.IGNORECASE)

_WORD_RE = re.compile(r"\p{L}+")


@dataclass(frozen=True)
class RhythmOperation:
    """One candidate edit, with the text it replaces and the text it becomes."""

    kind: str  # sentence_merge | sentence_split | paragraph_merge | paragraph_split
    rule: str
    paragraph_index: int
    sentence_index: int
    original: str
    candidate: str
    # New sentence lengths and paragraph shape if applied, so the runner can
    # score a candidate without building the whole document.
    preview_lengths: list[int]
    preview_shape: list[int]


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def is_enumeration(text: str) -> bool:
    return bool(_ENUMERATION.match(text.strip()))


def mixes_deontic_scope(left: str, right: str) -> bool:
    """True when joining would put a permission and an obligation in one sentence.

    Necessary because `validate_candidate` cannot catch it: the gate compares
    counts per modality category and a merge changes none of them. What
    changes is which subject each modality governs, and that is invisible to a
    count.
    """
    left_perm, left_obl = bool(_PERMISSION.search(left)), bool(_OBLIGATION.search(left))
    right_perm, right_obl = bool(_PERMISSION.search(right)), bool(_OBLIGATION.search(right))
    return (left_perm and right_obl) or (left_obl and right_perm)


def token_multiset_preserved(original: str, candidate: str) -> bool:
    """A boundary move may change punctuation and one joining word, nothing else.

    Checked on lowercased word tokens. A merge is allowed to add one marker
    ("natomiast", "przy czym") and to lowercase the second sentence's first
    letter; anything beyond that means the operation rewrote content, which is
    a bug in the operation rather than a matter of taste.
    """
    before = [token.lower() for token in _WORD_RE.findall(original)]
    after = [token.lower() for token in _WORD_RE.findall(candidate)]
    if len(after) > len(before) + 2 or len(after) < len(before) - 2:
        return False
    missing = _counter_diff(before, after)
    added = _counter_diff(after, before)
    allowed = {"natomiast", "jednak", "przy", "czym", "a", "także", "z", "kolei", "ponadto"}
    return all(word in allowed for word in missing) and all(word in allowed for word in added)


def _counter_diff(left: list[str], right: list[str]) -> list[str]:
    from collections import Counter

    return list((Counter(left) - Counter(right)).elements())


def _legal_ref_near(left: str, right: str, window: int = 6) -> bool:
    tail = " ".join(left.split()[-window:])
    head = " ".join(right.split()[:window])
    return bool(_LEGAL_REF_RE.search(tail + " " + head))


def merge_candidates(
    sentences: list[str],
    *,
    paragraph_index: int,
    limits: Any,
    lengths: list[int],
    shape: list[int],
    offset: int,
    excluded: frozenset[int],
) -> list[RhythmOperation]:
    """Join two short adjacent sentences, without changing what they say."""
    out: list[RhythmOperation] = []
    for index in range(len(sentences) - 1):
        if index in excluded or index + 1 in excluded:
            continue
        left, right = sentences[index].strip(), sentences[index + 1].strip()
        if not left or not right:
            continue
        left_words, right_words = word_count(left), word_count(right)
        # The cap is on the result. Capping the parts instead would admit only
        # the merges that lower the metric - see ModeLimits.merge_max_result.
        if left_words + right_words > limits.merge_max_result:
            continue
        # A colon announces a list; joining across it buries the announcement.
        if left.endswith(":"):
            continue
        if is_enumeration(left) or is_enumeration(right):
            continue
        if _legal_ref_near(left, right):
            continue
        if mixes_deontic_scope(left, right):
            continue

        joined = _join(left, right)
        if joined is None:
            continue

        new_lengths = list(lengths)
        position = offset + index
        new_lengths[position : position + 2] = [left_words + right_words]
        new_shape = list(shape)
        new_shape[paragraph_index] = max(1, new_shape[paragraph_index] - 1)

        out.append(
            RhythmOperation(
                kind="sentence_merge",
                rule="rhythm:merge_sentences",
                paragraph_index=paragraph_index,
                sentence_index=index,
                original=f"{left} {right}",
                candidate=joined,
                preview_lengths=new_lengths,
                preview_shape=new_shape,
            )
        )
    return out


def _join(left: str, right: str) -> str | None:
    """Preferred device first: undo a transition, else a semicolon."""
    stem = left[:-1].rstrip() if left.endswith((".", "!", "?")) else left
    if not stem:
        return None

    for transition, marker in _TRANSITION_TO_MARKER.items():
        if right.startswith(transition + " "):
            rest = right[len(transition) + 1 :]
            if not rest:
                return None
            return f"{stem}, {marker} {rest[0].lower()}{rest[1:]}"

    # A semicolon changes no word and no inflection, which makes it the safest
    # device available - but it is rare in Polish legal prose outside lists,
    # so the runner caps how often it may be used.
    for opener in SENTENCE_TRANSITIONS:
        if right.startswith(opener):
            return None
    return f"{stem}; {right[0].lower()}{right[1:]}"


def split_candidates(
    sentences: list[str],
    *,
    paragraph_index: int,
    mode: Mode,
    limits: Any,
    lengths: list[int],
    shape: list[int],
    offset: int,
) -> list[RhythmOperation]:
    """Split a long sentence, reusing the engine's own split points only.

    The four markers in `sentence_flow` are an accepted risk surface with a
    legal-reference guard already on them. Widening it - splitting on a comma,
    on a relative clause - is a separate decision needing its own evidence, so
    when `sentence_flow_candidates` returns nothing there is no operation.
    """
    out: list[RhythmOperation] = []
    for index, sentence in enumerate(sentences):
        words = word_count(sentence)
        if words < limits.split_min_words:
            continue
        candidates = sentence_flow_candidates(
            sentence, mode=mode, min_words=limits.split_min_words
        )
        if not candidates:
            continue
        best = candidates[0]
        parts = [part for part in best.text.split(". ") if part.strip()]
        if len(parts) < 2:
            continue
        new_lengths = list(lengths)
        position = offset + index
        new_lengths[position : position + 1] = [word_count(part) for part in parts]
        new_shape = list(shape)
        new_shape[paragraph_index] = new_shape[paragraph_index] + len(parts) - 1
        out.append(
            RhythmOperation(
                kind="sentence_split",
                rule="rhythm:split_sentence",
                paragraph_index=paragraph_index,
                sentence_index=index,
                original=sentence,
                candidate=best.text,
                preview_lengths=new_lengths,
                preview_shape=new_shape,
            )
        )
    return out


def paragraph_merge_candidates(
    paragraphs: list[list[str]],
    *,
    lengths: list[int],
    shape: list[int],
    frozen: frozenset[int],
) -> list[RhythmOperation]:
    """Join two short adjacent paragraphs that are about the same thing.

    Merging never tears an argument apart, which is why it is available in
    `standard` while splitting is not.
    """
    out: list[RhythmOperation] = []
    for index in range(len(paragraphs) - 1):
        if index in frozen or index + 1 in frozen:
            continue
        first, second = paragraphs[index], paragraphs[index + 1]
        if not first or not second:
            continue
        if len(first) > 2 or len(second) > 2:
            continue
        first_text, second_text = " ".join(first), " ".join(second)
        if is_heading(first_text) or is_heading(second_text):
            continue
        if is_enumeration(first_text) or is_enumeration(second_text):
            continue
        if _overlap(first_text, second_text) < 0.25:
            continue
        new_shape = list(shape)
        new_shape[index : index + 2] = [shape[index] + shape[index + 1]]
        out.append(
            RhythmOperation(
                kind="paragraph_merge",
                rule="rhythm:merge_paragraphs",
                paragraph_index=index,
                sentence_index=0,
                original=f"{first_text}\n{second_text}",
                candidate=f"{first_text} {second_text}",
                preview_lengths=list(lengths),
                preview_shape=new_shape,
            )
        )
    return out


def paragraph_split_candidates(
    paragraphs: list[list[str]],
    *,
    lengths: list[int],
    shape: list[int],
    frozen: frozenset[int],
) -> list[RhythmOperation]:
    """Cut a long paragraph at its weakest topical seam."""
    out: list[RhythmOperation] = []
    for index, sentences in enumerate(paragraphs):
        if index in frozen or len(sentences) < 5:
            continue
        text = " ".join(sentences)
        if is_heading(text) or is_enumeration(text):
            continue

        best_at, best_score = None, 1.0
        for cut in range(2, len(sentences) - 1):
            if _ANAPHORIC_OPENER.match(sentences[cut]):
                continue
            score = _overlap(sentences[cut - 1], sentences[cut])
            if score < best_score:
                best_at, best_score = cut, score
        if best_at is None or best_score > 0.10:
            continue

        first = " ".join(sentences[:best_at])
        second = " ".join(sentences[best_at:])
        new_shape = list(shape)
        new_shape[index : index + 1] = [best_at, len(sentences) - best_at]
        out.append(
            RhythmOperation(
                kind="paragraph_split",
                rule="rhythm:split_paragraph",
                paragraph_index=index,
                sentence_index=best_at,
                original=text,
                candidate=f"{first}\n{second}",
                preview_lengths=list(lengths),
                preview_shape=new_shape,
            )
        )
    return out


def _overlap(left: str, right: str) -> float:
    """Jaccard overlap of content anchors, the measure paragraph features use."""
    from humanize_pl.safety.anchors import content_anchor_tokens

    first = set(content_anchor_tokens(left))
    second = set(content_anchor_tokens(right))
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)
