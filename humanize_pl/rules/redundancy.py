from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import regex as re

import yaml

from humanize_pl.config import Mode
from humanize_pl.nlp.morfeusz import try_load_morfeusz
from humanize_pl.safety.anchors import content_anchor_tokens
from .base import Candidate
from .features import LEGAL_REF_PATTERN, ParagraphFeatures
from .finite_verbs import FINITE_VERB_SUFFIXES, FINITE_VERB_WORDS, PAST_TENSE_SUFFIXES

_PAIRS_YAML = Path(__file__).parent / "tautological_pairs.yaml"


@lru_cache(maxsize=1)
def _load_tautological_pairs() -> frozenset[frozenset[str]]:
    with open(_PAIRS_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    pairs: set[frozenset[str]] = set()
    for section in ("wordnet_pairs", "kancelaryjna_formulas"):
        for pair in data.get(section) or []:
            a, b = str(pair[0]), str(pair[1])
            if a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3:
                pairs.add(frozenset({a, b}))
    return frozenset(pairs)


@lru_cache(maxsize=1)
def _load_tautological_noun_pairs() -> frozenset[frozenset[str]]:
    with open(_PAIRS_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    pairs: set[frozenset[str]] = set()
    for pair in data.get("noun_wordnet_pairs") or []:
        a, b = str(pair[0]), str(pair[1])
        if a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3:
            pairs.add(frozenset({a, b}))
    return frozenset(pairs)


COMMON_OPENING_VERBS = {
    "chroni",
    "dotyczy",
    "ma",
    "mają",
    "obejmuje",
    "określa",
    "oznacza",
    "pozwala",
    "pozostaje",
    "przewiduje",
    "reguluje",
    "stanowi",
    "umożliwia",
    "wskazuje",
    "wynika",
}

SAFE_CONNECTORS = {"jednak", "natomiast", "ponadto", "dlatego"}
WORD_RE = re.compile(r"\p{L}+")

# Match: word  (i|oraz|a także)  word   — word boundaries via \b
_ADJ_CONJ_ADJ_RE = re.compile(
    r"\b(\p{L}+)\b(\s+(?:i|oraz|a\s+także)\s+)\b(\p{L}+)\b",
    re.IGNORECASE,
)


def redundancy_candidates(
    sentence: str,
    *,
    previous_sentence: str | None,
    mode: Mode,
    paragraph_features: ParagraphFeatures | None,
) -> list[Candidate]:
    if mode == Mode.conservative:
        return []

    out: list[Candidate] = []
    out.extend(_intra_sentence_redundancy(sentence, mode=mode))

    if previous_sentence:
        out.extend(
            _inter_sentence_redundancy(
                sentence,
                previous_sentence=previous_sentence,
                mode=mode,
                paragraph_features=paragraph_features,
            )
        )

    # De-duplicate strong-mode variants that may converge to the same text.
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in out:
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        unique.append(candidate)
    return unique


def _inter_sentence_redundancy(
    sentence: str,
    *,
    previous_sentence: str,
    mode: Mode,
    paragraph_features: ParagraphFeatures | None,
) -> list[Candidate]:
    if paragraph_features and paragraph_features.repeated_anchor_count <= 0:
        return []

    previous_anchors = content_anchor_tokens(previous_sentence)
    if not previous_anchors:
        return []

    words = list(WORD_RE.finditer(sentence))
    if len(words) < 3:
        return []

    out: list[Candidate] = []
    prefix_lengths = (4, 3, 2) if mode == Mode.standard else (4, 3, 2, 1)
    for prefix_len in prefix_lengths:
        if len(words) <= prefix_len:
            continue
        prefix_matches = words[:prefix_len]
        prefix_text = sentence[: prefix_matches[-1].end()]
        if "__PROTECTED_" in prefix_text or LEGAL_REF_PATTERN.search(prefix_text):
            continue
        prefix_anchors = content_anchor_tokens(prefix_text)
        if len(prefix_anchors) != prefix_len:
            continue
        if not prefix_anchors <= previous_anchors:
            continue

        remainder = sentence[prefix_matches[-1].end() :].lstrip(" ,;:-")
        if not _starts_safely(remainder):
            continue
        candidate_text = _sentence_case(remainder)
        if candidate_text and candidate_text != sentence:
            out.append(
                Candidate(
                    candidate_text,
                    "redundancy:drop_repeated_opening",
                    0.52 if prefix_len >= 2 else 0.46,
                    stage="adaptive_scoring",
                    operation_type="redundancy_reduction",
                    risk=0.18 if prefix_len >= 2 else 0.23,
                )
            )
            if mode == Mode.standard:
                break
    return out


_NOUN_CONJ_NOUN_RE = re.compile(
    r"\b(\p{L}+)\b(\s+(?:i|oraz|a\s+także)\s+)\b(\p{L}+)\b",
    re.IGNORECASE,
)


def _intra_sentence_redundancy(sentence: str, *, mode: Mode) -> list[Candidate]:
    out = list(_repeated_subject_candidates(sentence, mode=mode))
    out.extend(_tautological_adj_candidates(sentence, mode=mode))
    out.extend(_tautological_noun_candidates(sentence, mode=mode))
    return out


def _repeated_subject_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    pattern = re.compile(
        r"(?P<head>\b(?P<subject>\p{Lu}?\p{L}+(?:\s+\p{L}+){0,2})\s+"
        r"(?P<verb>\p{L}+)\b(?P<middle>[^.!?]{8,160}?),\s+"
        r"(?P<conj>oraz|a także|i)\s+(?P=subject)\s+"
        r"(?P<nextverb>\p{L}+)\b)",
        re.IGNORECASE,
    )
    out: list[Candidate] = []
    for match in pattern.finditer(sentence):
        subject = match.group("subject")
        if "__PROTECTED_" in subject or LEGAL_REF_PATTERN.search(subject):
            continue
        subject_anchors = content_anchor_tokens(subject)
        if not subject_anchors:
            continue
        if not _looks_like_finite_verb(match.group("verb").lower()):
            continue
        if not _looks_like_finite_verb(match.group("nextverb").lower()):
            continue
        replacement = (
            f"{subject} {match.group('verb')}{match.group('middle')}, "
            f"{match.group('conj')} {match.group('nextverb')}"
        )
        candidate_text = sentence[: match.start()] + replacement + sentence[match.end() :]
        if candidate_text != sentence:
            out.append(
                Candidate(
                    candidate_text,
                    "redundancy:drop_repeated_subject_in_sentence",
                    0.50 if mode == Mode.standard else 0.54,
                    stage="adaptive_scoring",
                    operation_type="redundancy_reduction",
                    risk=0.20,
                )
            )
            break
    return out


def _tautological_adj_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    """Drop the first of two adjacent synonymous adjectives joined by a conjunction.

    Uses Morfeusz for lemmatisation so the rule is inflection-agnostic.
    Degrades gracefully to a no-op when Morfeusz is unavailable.
    """
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return []

    out: list[Candidate] = []
    seen_pairs: set[frozenset[str]] = set()

    for match in _ADJ_CONJ_ADJ_RE.finditer(sentence):
        w1, w2 = match.group(1), match.group(3)
        if "__PROTECTED_" in w1 or "__PROTECTED_" in w2:
            continue

        lemmas1 = {a.lemma for a in morfeusz.analyses(w1) if a.upos == "ADJ"}
        lemmas2 = {a.lemma for a in morfeusz.analyses(w2) if a.upos == "ADJ"}
        if not lemmas1 or not lemmas2:
            continue

        found_pair: frozenset[str] | None = None
        for l1 in lemmas1:
            for l2 in lemmas2:
                pair = frozenset({l1, l2})
                if pair in _load_tautological_pairs() and pair not in seen_pairs:
                    found_pair = pair
                    break
            if found_pair:
                break
        if not found_pair:
            continue

        seen_pairs.add(found_pair)
        # Drop "w1 conj " — keep only w2 and everything after.
        candidate_text = sentence[: match.start()] + sentence[match.start(3):]
        candidate_text = _sentence_case(candidate_text.strip())
        if candidate_text and candidate_text != sentence:
            out.append(
                Candidate(
                    candidate_text,
                    "redundancy:drop_tautological_adj_pair",
                    0.50 if mode == Mode.standard else 0.54,
                    stage="adaptive_scoring",
                    operation_type="redundancy_reduction",
                    risk=0.18,
                )
            )

    return out


def _tautological_noun_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    """Drop the first of two adjacent synonymous nouns joined by a conjunction."""
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return []

    noun_pairs = _load_tautological_noun_pairs()
    if not noun_pairs:
        return []

    out: list[Candidate] = []
    seen_pairs: set[frozenset[str]] = set()

    for match in _NOUN_CONJ_NOUN_RE.finditer(sentence):
        w1, w2 = match.group(1), match.group(3)
        if "__PROTECTED_" in w1 or "__PROTECTED_" in w2:
            continue

        lemmas1 = {a.lemma for a in morfeusz.analyses(w1) if a.upos == "NOUN"}
        lemmas2 = {a.lemma for a in morfeusz.analyses(w2) if a.upos == "NOUN"}
        if not lemmas1 or not lemmas2:
            continue

        found_pair: frozenset[str] | None = None
        for l1 in lemmas1:
            for l2 in lemmas2:
                pair = frozenset({l1, l2})
                if pair in noun_pairs and pair not in seen_pairs:
                    found_pair = pair
                    break
            if found_pair:
                break
        if not found_pair:
            continue

        seen_pairs.add(found_pair)
        candidate_text = sentence[: match.start()] + sentence[match.start(3):]
        candidate_text = _sentence_case(candidate_text.strip())
        if candidate_text and candidate_text != sentence:
            out.append(
                Candidate(
                    candidate_text,
                    "redundancy:drop_tautological_noun_pair",
                    0.50 if mode == Mode.standard else 0.54,
                    stage="adaptive_scoring",
                    operation_type="redundancy_reduction",
                    risk=0.20,
                )
            )

    return out


def mark_redundancy_candidate(candidate: Candidate) -> Candidate:
    return replace(
        candidate,
        stage=candidate.stage or "adaptive_scoring",
        operation_type="redundancy_reduction",
        risk=max(candidate.risk, 0.18),
    )


def _starts_safely(text: str) -> bool:
    words = [word.lower() for word in WORD_RE.findall(text)]
    if not words:
        return False
    if words[0] in SAFE_CONNECTORS:
        return len(words) >= 2 and _looks_like_finite_verb(words[1])
    return _looks_like_finite_verb(words[0])


def _looks_like_finite_verb(word: str) -> bool:
    if word in FINITE_VERB_WORDS or word in COMMON_OPENING_VERBS:
        return True
    if len(word) >= 6 and word.endswith(FINITE_VERB_SUFFIXES):
        return True
    if len(word) >= 5 and word.endswith(PAST_TENSE_SUFFIXES):
        return True
    return False


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[:1].upper() + text[1:]
