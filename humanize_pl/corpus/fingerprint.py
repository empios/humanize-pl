"""Measured Polish slop fingerprint: n-grams today's models overuse.

Log-odds alone ranks topic as readily as style: a corpus of privacy policies
puts "danych osobowych" at the top. Three filters keep the style and drop the
topic. A candidate must be overused by more than one model (one model's quirk
is not the language's), appear in a fair share of AI documents (a scenario's
subject sits in a few), and stay rarer in held-out human text than in AI text
(a phrase the genre itself uses is not a tell).
"""

from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePath

from humanize_pl.corpus.logodds import Z_THRESHOLD, count_terms, log_odds_with_prior
from humanize_pl.safety.deontic import extract_deontic_profile

# Layer name -> (n, sentence openings only).
KINDS: dict[str, tuple[int, bool]] = {
    "unigram": (1, False),
    "bigram": (2, False),
    "trigram": (3, False),
    "opening_bigram": (2, True),
    "opening_trigram": (3, True),
}
GENRES = ("legal", "general")

MIN_DOC_SHARE = 0.10
MIN_MODELS = 2
# AI rate over the held-out human rate below which a phrase is the genre's own.
MIN_HOLDOUT_RATIO = 2.0
# Per-model support: the model's own log-odds is positive on this many uses.
MIN_MODEL_COUNT = 2

# Tokens of statutory references. A tell built on them would flag citations.
REFERENCE_TOKENS = frozenset(
    {"art", "ust", "pkt", "lit", "nr", "poz", "dz", "kc", "kpc", "kp", "par"}
)


@dataclass(frozen=True)
class SlopCandidate:
    term: str
    kind: str
    genre: str
    z_score: float
    ai_per_1000: float
    human_per_1000: float
    holdout_per_1000: float | None
    ai_doc_share: float
    models: tuple[str, ...]
    protected: bool
    indicative: bool


def scenario_half(file_name: str) -> str:
    """"train" or "holdout", the same for one scenario in every model's folder."""
    return "train" if zlib.crc32(PurePath(file_name).name.encode()) % 2 == 0 else "holdout"


def is_protected(term: str) -> bool:
    """True when the term touches modality or a statutory reference."""
    if REFERENCE_TOKENS.intersection(term.split()):
        return True
    return any(extract_deontic_profile(term).values())


def slop_candidates(
    ai_by_model: dict[str, list[str]],
    human: list[str],
    *,
    genre: str,
    human_holdout: list[str] | None = None,
    min_count: int = 5,
) -> list[SlopCandidate]:
    """Terms the models overuse against `human`, filtered for topic and genre."""
    if genre not in GENRES:
        raise ValueError(f"Nieznany gatunek: {genre}")
    ai_texts = [text for texts in ai_by_model.values() for text in texts]
    if not ai_texts or not human:
        return []
    indicative = len(ai_by_model) < MIN_MODELS
    min_models = min(MIN_MODELS, len(ai_by_model))

    out: list[SlopCandidate] = []
    for kind, (n, openings) in KINDS.items():
        human_counts = count_terms(human, n=n, openings=openings)
        scores = log_odds_with_prior(
            count_terms(ai_texts, n=n, openings=openings), human_counts, min_count=min_count
        )
        significant = [score for score in scores if score.z_score >= Z_THRESHOLD]
        if not significant:
            continue

        supported_by = _model_support(ai_by_model, human_counts, n=n, openings=openings)
        documents = [set(count_terms([text], n=n, openings=openings)) for text in ai_texts]
        holdout = (
            count_terms(human_holdout, n=n, openings=openings) if human_holdout else None
        )
        holdout_total = sum(holdout.values()) if holdout else 0

        for score in significant:
            models = tuple(sorted(supported_by.get(score.term, ())))
            if len(models) < min_models:
                continue
            share = sum(score.term in terms for terms in documents) / len(documents)
            if share < MIN_DOC_SHARE:
                continue
            holdout_rate = None
            if holdout is not None and holdout_total:
                holdout_rate = round(holdout[score.term] / holdout_total * 1000, 4)
                if holdout_rate and score.rate_a_per_1000 / holdout_rate < MIN_HOLDOUT_RATIO:
                    continue
            out.append(
                SlopCandidate(
                    term=score.term,
                    kind=kind,
                    genre=genre,
                    z_score=score.z_score,
                    ai_per_1000=score.rate_a_per_1000,
                    human_per_1000=score.rate_b_per_1000,
                    holdout_per_1000=holdout_rate,
                    ai_doc_share=round(share, 4),
                    models=models,
                    protected=is_protected(score.term),
                    indicative=indicative,
                )
            )
    return sorted(out, key=lambda candidate: candidate.z_score, reverse=True)


def _model_support(
    ai_by_model: dict[str, list[str]], human_counts: Counter[str], *, n: int, openings: bool
) -> dict[str, set[str]]:
    """Term -> models whose own text overuses it against the human corpus."""
    support: dict[str, set[str]] = {}
    for model, texts in ai_by_model.items():
        for score in log_odds_with_prior(
            count_terms(texts, n=n, openings=openings), human_counts, min_count=1
        ):
            if score.log_odds > 0 and score.count_a >= MIN_MODEL_COUNT:
                support.setdefault(score.term, set()).add(model)
    return support
