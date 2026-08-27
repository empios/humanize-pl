"""Log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008).

Replaces hand-written pattern lists with measurement. Raw frequency ratios are
useless here: rare n-grams produce enormous ratios on a handful of occurrences,
and the engine's existing patterns were picked by intuition, which is how the
rule set ended up matching the benchmark fixtures and little else.

The prior is the pooled corpus, so a term is only surprising relative to how
often it appears overall. The reported z-score is the usable ranking key.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import regex as re

from humanize_pl.sentence_splitter import split_sentences

WORD_RE = re.compile(r"\p{L}+")

# z above which a difference is worth looking at. 1.96 is the nominal 5% level;
# with thousands of candidate n-grams the real false-discovery rate is much
# higher, so treat this as a ranking cut, not a significance test.
Z_THRESHOLD = 1.96


@dataclass(frozen=True)
class TermScore:
    term: str
    z_score: float
    log_odds: float
    count_a: int
    count_b: int
    rate_a_per_1000: float
    rate_b_per_1000: float

    @property
    def favours(self) -> str:
        return "a" if self.z_score > 0 else "b"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 1:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_openings(text: str, *, length: int = 2) -> list[str]:
    """First `length` tokens of each sentence.

    Openings are counted separately from ordinary n-grams because AI-style
    monotony shows up at the start of sentences far more than in the middle,
    and pooling the two buries the signal.
    """
    out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue
        for sentence in split_sentences(paragraph):
            tokens = tokenize(sentence)[:length]
            if len(tokens) == length:
                out.append(" ".join(tokens))
    return out


def count_terms(texts: list[str], *, n: int, openings: bool = False) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        if openings:
            counts.update(sentence_openings(text, length=n))
        else:
            counts.update(ngrams(tokenize(text), n))
    return counts


def log_odds_with_prior(
    counts_a: Counter[str],
    counts_b: Counter[str],
    *,
    min_count: int = 5,
    prior_scale: float = 1.0,
) -> list[TermScore]:
    """Rank terms by how much more characteristic they are of A than of B.

    `min_count` is the combined-count floor. Without it the ranking fills with
    hapax legomena whose z-scores are noise.
    """
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    if not total_a or not total_b:
        return []

    pooled = Counter(counts_a)
    pooled.update(counts_b)
    prior_total = sum(pooled.values()) * prior_scale

    scored: list[TermScore] = []
    for term, pooled_count in pooled.items():
        if pooled_count < min_count:
            continue
        count_a = counts_a.get(term, 0)
        count_b = counts_b.get(term, 0)

        alpha = pooled_count * prior_scale
        numerator_a = count_a + alpha
        numerator_b = count_b + alpha
        denominator_a = total_a + prior_total - numerator_a
        denominator_b = total_b + prior_total - numerator_b
        if denominator_a <= 0 or denominator_b <= 0:
            continue

        log_odds = math.log(numerator_a / denominator_a) - math.log(
            numerator_b / denominator_b
        )
        variance = 1.0 / numerator_a + 1.0 / numerator_b
        z_score = log_odds / math.sqrt(variance)

        scored.append(
            TermScore(
                term=term,
                z_score=round(z_score, 4),
                log_odds=round(log_odds, 4),
                count_a=count_a,
                count_b=count_b,
                rate_a_per_1000=round(count_a / total_a * 1000, 4),
                rate_b_per_1000=round(count_b / total_b * 1000, 4),
            )
        )

    return sorted(scored, key=lambda score: score.z_score, reverse=True)
