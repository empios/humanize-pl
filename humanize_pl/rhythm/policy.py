"""When the rhythm layer runs at all, and how far it may go.

Most of this file is refusals. That is deliberate: the layer moves sentence
and paragraph boundaries in legal text, which is the most structural thing the
engine does, and its measured headroom is small - on real model output,
closing both axes completely is worth about 0.045 of calibrated score against
a measurement noise band of 0.035. A layer with that much to gain and that
much to lose should decline whenever it is unsure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from humanize_pl.config import Mode
from humanize_pl.detect.reference import ReferenceProfile


class RhythmScope(str, Enum):
    """Which axes may be used.

    `sentences_only` freezes the paragraph count. It is the default because
    DOCX cannot accept anything else: `DocxInventory.structural_differences`
    compares the paragraph count, and a mismatch makes the flow discard the
    entire rewrite and restore the source. The cost of getting it wrong is
    not a worse paragraph shape, it is every sentence-level change in the
    document silently thrown away.
    """

    sentences_only = "sentences_only"
    full = "full"


@dataclass(frozen=True)
class ModeLimits:
    max_steps: int
    # Cap on the MERGED sentence, not on the two parts.
    #
    # The first version capped each part at 14 words, on the reasoning that
    # short sentences are the safest to join. Measured, that restriction
    # picked exactly the merges that move the metric the wrong way: joining
    # two short sentences replaces two below-mean values with one near-mean
    # value, which lowers the coefficient of variation. On a real document
    # (lengths 6..27, CV 0.45) every short merge dropped CV and only merges
    # producing 28+ words raised it - 20+21 into 41 took CV from 0.45 to 0.60.
    #
    # Raising the cap is also the right move on its own terms: this document
    # averages 14.6 words per sentence against a human median of 18.9, so it
    # is both too uniform AND too clipped, and the same edit fixes both.
    merge_max_result: int
    split_min_words: int
    allow_paragraph_split: bool
    max_semicolons_per_1000: float


MODE_LIMITS: dict[str, ModeLimits] = {
    # Conservative is chosen by people who do not accept structural risk, and
    # editing rhythm is structural by definition. The gate's regeneration
    # constraints remain the whole answer there.
    Mode.conservative.value: ModeLimits(0, 0, 99, False, 0.0),
    # 36 words sits just above the human p95 for sentence length (28.0) and
    # well inside the range the reference corpus actually spans; 48 in strong
    # reaches the long tail that gives human legal prose its spread.
    Mode.standard.value: ModeLimits(8, 36, 26, False, 1.5),
    # Paragraph splitting only in strong: a seam chosen by low anchor overlap
    # can still separate a premise from its conclusion, and the anaphora list
    # catches the typical cases rather than all of them.
    Mode.strong.value: ModeLimits(14, 48, 22, True, 1.5),
}

MIN_SENTENCES = 12
MIN_PARAGRAPHS = 3


@dataclass(frozen=True)
class RhythmDecision:
    run: bool
    limits: ModeLimits
    allow_paragraph_axis: bool
    reason: str | None = None


def decide(
    *,
    mode: Mode,
    scope: RhythmScope,
    profile: ReferenceProfile | None,
    sentence_count: int,
    paragraph_count: int,
) -> RhythmDecision:
    """Whether to run, and with which axes. The reason is reported in Polish."""
    limits = MODE_LIMITS.get(
        mode.value if hasattr(mode, "value") else str(mode), MODE_LIMITS[Mode.standard.value]
    )

    if limits.max_steps == 0:
        return RhythmDecision(False, limits, False, "Tryb conservative: rytmu nie zmieniamy.")
    if profile is None:
        # Without a measured human band there is no target. Chasing a
        # hardcoded 0.83 would reinvent the fabricated precision that
        # calibration refuses everywhere else.
        return RhythmDecision(
            False, limits, False,
            "Brak profilu referencyjnego dla tej rodziny: rytmu nie porównujemy do niczego.",
        )
    if sentence_count < MIN_SENTENCES:
        return RhythmDecision(
            False, limits, False,
            f"Dokument ma {sentence_count} zdań (mniej niż {MIN_SENTENCES}); "
            "rozrzut długości nic tu nie znaczy.",
        )

    allow_paragraph = (
        scope == RhythmScope.full
        and paragraph_count >= MIN_PARAGRAPHS
        and not profile.paragraph_shape_cv.is_empty
    )
    reason = None
    if scope != RhythmScope.full:
        # Said out loud rather than left as a silent capability gap: for DOCX
        # this is the axis that actually separates AI from human writing, and
        # it is the one we cannot use.
        reason = (
            "Oś akapitowa wyłączona: format nie dopuszcza zmiany liczby akapitów."
        )
    elif paragraph_count < MIN_PARAGRAPHS:
        reason = (
            f"Oś akapitowa wyłączona: {paragraph_count} akapity to za mało, "
            "żeby kształt akapitu był mierzalny."
        )

    return RhythmDecision(True, limits, allow_paragraph, reason)
