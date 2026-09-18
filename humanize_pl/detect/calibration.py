from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .base import DocumentDiagnosis
from .reference import ReferenceProfile

PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference_profiles"
DEFAULT_PROFILE = "saos_common_2018_2024"

# Which reference profile calibrates which document family.
#
# Families, not categories. The detector measures register-level statistics —
# sentence length and its variance, opening diversity, family rates — and
# those separate a pleading from a contract far more sharply than they
# separate a pleading from an appeal. Calibrating all 28 legal categories
# would need 28 corpora, and only a few registers have sources that are both
# open and lawfully reusable: court and administrative rulings, procurement
# contract templates, official correspondence.
#
# A family absent from this map is reported uncalibrated rather than
# calibrated against something adjacent. The SAOS profile is court reasoning;
# using it for a contract would compare a document against a register it does
# not belong to and dress the result up as a measurement.
FAMILY_PROFILES: dict[str, str] = {
    "filing_official": DEFAULT_PROFILE,
    # Measured on 51 approved contract-family documents from one firm, all of
    # which score below the review threshold against the SAOS profile - so
    # they read as human by our own measure, which is the only check that
    # matters for a baseline. Median 524 words.
    #
    # It also answers the genre caveat the court-reasoning profile could not:
    # a contract compared against judgments differs in register as well as
    # authorship, and part of any separation came from that. Read it knowing
    # it rests on 51 documents against SAOS's 1804, and on one firm's house
    # style rather than the register at large.
    "contract": "law_firm_contract",
}

# Families whose human p95 is at or near zero need a floor, otherwise a single
# occurrence divides by ~0 and saturates the score on its own.
RATE_FLOOR_PER_1000 = 0.5

# Metrics measured on the reference corpus that point the *opposite* way to the
# English-language literature for this genre pair, because Polish court
# reasoning repeats party names, legal terms and formulaic openings far more
# than an AI-drafted opinion does. They are reported for transparency and
# excluded from the score: using them would penalise human writing.
GENRE_CONFOUNDED = {"type_token_ratio", "opening_diversity"}

# Operating point measured on 599 held-out SAOS judgments (not used to build
# the profile) against 9 AI-generated legal documents:
#
#   threshold  recall(AI)  FPR(human)
#   0.15       100%        1.17%
#   0.20       100%        0.67%
#   0.25       100%        0.00%    <- REVIEW_THRESHOLD
#
# The populations do not overlap: human max 0.215, AI min 0.332, and the
# threshold sits in that gap. The structural families opened it — before them
# the human maximum (0.335) sat above the lowest AI document (0.262).
#
# Two caveats that must travel with this number:
#   1. The AI side is 9 documents. The human side is credible, the AI side is
#      not yet — it needs a real corpus of varied prompts and models.
#   2. The two sides differ in genre as well as authorship (court reasoning vs
#      opinions, contracts and letters), so part of the separation may be
#      genre. A same-genre human profile would settle it.
#
# Caveat 1 still stands. Caveat 2 has since been measured: see
# FAMILY_THRESHOLDS below.
REVIEW_THRESHOLD = 0.25

# Where each family's human writing stops.
#
# The threshold is not a universal constant, because the score is not on a
# universal scale: it is a weighted mean exceedance above ONE profile's human
# range, so it means "how far past these particular humans" and moves with
# the profile. 0.25 was chosen by measuring where SAOS judgments stop
# (max 0.2309) and sitting just above it. Applying the same method to another
# family gives another number.
#
# Measured against `law_firm_contract`: 51 human contracts top out at 0.1137
# while AI-drafted contracts start at 0.2009. The global 0.25 sits above the
# whole AI range there, so a contract could never be flagged - the family
# would be calibrated and silent, which is worse than uncalibrated.
#
# Re-measured on 27 full-length documents from tools/build_ai_corpus.py
# (244-2778 words, one model, three prompt registers, three temperatures),
# against 300 held-out SAOS judgments and 51 human contracts:
#
#   filing_official   threshold  recall  FPR        contract  recall  FPR
#                     0.12       100%    6.0%                 0%      0.0%
#                     0.15       100%    3.0%   <-            0%      0.0%
#                     0.08        -      -                   78%      5.9%
#                     0.25        22%    0.0%                 0%      0.0%
#
# The old 0.25 catches 22% of real filings and no contracts at all. It was
# measured on documents of 89-160 words, where the score is inflated and
# unstable: truncating one document to 150 words doubled its score, and the
# same happens to human text. The human side was always full-length
# judgments, so the published operating point compared an inflated
# population against a stable one.
#
# Contracts remain weak and the number below says so: at 0.08 the two
# populations still overlap (AI 0.064-0.114, human max 0.114). Eleven of the
# fourteen signal families never fire even on AI-written contracts - see
# data/family_activity.json - so the score there rests on nominalisation
# density, enumeration and sentence shape alone.
#
# SUPERSEDED, numbers kept for the record: the table above was measured while
# the dash detector counted markdown "---" and "|---|" as em dashes. With that
# fixed and these thresholds unchanged, filings are caught 6 of 12 (FPR 2.4%
# on 592 held-out judgments) and contracts 0 of 9 (AI 0.016-0.061). Markdown
# is now reported by `humanize_pl.artifacts`, outside the score. The
# thresholds stay until they are re-measured on purpose.
#
# STILL PROVISIONAL: one model, 18 filings and 9 contracts. Re-measure when
# the corpus spans several generators.
FAMILY_THRESHOLDS: dict[str, float] = {
    "filing_official": 0.15,
    "contract": 0.08,
}


def threshold_for_family(family: str | None) -> float:
    """The review threshold that applies to `family`."""
    return FAMILY_THRESHOLDS.get(family or "", REVIEW_THRESHOLD)

# Half-width of the band around the threshold where the verdict is not
# reliable. Measured, not chosen: rebuilding the reference profile from
# independent samples of the same corpus moves a document's calibrated score
# by about 0.03, and that spread does NOT shrink with corpus size - it held at
# 10, 25, 50, 100 and 200 documents alike. The score is built from a dozen
# signals crossing their percentiles in steps, so it is grainy by construction.
#
# A document inside this band therefore gets a coin-flip, and reporting one
# side of that coin as a verdict would be inventing precision. It is reported
# as borderline instead.
UNCERTAIN_BAND = 0.035


@dataclass(frozen=True)
class CalibratedSignal:
    name: str
    observed: float
    human_p50: float
    human_p95: float
    direction: str  # "high" when AI-generated text sits above the human range
    exceedance: float  # 0.0 inside the human range, 1.0 at twice the p95 margin
    weight: float
    confounded: bool = False


@dataclass(frozen=True)
class Calibration:
    profile_name: str
    profile_genre: str
    profile_documents: int
    calibrated_score: float
    human_score_p50: float
    human_score_p95: float
    signals: list[CalibratedSignal] = field(default_factory=list)

    @property
    def above_human_range(self) -> bool:
        """True when the document warrants human review, not a verdict of AI."""
        return self.calibrated_score >= REVIEW_THRESHOLD

    @property
    def borderline(self) -> bool:
        """True when the score is too close to the threshold to call.

        Inside this band the answer changes with which documents happened to
        be in the reference corpus, so it is not an answer.
        """
        return abs(self.calibrated_score - REVIEW_THRESHOLD) <= UNCERTAIN_BAND

    @property
    def verdict(self) -> str:
        if self.borderline:
            return "borderline"
        return "above_human_range" if self.above_human_range else "within_human_range"


def profile_for_family(family: str) -> ReferenceProfile | None:
    """The profile calibrating `family`, or None when none is measured yet."""
    name = FAMILY_PROFILES.get(family)
    return load_profile(name) if name else None


@lru_cache(maxsize=8)
def load_profile(name: str = DEFAULT_PROFILE) -> ReferenceProfile | None:
    path = PROFILE_DIR / f"{name}.json"
    if not path.exists():
        return None
    return ReferenceProfile.load(path)


def calibrate(
    diagnosis: DocumentDiagnosis,
    text: str,
    *,
    profile: ReferenceProfile | None = None,
) -> Calibration | None:
    """Express a diagnosis relative to measured human writing.

    Returns None when no reference profile is available — an uncalibrated
    score is still reported, but it must not be presented as a verdict.
    """
    profile = profile or load_profile()
    if profile is None or not diagnosis.metrics:
        return None

    signals: list[CalibratedSignal] = []
    observed_rates = {row.family: row.per_1000_words for row in diagnosis.families}

    for family, distribution in profile.family_rates.items():
        observed = observed_rates.get(family, 0.0)
        signals.append(
            CalibratedSignal(
                name=f"family:{family}",
                observed=round(observed, 4),
                human_p50=distribution.p50,
                human_p95=distribution.p95,
                direction="high",
                exceedance=_exceedance_high(observed, distribution.p95),
                weight=1.0,
            )
        )

    # Burstiness: the clearest discriminator in the reference data — human
    # court reasoning varies sentence length far more than AI-drafted prose.
    cv = diagnosis.metrics.get("sentence_length_cv", 0.0)
    signals.append(
        CalibratedSignal(
            name="sentence_length_cv",
            observed=round(cv, 4),
            human_p50=profile.sentence_length_cv.p50,
            human_p95=profile.sentence_length_cv.p95,
            direction="low",
            exceedance=_exceedance_low(cv, profile.sentence_length_cv.p50),
            weight=1.5,
        )
    )

    # Shape signals are scored only once a reference profile carries a measured
    # baseline for them. An empty baseline must not enter the average: a "low"
    # signal would be dead weight (diluting the score) and a "high" one would
    # fall to the rate floor and fabricate a hit from nothing.
    # Burstiness is reported, never scored, for two independent reasons.
    #
    # It carries no information beyond `sentence_length_cv`. Dividing the
    # (sd - mean) / (sd + mean) formula through by the mean gives
    # (cv - 1) / (cv + 1), a strictly increasing function of cv - the same
    # quantity on a compressed scale. Scoring both would count one piece of
    # evidence twice, and cv already carries the heaviest weight there is.
    #
    # Worse, the compressed scale is negative across the whole human range
    # (cv 0.83 maps to -0.093), and `_exceedance_low` returns 0.0 for any
    # non-positive median. The signal could therefore never fire: it would
    # sit in the weighted average contributing a permanent zero, diluting
    # every document's score - exactly the failure the comment above warns
    # about for empty baselines.
    burstiness = diagnosis.metrics.get("sentence_burstiness", 0.0)
    if not profile.sentence_burstiness.is_empty:
        signals.append(
            CalibratedSignal(
                name="sentence_burstiness",
                observed=round(burstiness, 4),
                human_p50=profile.sentence_burstiness.p50,
                human_p95=profile.sentence_burstiness.p95,
                direction="low",
                exceedance=0.0,
                weight=0.0,
            )
        )

    entropy = diagnosis.metrics.get("sentence_entropy", 0.0)
    if not profile.sentence_entropy.is_empty:
        signals.append(
            CalibratedSignal(
                name="sentence_entropy",
                observed=round(entropy, 4),
                human_p50=profile.sentence_entropy.p50,
                human_p95=profile.sentence_entropy.p95,
                direction="low",
                exceedance=_exceedance_low(entropy, profile.sentence_entropy.p50),
                weight=1.0,
            )
        )

    # Paragraph shape: AI output clusters near a fixed paragraph size, human
    # legal writing mixes one-line findings with long argument blocks.
    shape_cv = diagnosis.metrics.get("paragraph_shape_cv", 0.0)
    if shape_cv > 0 and profile.paragraph_shape_cv.p50 > 0:
        signals.append(
            CalibratedSignal(
                name="paragraph_shape_cv",
                observed=round(shape_cv, 4),
                human_p50=profile.paragraph_shape_cv.p50,
                human_p95=profile.paragraph_shape_cv.p95,
                direction="low",
                exceedance=_exceedance_low(shape_cv, profile.paragraph_shape_cv.p50),
                weight=1.0,
            )
        )

    mean_words = diagnosis.metrics.get("mean_sentence_words", 0.0)
    signals.append(
        CalibratedSignal(
            name="mean_sentence_words",
            observed=round(mean_words, 4),
            human_p50=profile.sentence_words.p50,
            human_p95=profile.sentence_words.p95,
            direction="low",
            exceedance=_exceedance_low(mean_words, profile.sentence_words.p50),
            weight=0.5,
        )
    )

    connective = diagnosis.metrics.get("connective_density", 0.0)
    if not profile.connective_density.is_empty:
        signals.append(
            CalibratedSignal(
                name="connective_density",
                observed=round(connective, 4),
                human_p50=profile.connective_density.p50,
                human_p95=profile.connective_density.p95,
                direction="high",
                exceedance=_exceedance_high(connective, profile.connective_density.p95),
                weight=1.0,
            )
        )

    for name, observed, distribution in (
        ("opening_diversity", diagnosis.metrics.get("opening_diversity", 0.0),
         profile.opening_diversity),
        ("type_token_ratio", diagnosis.metrics.get("type_token_ratio", 0.0), profile.mtld),
    ):
        signals.append(
            CalibratedSignal(
                name=name,
                observed=round(observed, 4),
                human_p50=distribution.p50,
                human_p95=distribution.p95,
                direction="high",
                exceedance=0.0,
                weight=0.0,
                confounded=True,
            )
        )

    scored = [signal for signal in signals if not signal.confounded and signal.weight > 0]
    total_weight = sum(signal.weight for signal in scored)
    calibrated = (
        sum(signal.exceedance * signal.weight for signal in scored) / total_weight
        if total_weight
        else 0.0
    )

    return Calibration(
        profile_name=profile.name,
        profile_genre=profile.genre,
        profile_documents=profile.document_count,
        calibrated_score=round(min(1.0, calibrated), 4),
        human_score_p50=profile.signal_score.p50,
        human_score_p95=profile.signal_score.p95,
        signals=sorted(signals, key=lambda s: s.exceedance * s.weight, reverse=True),
    )


def _exceedance_high(observed: float, human_p95: float) -> float:
    """How far above the human 95th percentile, saturating at twice the margin."""
    threshold = max(human_p95, RATE_FLOOR_PER_1000)
    if observed <= threshold:
        return 0.0
    return round(min(1.0, (observed - threshold) / threshold), 4)


def _exceedance_low(observed: float, human_p50: float) -> float:
    """How far below the human median, saturating at half of it."""
    if human_p50 <= 0 or observed >= human_p50:
        return 0.0
    return round(min(1.0, (human_p50 - observed) / (human_p50 * 0.5)), 4)
