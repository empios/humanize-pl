from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .reference import ReferenceProfile, windowed_ttr
from .base import DocumentDiagnosis

PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference_profiles"
DEFAULT_PROFILE = "saos_common_2018_2024"

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
#   0.15       100%        3.7%
#   0.25       100%        0.5%     <- REVIEW_THRESHOLD
#   0.30        78%        0.3%
#
# Two caveats that must travel with this number:
#   1. The AI side is 9 documents. The human side is credible, the AI side is
#      not yet — it needs a real corpus of varied prompts and models.
#   2. The two sides differ in genre as well as authorship (court reasoning vs
#      opinions, contracts and letters), so part of the separation may be
#      genre. A same-genre human profile would settle it.
REVIEW_THRESHOLD = 0.25


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

    for name, observed, distribution in (
        ("opening_diversity", diagnosis.metrics.get("opening_diversity", 0.0),
         profile.opening_diversity),
        ("type_token_ratio", windowed_ttr(text), profile.windowed_ttr),
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
