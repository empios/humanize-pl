from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calibration import Calibration


# Every signal family the detector can emit, in one place.
#
# It exists because the alternative failed silently. A reference profile only
# records the families its corpus happened to contain, the quality gate only
# constrains the families someone remembered to write a constraint for, and
# nothing compared the two lists. `typography_artifact` fell through both:
# fired by the detector, absent from the shipped profile, absent from the
# gate — so the em dash, one of the most recognisable AI tells in Polish, was
# measured and then dropped on the floor twice.
#
# A family added below and nowhere else fails the tests in
# tests/test_calibration.py, which is the point: the list is the contract.
AI_FAMILIES: tuple[str, ...] = (
    "abstract_frame",
    "antithesis",
    "balanced_pair",
    "concessive_reversal",
    "discourse_frame",
    "empty_emphasis",
    "nominalization",
    "practical_implication",
    "repeated_opening",
    "summary_frame",
    "transition_marker",
    "tricolon",
    "typography_artifact",
    "vague_reference",
)


@dataclass(frozen=True)
class Finding:
    """A single located AI-style signal in the source text.

    A finding is an observation, never a proposed edit. `rewritable` records
    whether the engine also owns a rewrite rule for this pattern — that is the
    difference between "we found nothing" and "we found something we will not
    touch", which the pipeline previously collapsed into silence.
    """

    family: str
    rule: str
    evidence: str
    paragraph_index: int
    sentence_index: int
    char_start: int
    char_end: int
    weight: float
    rewritable: bool = False
    detail: str | None = None


@dataclass(frozen=True)
class FamilySummary:
    family: str
    count: int
    per_1000_words: float
    weight_total: float
    rewritable_count: int


@dataclass(frozen=True)
class ParagraphDiagnosis:
    paragraph_index: int
    word_count: int
    sentence_count: int
    signal_score: float
    finding_count: int


@dataclass(frozen=True)
class DocumentDiagnosis:
    """Document-level AI-style diagnosis, independent of any rewriting.

    `ai_signal_score` is a saturating weighted density, not a probability. It
    is uncalibrated until a human Polish legal reference corpus exists; use it
    to rank documents against each other, not as a verdict.
    """

    ai_signal_score: float
    word_count: int
    sentence_count: int
    paragraph_count: int
    findings: list[Finding] = field(default_factory=list)
    families: list[FamilySummary] = field(default_factory=list)
    paragraphs: list[ParagraphDiagnosis] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    # None when no reference profile is installed for the genre.
    calibration: "Calibration | None" = None

    @property
    def rewritable_count(self) -> int:
        return sum(1 for finding in self.findings if finding.rewritable)

    @property
    def detected_only_count(self) -> int:
        """Signals the engine can see but has no rewrite rule for."""
        return sum(1 for finding in self.findings if not finding.rewritable)
