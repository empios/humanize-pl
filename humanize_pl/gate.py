"""Quality gate for AI-drafted, client-facing legal responses.

The problem this solves is register, not detector evasion: an "AI lawyer"
answer that reads as machine-written is a worse product, and Polish clients
read that register as evasive and generic.

The engine stays deterministic and never paraphrases. It returns a verdict and
a set of constraints; regenerating the answer is the caller's job, with its own
model. That division is deliberate — word-level paraphrasing of legal text
trades away the precision that is the whole product, and the failure mode is
silent.

    verdict = review_response(answer)
    if verdict.needs_revision:
        answer = my_llm.regenerate(question, constraints=verdict.prompt_constraints)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from humanize_pl.detect import DocumentDiagnosis, detect_document
from humanize_pl.detect.calibration import REVIEW_THRESHOLD

# Constraints are phrased as instructions to a Polish-language model, because
# that is where they are consumed.
FAMILY_CONSTRAINTS: dict[str, str] = {
    "discourse_frame": (
        "Nie otwieraj zdań ramami typu „Warto wskazać, że”, „Należy zauważyć, że”. "
        "Przejdź od razu do treści."
    ),
    "abstract_frame": (
        "Nie używaj zwrotów typu „ma istotne znaczenie”, „kluczowe znaczenie ma”, "
        "„odgrywa istotną rolę”. Napisz konkretnie, co z czego wynika."
    ),
    "balanced_pair": (
        "Nie buduj wywodu na parach „z jednej strony… z drugiej strony”. "
        "Podaj stanowisko wprost, a zastrzeżenia osobnym zdaniem."
    ),
    "antithesis": "Unikaj konstrukcji „nie chodzi o X, lecz o Y”.",
    "concessive_reversal": (
        "Unikaj zwrotu „Nie oznacza to (jednak), że…”. Jeżeli jest wyjątek, nazwij go wprost."
    ),
    "practical_implication": (
        "Unikaj zwrotu „W praktyce oznacza to…”. Napisz, co klient ma zrobić i do kiedy."
    ),
    "summary_frame": (
        "Nie kończ akapitem podsumowującym („Podsumowując”, „Reasumując”). "
        "Wniosek postaw na początku odpowiedzi."
    ),
    "tricolon": (
        "Nie zestawiaj trzech wyliczeń o zbliżonej długości. "
        "Użyj dwóch pozycji albo rozbij je na osobne zdania."
    ),
    "empty_emphasis": "Usuń wzmocnienia bez treści („to właśnie”, „w znacznym stopniu”).",
    "transition_marker": (
        "Ogranicz łączniki typu „ponadto”, „co więcej”, „dodatkowo” do najwyżej jednego."
    ),
    "vague_reference": (
        "Zastąp odniesienia typu „powyższy”, „przedmiotowy”, „niniejszy” nazwą rzeczy."
    ),
    "nominalization": (
        "Zamień rzeczowniki odczasownikowe na czasowniki osobowe "
        "(„dokonanie zapłaty” → „zapłacić”)."
    ),
    "repeated_opening": "Nie powtarzaj tego samego otwarcia zdania w odpowiedzi.",
    "typography_artifact": (
        "Nie używaj długiego myślnika jako wtrącenia. Wtrącenie zapisz w nawiasie "
        "albo osobnym zdaniem."
    ),
}

SHAPE_CONSTRAINTS: dict[str, str] = {
    "sentence_length_cv": (
        "Zróżnicuj długość zdań: obok zdań rozbudowanych postaw kilka krótkich, "
        "jedno- lub dwuczłonowych."
    ),
    "paragraph_shape_cv": (
        "Zróżnicuj długość akapitów. Nie utrzymuj stałego rozmiaru 3–5 zdań."
    ),
    "mean_sentence_words": "Nie skracaj wszystkich zdań do jednego wzorca długości.",
}

# An answer with no concrete anchors reads as generic regardless of its
# phrasing. This is a quality floor, not an AI signal.
ANCHOR_CONSTRAINT = (
    "Odpowiedź nie zawiera żadnej konkretnej kotwicy. Wskaż przepis, kwotę, termin "
    "albo nazwę strony."
)


@dataclass(frozen=True)
class GateViolation:
    family: str
    count: int
    evidence: str
    constraint: str


# Below this the calibrated score stops being a measurement.
#
# Every family signal is a rate per 1000 words and every shape signal is a
# distribution over sentences. In a 55-word answer one em dash reads as 18.2
# per 1000 - enough to saturate any human baseline on its own - and seven
# sentences cannot produce a human spread of lengths no matter who wrote
# them. The reference corpus is court judgments thousands of words long, so
# a short answer is not a small sample of that population; it is a different
# kind of object.
#
# 150 words is the threshold `humanize_pl.corpus.normalize` already uses to
# decide a document is worth measuring at all. Reusing it keeps one answer to
# one question.
MIN_SCORABLE_WORDS = 150

# What makes a short answer read as machine-written is reaching for several
# different frames at once, not one isolated tic.
#
# Measured on the two gate fixtures, both ~55 words: the machine-drafted one
# fires six distinct families (discourse frame, abstract frame, balanced
# pair, concessive reversal, practical implication, summary frame), the
# human-drafted one fires two, and both of those are hygiene rather than
# register - an em dash and a nominalisation. Three separates them with room
# on either side.
SHORT_TEXT_FAMILY_THRESHOLD = 3


@dataclass(frozen=True)
class GateVerdict:
    needs_revision: bool
    score: float
    threshold: float
    diagnosis: DocumentDiagnosis
    violations: list[GateViolation] = field(default_factory=list)
    prompt_constraints: list[str] = field(default_factory=list)
    # False when the text is too short for the score to mean anything. The
    # score is still reported - hiding it would invite someone to compute
    # their own - but it did not decide the verdict.
    score_is_meaningful: bool = True

    def to_json(self) -> dict:
        return {
            "needs_revision": self.needs_revision,
            "score": self.score,
            "score_is_meaningful": self.score_is_meaningful,
            "threshold": self.threshold,
            "violations": [
                {
                    "family": violation.family,
                    "count": violation.count,
                    "evidence": violation.evidence,
                    "constraint": violation.constraint,
                }
                for violation in self.violations
            ],
            "prompt_constraints": self.prompt_constraints,
        }


def review_response(
    text: str,
    *,
    threshold: float = REVIEW_THRESHOLD,
    require_anchor: bool = True,
    calibrate_against_default: bool = True,
    profile=None,
) -> GateVerdict:
    """Judge an AI-drafted answer and say what to change, without changing it.

    `profile` calibrates the score before it meets the threshold. Without one
    the raw score is compared instead, and that score saturates - which made
    the gate demand review of very nearly everything.
    """
    diagnosis = detect_document(
        text, profile=profile, calibrate_against_default=calibrate_against_default
    )
    calibration = diagnosis.calibration
    score = calibration.calibrated_score if calibration else diagnosis.ai_signal_score

    violations: list[GateViolation] = []
    for row in diagnosis.families:
        constraint = FAMILY_CONSTRAINTS.get(row.family)
        if not constraint:
            continue
        evidence = next(
            (finding.evidence for finding in diagnosis.findings if finding.family == row.family),
            "",
        )
        violations.append(
            GateViolation(
                family=row.family,
                count=row.count,
                evidence=evidence,
                constraint=constraint,
            )
        )

    constraints = [violation.constraint for violation in violations]

    if calibration is not None:
        for signal in calibration.signals:
            if signal.confounded or signal.exceedance <= 0:
                continue
            shape_constraint = SHAPE_CONSTRAINTS.get(signal.name)
            if shape_constraint and shape_constraint not in constraints:
                constraints.append(shape_constraint)

    missing_anchor = require_anchor and not _has_concrete_anchor(text)
    if missing_anchor:
        constraints.append(ANCHOR_CONSTRAINT)

    # A short answer is judged on how many different machine frames it reaches
    # for, because its rates are unreadable (see MIN_SCORABLE_WORDS). The
    # findings themselves stay in the verdict either way: an em dash is worth
    # reporting even when it is not worth a rewrite.
    score_is_meaningful = diagnosis.word_count >= MIN_SCORABLE_WORDS
    if score_is_meaningful:
        reads_as_machine = score >= threshold
    else:
        reads_as_machine = len(violations) >= SHORT_TEXT_FAMILY_THRESHOLD

    return GateVerdict(
        needs_revision=reads_as_machine or missing_anchor,
        score=score,
        threshold=threshold,
        diagnosis=diagnosis,
        violations=violations,
        prompt_constraints=constraints,
        score_is_meaningful=score_is_meaningful,
    )


def _has_concrete_anchor(text: str) -> bool:
    """A legal reference, an amount, a date or a deadline."""
    import regex as re

    patterns = (
        r"\bart\.\s*\d|§\s*\d|\bust\.\s*\d|\bpkt\s*\d",
        r"\d+(?:[ .]\d{3})*(?:[,.]\d+)?\s*(?:zł|PLN|EUR|USD)",
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}\s*r\.",
        r"\bw\s+terminie\s+\d+|\bdo\s+dnia\s+\d",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
