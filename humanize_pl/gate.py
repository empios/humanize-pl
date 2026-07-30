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


@dataclass(frozen=True)
class GateVerdict:
    needs_revision: bool
    score: float
    threshold: float
    diagnosis: DocumentDiagnosis
    violations: list[GateViolation] = field(default_factory=list)
    prompt_constraints: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "needs_revision": self.needs_revision,
            "score": self.score,
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
) -> GateVerdict:
    """Judge an AI-drafted answer and say what to change, without changing it."""
    diagnosis = detect_document(text)
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

    return GateVerdict(
        needs_revision=score >= threshold or missing_anchor,
        score=score,
        threshold=threshold,
        diagnosis=diagnosis,
        violations=violations,
        prompt_constraints=constraints,
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
