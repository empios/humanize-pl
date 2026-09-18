"""The rhythm pass: measure, pick the best safe edit, repeat, stop early.

Greedy and one operation at a time, with a full re-measure after each. Not
beam search: every operation changes the mean that every other operation is
scored against, so a plan made two steps ahead is worth nothing by the time
it executes.

The loop stops as soon as the document is inside the human band. That is the
whole answer to "move toward human, do not maximise" - there is no reward for
going further, and `band_distance` starts charging for overshoot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from humanize_pl.blueprint import is_heading
from humanize_pl.config import Mode
from humanize_pl.detect.reference import ReferenceProfile
from humanize_pl.rhythm.objective import MIN_LOSS_GAIN, evaluate, measure_lengths
from humanize_pl.rhythm.operations import (
    RhythmOperation,
    is_enumeration,
    merge_candidates,
    paragraph_merge_candidates,
    paragraph_split_candidates,
    split_candidates,
    token_multiset_preserved,
    word_count,
)
from humanize_pl.rhythm.policy import RhythmDecision, RhythmScope, decide
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate
from humanize_pl.sentence_splitter import split_sentences


class RhythmInvariantError(RuntimeError):
    """The pass broke a promise it makes to the caller."""


@dataclass
class RhythmResult:
    text: str
    changes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Why the pass ran as it did - no profile, too few sentences, the
    # paragraph axis off. About the process, not the document, so it must
    # not reach the warnings that decide readiness: on DOCX the paragraph
    # note alone put 30 of 32 documents below "ready".
    notes: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    loss_before: float = 0.0
    loss_after: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def _paragraphs_of(text: str) -> tuple[list[str], list[int], list[list[str]]]:
    """Non-empty lines, their indices in the original, and their sentences."""
    lines = text.split("\n")
    indices = [index for index, line in enumerate(lines) if line.strip()]
    sentences = [split_sentences(lines[index]) for index in indices]
    return lines, indices, sentences


def _frozen_paragraphs(
    sentences: list[list[str]], protected_indices: set[int]
) -> frozenset[int]:
    """Paragraphs no operation may touch on either side."""
    frozen = set(protected_indices)
    for index, group in enumerate(sentences):
        text = " ".join(group)
        # A heading merged into its body would break the structure check that
        # blueprints run on the output; an enumeration item is not prose.
        if is_heading(text) or is_enumeration(text):
            frozen.add(index)
    return frozenset(frozen)


def apply_rhythm_pass(
    text: str,
    *,
    profile: ReferenceProfile | None,
    mode: Mode,
    scope: RhythmScope = RhythmScope.sentences_only,
    protected_paragraph_indices: set[int] | None = None,
) -> RhythmResult:
    """Move a document's rhythm toward the human band, or decline and say why."""
    lines, indices, paragraphs = _paragraphs_of(text)
    sentence_lengths = [word_count(s) for group in paragraphs for s in group]
    shape = [len(group) for group in paragraphs]

    decision: RhythmDecision = decide(
        mode=mode,
        scope=scope,
        profile=profile,
        sentence_count=len(sentence_lengths),
        paragraph_count=len(paragraphs),
    )
    objective = evaluate(measure_lengths(sentence_lengths, shape), profile)
    result = RhythmResult(text=text, loss_before=objective.loss, loss_after=objective.loss)

    if not decision.run:
        result.skipped_reason = decision.reason
        if decision.reason:
            result.notes.append(f"Rytm: {decision.reason}")
        return result
    if decision.reason:
        result.notes.append(f"Rytm: {decision.reason}")
    if objective.inside_band:
        result.skipped_reason = "Rytm dokumentu mieści się w ludzkim zakresie."
        return result

    frozen = _frozen_paragraphs(paragraphs, protected_paragraph_indices or set())
    # Sentences created by a split in this run, so the merge side cannot
    # immediately undo them and oscillate.
    created: dict[int, set[int]] = {}
    semicolons = 0
    # `max(1, ...)`, not `int(...)`: a rate per 1000 words truncates to zero
    # below ~700 words, and most documents here are shorter than that. The
    # cap exists to stop the semicolon becoming its own AI tell - it is rare
    # in Polish legal prose outside enumerations - not to forbid the device.
    semicolon_budget = max(
        1,
        round(decision.limits.max_semicolons_per_1000 * sum(sentence_lengths) / 1000),
    )

    for _step in range(decision.limits.max_steps):
        current = evaluate(measure_lengths(sentence_lengths, shape), profile)
        if current.inside_band:
            break

        candidates = _collect(
            paragraphs,
            mode=mode,
            limits=decision.limits,
            lengths=sentence_lengths,
            shape=shape,
            frozen=frozen,
            created=created,
            allow_paragraph_axis=decision.allow_paragraph_axis,
        )
        scored = []
        for operation in candidates:
            preview = evaluate(
                measure_lengths(operation.preview_lengths, operation.preview_shape), profile
            )
            # Ranked on the unsaturated distance: the saturated one is
            # flat at 1.0 for a document far outside the band, so every
            # candidate would tie at zero gain exactly when the layer has
            # the most to do.
            gain = round(current.search_loss - preview.search_loss, 4)
            if gain >= MIN_LOSS_GAIN:
                scored.append((gain, preview, operation))
        if not scored:
            break
        scored.sort(key=lambda row: -row[0])

        applied = None
        for gain, preview, operation in scored[:8]:
            uses_semicolon = (
                operation.kind == "sentence_merge" and "; " in operation.candidate
            )
            if uses_semicolon and semicolons >= semicolon_budget:
                continue
            if not _accepts(operation):
                continue
            applied = (gain, preview, operation)
            break
        if applied is None:
            break

        gain, preview, operation = applied
        if "; " in operation.candidate and operation.kind == "sentence_merge":
            semicolons += 1
        paragraphs, sentence_lengths, shape = _apply(operation, paragraphs)
        if operation.kind == "sentence_split":
            created.setdefault(operation.paragraph_index, set()).add(operation.sentence_index)
        result.changes.append(
            {
                "before": operation.original,
                "after": operation.candidate,
                "issue": "sentence_length_cv"
                if operation.kind.startswith("sentence")
                else "paragraph_shape_cv",
                "rule": operation.rule,
                "operation_type": operation.kind,
                "paragraph_index": operation.paragraph_index,
                "sentence_index": operation.sentence_index,
                "rhythm_loss_before": current.loss,
                "rhythm_loss_after": preview.loss,
            }
        )
        result.loss_after = preview.loss

    if not result.changes:
        return result

    rebuilt = _rebuild(lines, indices, paragraphs)
    if scope == RhythmScope.sentences_only:
        before = len([line for line in text.split("\n") if line.strip()])
        after = len([line for line in rebuilt.split("\n") if line.strip()])
        if before != after:
            # Loud rather than silent: in DOCX a paragraph-count change makes
            # the flow throw away every edit in the document.
            raise RhythmInvariantError(
                f"sentences_only zmieniło liczbę akapitów: {before} -> {after}"
            )
    result.text = rebuilt
    return result


def _collect(
    paragraphs: list[list[str]],
    *,
    mode: Mode,
    limits: Any,
    lengths: list[int],
    shape: list[int],
    frozen: frozenset[int],
    created: dict[int, set[int]],
    allow_paragraph_axis: bool,
) -> list[RhythmOperation]:
    out: list[RhythmOperation] = []
    offset = 0
    for index, sentences in enumerate(paragraphs):
        if index not in frozen:
            out.extend(
                merge_candidates(
                    sentences,
                    paragraph_index=index,
                    limits=limits,
                    lengths=lengths,
                    shape=shape,
                    offset=offset,
                    excluded=frozenset(created.get(index, set())),
                )
            )
            out.extend(
                split_candidates(
                    sentences,
                    paragraph_index=index,
                    mode=mode,
                    limits=limits,
                    lengths=lengths,
                    shape=shape,
                    offset=offset,
                )
            )
        offset += len(sentences)

    if allow_paragraph_axis:
        out.extend(
            paragraph_merge_candidates(
                paragraphs, lengths=lengths, shape=shape, frozen=frozen
            )
        )
        if limits.allow_paragraph_split:
            out.extend(
                paragraph_split_candidates(
                    paragraphs, lengths=lengths, shape=shape, frozen=frozen
                )
            )
    return out


def _accepts(operation: RhythmOperation) -> bool:
    """Every gate the engine already owns, plus the one it cannot express."""
    if not token_multiset_preserved(operation.original, operation.candidate):
        return False
    if operation.kind.startswith("paragraph"):
        # Nothing but a boundary moved, so there is no text to validate. The
        # invariant is that the sentences survive unchanged.
        return _same_sentences(operation.original, operation.candidate)

    protected = protect_text(operation.original)
    validation = validate_candidate(
        protected.text,
        operation.candidate,
        protected=protected,
        max_length_ratio=1.60,
        rule=operation.rule,
        operation_type=operation.kind,
    )
    return validation.ok


def _same_sentences(original: str, candidate: str) -> bool:
    left = [s.strip() for s in split_sentences(original.replace("\n", " ")) if s.strip()]
    right = [s.strip() for s in split_sentences(candidate.replace("\n", " ")) if s.strip()]
    return left == right


def _apply(
    operation: RhythmOperation, paragraphs: list[list[str]]
) -> tuple[list[list[str]], list[int], list[int]]:
    paragraphs = [list(group) for group in paragraphs]
    index, position = operation.paragraph_index, operation.sentence_index

    if operation.kind == "sentence_merge":
        paragraphs[index][position : position + 2] = [operation.candidate]
    elif operation.kind == "sentence_split":
        parts = [part.strip() for part in operation.candidate.split(". ") if part.strip()]
        parts = [part if part.endswith((".", "!", "?")) else part + "." for part in parts]
        paragraphs[index][position : position + 1] = parts
    elif operation.kind == "paragraph_merge":
        paragraphs[index : index + 2] = [paragraphs[index] + paragraphs[index + 1]]
    elif operation.kind == "paragraph_split":
        paragraphs[index : index + 1] = [
            paragraphs[index][:position],
            paragraphs[index][position:],
        ]

    lengths = [word_count(s) for group in paragraphs for s in group]
    shape = [len(group) for group in paragraphs]
    return paragraphs, lengths, shape


def _rebuild(lines: list[str], indices: list[int], paragraphs: list[list[str]]) -> str:
    """Put the paragraphs back where they came from, keeping blank lines.

    When the paragraph count is unchanged each group returns to its original
    line, so surrounding whitespace survives untouched. A changed count can
    only happen in `full` scope, where the layout is rebuilt instead.
    """
    if len(paragraphs) == len(indices):
        out = list(lines)
        for group, line_index in zip(paragraphs, indices):
            out[line_index] = " ".join(group)
        return "\n".join(out)
    return "\n".join(" ".join(group) for group in paragraphs)
