from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from humanize_pl.config import HumanizeConfig, Mode
from humanize_pl.results import CandidateRejection, CandidateTrace, SentenceChange, SentenceSkip
from humanize_pl.rules.base import Candidate
from humanize_pl.rules.engine import RuleEngine
from humanize_pl.rules.features import (
    ParagraphFeatures,
    SentenceFeatures,
    analyze_paragraph_features,
    analyze_sentence_features,
    enrich_features_with_analysis,
)
from humanize_pl.rules.redundancy import redundancy_candidates
from humanize_pl.rules.scoring import score_candidate
from humanize_pl.safety.agreement import agreement_gate
from humanize_pl.safety.protectors import ProtectedText
from humanize_pl.safety.syntax import stanza_finite_verb_gate
from humanize_pl.safety.validators import GateCheck, validate_candidate
from humanize_pl.sentence_splitter import split_sentences


# A masked-LM fluency scorer is structurally biased against AI-artifact removal:
# "fluent" and "high-probability" are the same thing to a language model, and
# AI-style discourse frames are high-probability Polish. HerBERT scores
# "Warto wskazać, że X" above bare "X", so the gate was rejecting exactly the
# edits the engine exists to make. Measured on the sample set: with the gate
# applied, hybrid left the AI signal at 0.52 where basic reduced it to 0.46.
#
# These operations stay subject to every other gate — syntax, agreement,
# normativity, anchor retention and semantic similarity. Only the fluency
# scorer is skipped, because dropping a discourse marker does not break Polish,
# it only makes it less predictable, which is the point.
FLUENCY_EXEMPT_OPERATIONS = frozenset(
    {"ai_artifact_reduction", "legal_ai_style_rewrite", "redundancy_reduction"}
)


@dataclass
class PipelineContext:
    config: HumanizeConfig
    protected: ProtectedText
    paragraph_index: int
    include_candidates: bool = False
    paragraph_features: ParagraphFeatures | None = None
    trace: list[CandidateTrace] = field(default_factory=list)
    analysis_cache: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> Mode:
        return self.config.mode

    @property
    def intensity(self) -> int:
        return self.config.intensity()


@dataclass
class StageResult:
    stage: str
    candidates: list[Candidate] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    gate_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParagraphPipelineResult:
    text: str
    changes: list[SentenceChange] = field(default_factory=list)
    rejected: list[CandidateRejection] = field(default_factory=list)
    skipped: list[SentenceSkip] = field(default_factory=list)
    traces: list[CandidateTrace] = field(default_factory=list)


@dataclass
class SentencePipelineResult:
    text: str
    changes: list[SentenceChange] = field(default_factory=list)
    rejected: list[CandidateRejection] = field(default_factory=list)
    traces: list[CandidateTrace] = field(default_factory=list)
    candidates_seen: bool = False


class LegalPipeline:
    def __init__(
        self,
        *,
        config: HumanizeConfig,
        protected: ProtectedText,
        rule_engine: RuleEngine,
        stanza_engine: Any = None,
        semantic: Any = None,
        fluency: Any = None,
        morfeusz: Any = None,
        include_candidates: bool = False,
    ) -> None:
        self.config = config
        self.protected = protected
        self.rule_engine = rule_engine
        self.stanza_engine = stanza_engine
        self.semantic = semantic
        self.fluency = fluency
        self.morfeusz = morfeusz
        self.include_candidates = include_candidates

    def process_paragraph(self, body: str, *, paragraph_index: int) -> ParagraphPipelineResult:
        sentences = split_sentences(body)
        paragraph_features = analyze_paragraph_features(sentences)
        context = PipelineContext(
            config=self.config,
            protected=self.protected,
            paragraph_index=paragraph_index,
            include_candidates=self.include_candidates,
            paragraph_features=paragraph_features,
        )

        output_sentences: list[str] = []
        changes: list[SentenceChange] = []
        rejected: list[CandidateRejection] = []
        skipped: list[SentenceSkip] = []

        for sentence_index, sentence in enumerate(sentences):
            sentence_result = self._process_sentence(
                sentence,
                context=context,
                sentence_index=sentence_index,
                previous_sentence=output_sentences[-1] if output_sentences else None,
                paragraph_sentences=sentences,
            )
            rejected.extend(sentence_result.rejected)
            context.trace.extend(sentence_result.traces)
            output_sentences.append(sentence_result.text)
            if sentence_result.changes:
                changes.extend(sentence_result.changes)
            else:
                skipped.append(
                    SentenceSkip(
                        original=self.protected.restore(sentence),
                        reason="no_candidate_accepted" if sentence_result.candidates_seen else "no_candidate",
                        paragraph_index=paragraph_index,
                        sentence_index=sentence_index,
                    )
                )

        return ParagraphPipelineResult(
            text=" ".join(s.strip() for s in output_sentences if s.strip()),
            changes=changes,
            rejected=rejected,
            skipped=skipped,
            traces=context.trace,
        )

    def _process_sentence(
        self,
        sentence: str,
        *,
        context: PipelineContext,
        sentence_index: int,
        previous_sentence: str | None,
        paragraph_sentences: list[str],
    ) -> SentencePipelineResult:
        current_text = sentence
        changes: list[SentenceChange] = []
        rejected: list[CandidateRejection] = []
        traces: list[CandidateTrace] = []
        applied_rules: set[str] = set()
        candidates_seen = False

        for step_index in range(self._step_limit()):
            features_before = analyze_sentence_features(current_text)
            analysis = self._stanza_analysis(current_text, context)
            features_before = enrich_features_with_analysis(features_before, analysis)
            candidates = self.rule_engine.generate_candidates(
                current_text,
                analysis=analysis,
                features=features_before,
                paragraph_features=context.paragraph_features,
                intensity=context.intensity,
            )
            candidates.extend(
                self._redundancy_candidates(
                    current_text,
                    previous_sentence=previous_sentence,
                    features=features_before,
                    paragraph_features=context.paragraph_features,
                )
            )
            candidates = [
                self._with_context_metadata(
                    candidate,
                    analysis=analysis,
                    paragraph_features=context.paragraph_features,
                )
                for candidate in candidates
            ]
            candidates = [candidate for candidate in candidates if candidate.rule not in applied_rules]
            candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
            candidates_seen = candidates_seen or bool(candidates)
            if not candidates:
                break

            stage = StageResult(
                stage="candidate_generation",
                candidates=candidates,
                metrics={
                    "intensity": context.intensity,
                    "step_index": step_index,
                    "features": asdict(features_before),
                    "paragraph_features": asdict(context.paragraph_features)
                    if context.paragraph_features
                    else None,
                    "paragraph_sentence_count": len(paragraph_sentences),
                },
            )
            change, next_text, local_rejected, local_traces = self._choose_candidate(
                current_text,
                stage=stage,
                features_before=features_before,
                paragraph_index=context.paragraph_index,
                sentence_index=sentence_index,
                step_index=step_index,
                include_candidates=context.include_candidates,
                context=context,
            )
            rejected.extend(local_rejected)
            traces.extend(local_traces)
            if not change.rule or next_text == current_text:
                break

            changes.append(change)
            applied_rules.add(change.rule)
            current_text = next_text

        return SentencePipelineResult(
            text=self.protected.restore(current_text),
            changes=changes,
            rejected=rejected,
            traces=traces,
            candidates_seen=candidates_seen,
        )

    def _step_limit(self) -> int:
        if self.config.mode == Mode.conservative:
            return 1
        if self.config.mode == Mode.standard:
            return 2
        return 3

    def _redundancy_candidates(
        self,
        sentence: str,
        *,
        previous_sentence: str | None,
        features: SentenceFeatures,
        paragraph_features: ParagraphFeatures | None,
    ) -> list[Candidate]:
        raw = redundancy_candidates(
            sentence,
            previous_sentence=previous_sentence,
            mode=self.config.mode,
            paragraph_features=paragraph_features,
        )
        return [
            score_candidate(
                sentence,
                candidate,
                features=features,
                mode=self.config.mode,
                intensity=self.config.intensity(),
                paragraph_features=paragraph_features,
            )
            for candidate in raw
        ]

    def _stanza_analysis(self, sentence: str, context: PipelineContext):
        if self.stanza_engine is None:
            return None
        if sentence not in context.analysis_cache:
            context.analysis_cache[sentence] = self.stanza_engine.analyze_sentence(sentence)
        return context.analysis_cache[sentence]

    def _with_context_metadata(
        self,
        candidate: Candidate,
        *,
        analysis,
        paragraph_features: ParagraphFeatures | None,
    ) -> Candidate:
        paragraph_snapshot = asdict(paragraph_features) if paragraph_features else None
        nlp_confidence = candidate.nlp_confidence
        if nlp_confidence is None and analysis is not None:
            nlp_confidence = self._nlp_confidence(analysis)
        return replace(
            candidate,
            nlp_confidence=nlp_confidence,
            paragraph_features_before=candidate.paragraph_features_before or paragraph_snapshot,
            paragraph_features_after=candidate.paragraph_features_after or paragraph_snapshot,
        )

    def _nlp_confidence(self, analysis) -> float | None:
        if analysis is None or not hasattr(analysis, "dependency_summary"):
            return None
        summary = analysis.dependency_summary()
        score = 0.0
        if summary.get("has_finite_verb"):
            score += 0.45
        if summary.get("has_subject"):
            score += 0.35
        if summary.get("has_object"):
            score += 0.20
        return round(score, 4)

    def _choose_candidate(
        self,
        original: str,
        *,
        stage: StageResult,
        features_before: SentenceFeatures,
        paragraph_index: int,
        sentence_index: int,
        step_index: int,
        include_candidates: bool,
        context: PipelineContext,
    ) -> tuple[SentenceChange, str, list[CandidateRejection], list[CandidateTrace]]:
        best = SentenceChange(
            original=self.protected.restore(original),
            rewritten=original,
            rule=None,
            accepted=True,
            reason="original",
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            step_index=step_index,
        )
        rejected: list[CandidateRejection] = []
        traces: list[CandidateTrace] = []

        for index, cand in enumerate(stage.candidates):
            validation = validate_candidate(
                original,
                cand.text,
                protected=self.protected,
                max_length_ratio=self.config.length_ratio(),
                rule=cand.rule,
                operation_type=cand.operation_type,
            )
            features_after = analyze_sentence_features(cand.text)
            if not validation.ok:
                if include_candidates:
                    traces.append(
                        self._candidate_trace(
                            original,
                            cand,
                            "rejected",
                            validation.reason,
                            features_before,
                            features_after,
                            validation.checks,
                            paragraph_index,
                            sentence_index,
                            step_index,
                        )
                    )
                rejected.append(
                    CandidateRejection(
                        self.protected.restore(original),
                        self.protected.restore(cand.text),
                        cand.rule,
                        validation.reason,
                        paragraph_index=paragraph_index,
                        sentence_index=sentence_index,
                    )
                )
                continue

            syntax_checks: list[GateCheck] = []
            if self.stanza_engine is not None:
                syntax_checks = stanza_finite_verb_gate(
                    original,
                    cand.text,
                    self.stanza_engine,
                    analysis_cache=context.analysis_cache,
                )
                failed_syntax = next((check for check in syntax_checks if not check.ok), None)
                if failed_syntax:
                    gate_checks = validation.checks + syntax_checks
                    if include_candidates:
                        traces.append(
                            self._candidate_trace(
                                original,
                                cand,
                                "rejected",
                                failed_syntax.reason,
                                features_before,
                                features_after,
                                gate_checks,
                                paragraph_index,
                                sentence_index,
                                step_index,
                            )
                        )
                    rejected.append(
                        CandidateRejection(
                            self.protected.restore(original),
                            self.protected.restore(cand.text),
                            cand.rule,
                            failed_syntax.reason,
                            paragraph_index=paragraph_index,
                            sentence_index=sentence_index,
                        )
                    )
                    continue

            agreement_checks: list[GateCheck] = []
            if self.config.agreement_gate_enabled and (
                self.stanza_engine is not None or self.morfeusz is not None
            ):
                restored_cand = self.protected.restore(cand.text)
                agreement_checks, effective_cand_text = agreement_gate(
                    self.protected.restore(original),
                    restored_cand,
                    stanza_engine=self.stanza_engine,
                    morfeusz=self.morfeusz,
                    analysis_cache=context.analysis_cache,
                )
                if effective_cand_text != restored_cand:
                    # NP agreement was auto-repaired — update cand to the repaired text.
                    cand = replace(cand, text=self.protected.re_protect(effective_cand_text))
                failed_agreement = next(
                    (check for check in agreement_checks if not check.ok), None
                )
                if failed_agreement:
                    gate_checks = validation.checks + syntax_checks + agreement_checks
                    if include_candidates:
                        traces.append(
                            self._candidate_trace(
                                original,
                                cand,
                                "rejected",
                                failed_agreement.reason,
                                features_before,
                                features_after,
                                gate_checks,
                                paragraph_index,
                                sentence_index,
                                step_index,
                            )
                        )
                    rejected.append(
                        CandidateRejection(
                            self.protected.restore(original),
                            self.protected.restore(cand.text),
                            cand.rule,
                            failed_agreement.reason,
                            paragraph_index=paragraph_index,
                            sentence_index=sentence_index,
                        )
                    )
                    continue

            sim = None
            semantic_checks: list[GateCheck] = []
            if self.semantic is not None and cand.text != original:
                sim = self.semantic.similarity(self.protected.restore(original), self.protected.restore(cand.text))
                if sim < self.config.similarity_threshold():
                    semantic_checks.append(
                        GateCheck(
                            "semantic_similarity",
                            False,
                            "semantic similarity below threshold",
                        )
                    )
                    gate_checks = validation.checks + syntax_checks + agreement_checks + semantic_checks
                    if include_candidates:
                        traces.append(
                            self._candidate_trace(
                                original,
                                cand,
                                "rejected",
                                "semantic similarity below threshold",
                                features_before,
                                features_after,
                                gate_checks,
                                paragraph_index,
                                sentence_index,
                                step_index,
                                semantic_similarity=sim,
                            )
                        )
                    rejected.append(
                        CandidateRejection(
                            self.protected.restore(original),
                            self.protected.restore(cand.text),
                            cand.rule,
                            "semantic similarity below threshold",
                            sim,
                            paragraph_index=paragraph_index,
                            sentence_index=sentence_index,
                        )
                    )
                    continue
                semantic_checks.append(GateCheck("semantic_similarity", True))
                cand = self._with_transformer_score(cand, semantic_similarity=sim)

            fluency_checks: list[GateCheck] = []
            if self.fluency is not None and cand.text != original:
                fluency_delta = self.fluency.delta(
                    self.protected.restore(original),
                    self.protected.restore(cand.text),
                )
                cand = self._with_transformer_score(cand, fluency_delta=fluency_delta)
                if (
                    fluency_delta < self.config.min_fluency_delta
                    and cand.operation_type not in FLUENCY_EXEMPT_OPERATIONS
                ):
                    fluency_checks.append(
                        GateCheck(
                            "fluency_delta",
                            False,
                            "fluency score degraded below threshold",
                        )
                    )
                    gate_checks = (
                        validation.checks
                        + syntax_checks
                        + agreement_checks
                        + semantic_checks
                        + fluency_checks
                    )
                    if include_candidates:
                        traces.append(
                            self._candidate_trace(
                                original,
                                cand,
                                "rejected",
                                "fluency score degraded below threshold",
                                features_before,
                                features_after,
                                gate_checks,
                                paragraph_index,
                                sentence_index,
                                step_index,
                                semantic_similarity=sim,
                            )
                        )
                    rejected.append(
                        CandidateRejection(
                            self.protected.restore(original),
                            self.protected.restore(cand.text),
                            cand.rule,
                            "fluency score degraded below threshold",
                            sim,
                            paragraph_index=paragraph_index,
                            sentence_index=sentence_index,
                        )
                    )
                    continue
                fluency_checks.append(GateCheck("fluency_delta", True))

            if include_candidates:
                traces.append(
                    self._candidate_trace(
                        original,
                        cand,
                        "accepted",
                        "accepted",
                        features_before,
                        features_after,
                        validation.checks
                        + syntax_checks
                        + agreement_checks
                        + semantic_checks
                        + fluency_checks,
                        paragraph_index,
                        sentence_index,
                        step_index,
                        semantic_similarity=sim,
                    )
                )
                for remaining in stage.candidates[index + 1 :]:
                    traces.append(
                        self._candidate_trace(
                            original,
                            remaining,
                            "not_evaluated",
                            "higher-scored candidate accepted first",
                            features_before,
                            analyze_sentence_features(remaining.text),
                            [],
                            paragraph_index,
                            sentence_index,
                            step_index,
                        )
                    )

            return SentenceChange(
                original=self.protected.restore(original),
                rewritten=self.protected.restore(cand.text),
                rule=cand.rule,
                accepted=True,
                reason="accepted",
                semantic_similarity=sim,
                paragraph_index=paragraph_index,
                sentence_index=sentence_index,
                step_index=step_index,
                stage=cand.stage,
                operation_type=cand.operation_type,
                risk=cand.risk,
                features_before=asdict(features_before),
                features_after=asdict(features_after),
                score_before_gate=cand.score_before_gate,
                score_after_gate=cand.score_after_gate,
                gate_results=[
                    asdict(check)
                    for check in (
                        validation.checks
                        + syntax_checks
                        + agreement_checks
                        + semantic_checks
                        + fluency_checks
                    )
                ],
                fluency_delta=cand.fluency_delta,
                nlp_confidence=cand.nlp_confidence,
                targeted_issue=cand.targeted_issue,
                paragraph_features_before=cand.paragraph_features_before,
                paragraph_features_after=cand.paragraph_features_after,
                score_breakdown=cand.score_breakdown,
            ), cand.text, rejected, traces

        return best, original, rejected, traces

    def _with_transformer_score(
        self,
        candidate: Candidate,
        *,
        semantic_similarity: float | None = None,
        fluency_delta: float | None = None,
    ) -> Candidate:
        score_after_gate = candidate.score_after_gate if candidate.score_after_gate is not None else candidate.score
        score_delta = 0.0
        if semantic_similarity is not None:
            margin = semantic_similarity - self.config.similarity_threshold()
            semantic_delta = max(-0.04, min(0.04, margin * 0.20))
            score_delta += semantic_delta
        if fluency_delta is not None:
            fluency_score_delta = max(-0.05, min(0.04, fluency_delta * 0.04))
            score_delta += fluency_score_delta
        adjusted_score = round(max(0.0, min(1.0, candidate.score + score_delta)), 4)
        adjusted_after_gate = round(max(0.0, min(1.0, score_after_gate + score_delta)), 4)
        features_delta = dict(candidate.features_delta or {})
        if semantic_similarity is not None:
            features_delta["semantic_score_delta"] = round(score_delta, 4)
        if fluency_delta is not None:
            features_delta["fluency_delta"] = round(fluency_delta, 4)
        score_breakdown = dict(candidate.score_breakdown or {})
        if semantic_similarity is not None:
            semantic_risk = max(0.0, self.config.similarity_threshold() - semantic_similarity)
            score_breakdown["semantic_risk"] = round(semantic_risk, 4)
        if fluency_delta is not None:
            score_breakdown["fluency_gain"] = round(fluency_score_delta, 4)
        score_breakdown["final_score"] = adjusted_after_gate
        return replace(
            candidate,
            score=adjusted_score,
            score_after_gate=adjusted_after_gate,
            fluency_delta=round(fluency_delta, 4) if fluency_delta is not None else candidate.fluency_delta,
            features_delta=features_delta,
            score_breakdown=score_breakdown,
        )

    def _candidate_trace(
        self,
        original: str,
        candidate: Candidate,
        status: str,
        reason: str,
        features_before: SentenceFeatures,
        features_after: SentenceFeatures,
        gate_results: list[GateCheck],
        paragraph_index: int,
        sentence_index: int,
        step_index: int,
        *,
        semantic_similarity: float | None = None,
    ) -> CandidateTrace:
        return CandidateTrace(
            original=self.protected.restore(original),
            candidate=self.protected.restore(candidate.text),
            rule=candidate.rule,
            score=candidate.score,
            status=status,
            reason=reason,
            semantic_similarity=semantic_similarity,
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            step_index=step_index,
            stage=candidate.stage,
            operation_type=candidate.operation_type,
            risk=candidate.risk,
            features_before=asdict(features_before),
            features_after=asdict(features_after),
            score_before_gate=candidate.score_before_gate,
            score_after_gate=candidate.score_after_gate,
            fluency_delta=candidate.fluency_delta,
            nlp_confidence=candidate.nlp_confidence,
            targeted_issue=candidate.targeted_issue,
            score_breakdown=candidate.score_breakdown,
            paragraph_features_before=candidate.paragraph_features_before,
            paragraph_features_after=candidate.paragraph_features_after,
            gate_results=[asdict(check) for check in gate_results],
        )
