from __future__ import annotations

from dataclasses import dataclass, field
import regex as re
from typing import Any

from .config import Engine, HumanizeConfig, LegalReviewProfile, Mode
from .detect import detect_document
from .nlp.morfeusz import MorfeuszAnalyzer
from .nlp.semantic import (
    DEFAULT_FLUENCY_MODEL,
    DEFAULT_SEMANTIC_MODEL,
    EmbeddingSimilarityValidator,
    MaskedLMFluencyScorer,
)
from .nlp.stanza_engine import StanzaEngine
from .pipeline import LegalPipeline
from .results import (
    CandidateRejection,
    CandidateTrace,
    HumanizeResult,
    SentenceChange,
    SentenceSkip,
)
from .rules.engine import RuleEngine
from .safety.protectors import protect_text


def _coerce_mode(value: str | Mode) -> Mode:
    return value if isinstance(value, Mode) else Mode(value)


def _coerce_engine(value: str | Engine) -> Engine:
    return value if isinstance(value, Engine) else Engine(value)


def _coerce_legal_review_profile(value: str | LegalReviewProfile) -> LegalReviewProfile:
    return value if isinstance(value, LegalReviewProfile) else LegalReviewProfile(value)


@dataclass
class HumanizerSession:
    config: HumanizeConfig
    stanza_engine: Any = None
    semantic: Any = None
    fluency: Any = None
    morfeusz: Any = None
    engine_used: str = "basic"
    model_status: dict[str, str] = field(default_factory=dict)
    semantic_model_used: str | None = None
    fluency_model_used: str | None = None
    warnings: list[str] = field(default_factory=list)
    rule_engine: RuleEngine | None = None

    def __post_init__(self) -> None:
        if self.rule_engine is None:
            self.rule_engine = RuleEngine(mode=self.config.mode)

    def humanize(self, text: str, *, include_candidates: bool = False) -> HumanizeResult:
        # Detection is deliberately outside the rewrite pipeline: it must report
        # signals even in conservative mode, where no candidate is generated.
        diagnosis = detect_document(text)
        protected = protect_text(text)
        pipeline = LegalPipeline(
            config=self.config,
            protected=protected,
            rule_engine=self.rule_engine or RuleEngine(mode=self.config.mode),
            stanza_engine=self.stanza_engine,
            semantic=self.semantic,
            fluency=self.fluency,
            morfeusz=self.morfeusz,
            include_candidates=include_candidates,
        )

        output_parts: list[str] = []
        changes: list[SentenceChange] = []
        rejected: list[CandidateRejection] = []
        skipped: list[SentenceSkip] = []
        all_candidates: list[CandidateTrace] = []
        paragraph_index = 0

        for part in re.split(r"(\n+)", protected.text):
            if not part:
                continue
            if part.startswith("\n"):
                output_parts.append(part)
                continue

            whitespace = re.match(
                r"^(?P<prefix>\s*)(?P<body>.*?)(?P<suffix>\s*)$",
                part,
                re.DOTALL,
            )
            prefix = whitespace.group("prefix") if whitespace else ""
            body = whitespace.group("body") if whitespace else part
            suffix = whitespace.group("suffix") if whitespace else ""
            if not body:
                output_parts.append(part)
                continue

            result = pipeline.process_paragraph(body, paragraph_index=paragraph_index)
            output_parts.append(prefix + result.text + suffix)
            changes.extend(result.changes)
            rejected.extend(result.rejected)
            skipped.extend(result.skipped)
            all_candidates.extend(result.traces)
            paragraph_index += 1

        output_protected = "".join(output_parts)
        output = protected.restore(output_protected)
        return HumanizeResult(
            text=output,
            changed=output != text,
            changes=changes,
            rejected=rejected,
            skipped=skipped,
            all_candidates=all_candidates,
            engine_used=self.engine_used,
            legal_review_profile=self.config.legal_review_profile.value,
            model_status=dict(self.model_status),
            semantic_model=(
                self.semantic_model_used if self.config.engine == Engine.hybrid else None
            ),
            fluency_model=self.fluency_model_used if self.config.engine == Engine.hybrid else None,
            warnings=list(self.warnings),
            diagnosis=diagnosis,
        )


def create_humanizer_session(
    *,
    mode: str | Mode = Mode.conservative,
    engine: str | Engine = Engine.basic,
    legal_review_profile: str | LegalReviewProfile = LegalReviewProfile.legal_ai_review,
    semantic_threshold: float | None = None,
    semantic_model: str | None = None,
    fluency_model: str | None = None,
    require_models: bool = False,
    offline_models: bool = False,
    agreement_gate_enabled: bool = True,
    require_morfeusz: bool = False,
) -> HumanizerSession:
    mode_v = _coerce_mode(mode)
    engine_v = _coerce_engine(engine)
    profile_v = _coerce_legal_review_profile(legal_review_profile)
    config = HumanizeConfig(
        mode=mode_v,
        engine=engine_v,
        legal_review_profile=profile_v,
        semantic_threshold=semantic_threshold,
        semantic_model=semantic_model,
        fluency_model=fluency_model,
        require_models=require_models,
        offline_models=offline_models,
        agreement_gate_enabled=agreement_gate_enabled,
        require_morfeusz=require_morfeusz,
    )

    warnings: list[str] = []
    model_status: dict[str, str] = {
        "stanza": "not_requested",
        "semantic": "not_requested",
        "fluency": "not_requested",
        "morfeusz": "not_requested",
    }

    stanza_engine = None
    engine_used = engine_v.value
    if engine_v in {Engine.nlp, Engine.hybrid}:
        model_status["stanza"] = "requested"
        try:
            stanza_engine = StanzaEngine(offline=offline_models)
            model_status["stanza"] = "ready"
        except Exception as exc:
            model_status["stanza"] = f"unavailable: {type(exc).__name__}"
            if require_models:
                raise RuntimeError(
                    f"Required Stanza model unavailable: {type(exc).__name__}: {exc}"
                ) from exc
            warnings.append(f"Stanza unavailable, fallback allowed: {type(exc).__name__}: {exc}")
            stanza_engine = None
            engine_used = "basic"

    semantic = None
    fluency = None
    semantic_model_used = semantic_model or DEFAULT_SEMANTIC_MODEL
    fluency_model_used = fluency_model or DEFAULT_FLUENCY_MODEL
    if engine_v == Engine.hybrid:
        model_status["semantic"] = "requested"
        try:
            semantic = EmbeddingSimilarityValidator(
                threshold=config.similarity_threshold(),
                model_name=semantic_model,
                offline=offline_models,
            )
            semantic_model_used = semantic.model_name
            model_status["semantic"] = "ready"
        except Exception as exc:
            model_status["semantic"] = f"unavailable: {type(exc).__name__}"
            if require_models:
                raise RuntimeError(
                    f"Required semantic model unavailable: {type(exc).__name__}: {exc}"
                ) from exc
            warnings.append(
                f"Semantic validator unavailable, continuing without it: "
                f"{type(exc).__name__}: {exc}"
            )
            semantic = None
        model_status["fluency"] = "requested"
        try:
            fluency = MaskedLMFluencyScorer(model_name=fluency_model, offline=offline_models)
            fluency_model_used = fluency.model_name
            model_status["fluency"] = "ready"
        except Exception as exc:
            model_status["fluency"] = f"unavailable: {type(exc).__name__}"
            if require_models:
                raise RuntimeError(
                    f"Required fluency model unavailable: {type(exc).__name__}: {exc}"
                ) from exc
            warnings.append(
                f"Fluency scorer unavailable, continuing without it: "
                f"{type(exc).__name__}: {exc}"
            )
            fluency = None

    if engine_v == Engine.hybrid:
        engine_used = "hybrid" if any([stanza_engine, semantic, fluency]) else "basic"
    elif engine_v == Engine.nlp:
        engine_used = "nlp" if stanza_engine is not None else "basic"

    morfeusz = None
    if agreement_gate_enabled:
        model_status["morfeusz"] = "requested"
        try:
            morfeusz = MorfeuszAnalyzer()
            model_status["morfeusz"] = "ready"
        except Exception as exc:
            model_status["morfeusz"] = f"unavailable: {type(exc).__name__}"
            if require_morfeusz:
                raise RuntimeError(
                    f"Required Morfeusz analyzer unavailable: {type(exc).__name__}: {exc}"
                ) from exc
            warnings.append(
                f"Morfeusz unavailable, agreement gate continues without lexical layer: "
                f"{type(exc).__name__}: {exc}"
            )
            morfeusz = None

    return HumanizerSession(
        config=config,
        stanza_engine=stanza_engine,
        semantic=semantic,
        fluency=fluency,
        morfeusz=morfeusz,
        engine_used=engine_used,
        model_status=model_status,
        semantic_model_used=semantic_model_used,
        fluency_model_used=fluency_model_used,
        warnings=warnings,
    )


def humanize_text(
    text: str,
    *,
    mode: str | Mode = Mode.conservative,
    engine: str | Engine = Engine.basic,
    legal_review_profile: str | LegalReviewProfile = LegalReviewProfile.legal_ai_review,
    semantic_threshold: float | None = None,
    semantic_model: str | None = None,
    fluency_model: str | None = None,
    require_models: bool = False,
    offline_models: bool = False,
    include_candidates: bool = False,
    agreement_gate_enabled: bool = True,
    require_morfeusz: bool = False,
) -> HumanizeResult:
    session = create_humanizer_session(
        mode=mode,
        engine=engine,
        legal_review_profile=legal_review_profile,
        semantic_threshold=semantic_threshold,
        semantic_model=semantic_model,
        fluency_model=fluency_model,
        require_models=require_models,
        offline_models=offline_models,
        agreement_gate_enabled=agreement_gate_enabled,
        require_morfeusz=require_morfeusz,
    )
    return session.humanize(text, include_candidates=include_candidates)
