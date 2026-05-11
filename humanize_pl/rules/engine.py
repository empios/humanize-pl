from __future__ import annotations

from dataclasses import dataclass

from humanize_pl.config import Mode
from .ai_artifacts import ai_artifact_candidates
from .base import Candidate
from .kancelaryzmy import kancelaryzm_candidates
from .lemma_engine import lemma_swap_candidates
from .legal_ai_style import legal_ai_style_candidates
from .legal_style import legal_style_candidates
from .passive_voice import passive_candidates
from .sentence_flow import sentence_flow_candidates
from .cleanup import cleanup_candidates
from .features import ParagraphFeatures, SentenceFeatures, analyze_sentence_features
from .scoring import score_candidate


@dataclass
class RuleEngine:
    mode: Mode = Mode.conservative

    def generate_candidates(
        self,
        sentence: str,
        *,
        analysis=None,
        features: SentenceFeatures | None = None,
        paragraph_features: ParagraphFeatures | None = None,
        intensity: int | None = None,
    ) -> list[Candidate]:
        features = features or analyze_sentence_features(sentence)
        candidates: list[Candidate] = []
        candidates.extend(cleanup_candidates(sentence))
        candidates.extend(legal_style_candidates(sentence, mode=self.mode))
        candidates.extend(ai_artifact_candidates(sentence, mode=self.mode))
        candidates.extend(
            legal_ai_style_candidates(
                sentence,
                mode=self.mode,
                features=features,
                paragraph_features=paragraph_features,
                analysis=analysis,
            )
        )
        candidates.extend(
            kancelaryzm_candidates(sentence, mode=self.mode, analysis=analysis)
        )
        candidates.extend(
            lemma_swap_candidates(sentence, analysis=analysis, mode=self.mode)
        )
        candidates.extend(passive_candidates(sentence, analysis=analysis, mode=self.mode))
        if self.mode in {Mode.standard, Mode.strong}:
            candidates.extend(sentence_flow_candidates(sentence, mode=self.mode))

        candidates = [
            score_candidate(
                sentence,
                candidate,
                features=features,
                mode=self.mode,
                intensity=intensity,
                paragraph_features=paragraph_features,
            )
            for candidate in candidates
        ]

        # Higher score first; stable order within same score.
        return sorted(candidates, key=lambda c: c.score, reverse=True)
