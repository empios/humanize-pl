from __future__ import annotations

from dataclasses import dataclass

from humanize_pl.config import Mode

from .ai_artifacts import ai_artifact_candidates
from .base import Candidate
from .cleanup import cleanup_candidates
from .features import ParagraphFeatures, SentenceFeatures, analyze_sentence_features
from .genitive_chains import genitive_chain_candidates
from .house_style import house_style_candidates
from .kancelaryzmy import kancelaryzm_candidates
from .legal_ai_style import legal_ai_style_candidates
from .legal_style import legal_style_candidates
from .lemma_engine import lemma_swap_candidates
from .nominalization import nominalization_candidates
from .passive_voice import passive_candidates
from .scoring import score_candidate
from .sentence_flow import sentence_flow_candidates


@dataclass
class RuleEngine:
    mode: Mode = Mode.conservative
    # Terminology the office wrote down, applied rather than only reported.
    # Empty for every caller that has no profile, which is most of them.
    preferred_terms: dict[str, str] | None = None
    # Rule ids, or prefixes before a ":", that never produce a candidate.
    disabled_rules: frozenset[str] = frozenset()

    def _enabled(self, rule: str) -> bool:
        return not any(rule == name or rule.startswith(name + ":") for name in self.disabled_rules)

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
        # First among the rewrite rules: where the office has named the
        # term it wants, that decision outranks the engine's generic
        # preference for a different word.
        candidates.extend(
            house_style_candidates(
                sentence, mode=self.mode, preferred_terms=self.preferred_terms
            )
        )
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
        candidates.extend(nominalization_candidates(sentence, mode=self.mode, analysis=analysis))
        candidates.extend(genitive_chain_candidates(sentence, analysis=analysis, mode=self.mode))
        candidates.extend(
            lemma_swap_candidates(sentence, analysis=analysis, mode=self.mode)
        )
        candidates.extend(passive_candidates(sentence, analysis=analysis, mode=self.mode))
        if self.mode in {Mode.standard, Mode.strong}:
            candidates.extend(sentence_flow_candidates(sentence, mode=self.mode))
        if self.disabled_rules:
            candidates = [c for c in candidates if self._enabled(c.rule)]

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
