from __future__ import annotations

from humanize_pl.rules.finite_verbs import FINITE_VERB_WORDS
from humanize_pl.sentence_splitter import split_sentences
from .validators import GateCheck, SENTENCE_TRANSITIONS


def stanza_finite_verb_gate(
    original: str,
    candidate: str,
    stanza_engine,
    *,
    analysis_cache: dict[str, object] | None = None,
) -> list[GateCheck]:
    """Use Stanza analysis to validate finite verbs in newly split sentences."""
    checks: list[GateCheck] = []
    parts = split_sentences(candidate)
    if len(parts) <= len(split_sentences(original)):
        checks.append(GateCheck("stanza_finite_verb", True))
        return checks

    for part in parts:
        checked = _without_transition(part)
        if not checked:
            checks.append(GateCheck("stanza_finite_verb", False, "empty split fragment"))
            return checks
        if analysis_cache is not None and checked in analysis_cache:
            analysis = analysis_cache[checked]
        else:
            analysis = stanza_engine.analyze_sentence(checked)
            if analysis_cache is not None:
                analysis_cache[checked] = analysis
        if not _analysis_has_finite_verb(analysis):
            checks.append(
                GateCheck(
                    "stanza_finite_verb",
                    False,
                    "split produced fragment without Stanza finite verb",
                )
            )
            return checks

    checks.append(GateCheck("stanza_finite_verb", True))
    return checks


def _analysis_has_finite_verb(analysis) -> bool:
    for token in getattr(analysis, "tokens", []):
        text = (getattr(token, "text", "") or "").lower()
        upos = getattr(token, "upos", None)
        feats = getattr(token, "feats", None) or ""
        if text in FINITE_VERB_WORDS:
            return True
        if upos in {"VERB", "AUX"} and ("VerbForm=Fin" in feats or "Tense=" in feats):
            return True
    return False


def _without_transition(sentence: str) -> str:
    body = sentence.strip()
    for transition in SENTENCE_TRANSITIONS:
        if body.lower().startswith(transition.lower()):
            return body[len(transition):].strip(" ,;")
    return body
