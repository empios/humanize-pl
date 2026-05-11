"""Lemma-based rule engine.

Reads humanize_pl/rules/lemma_swaps.yaml, matches source lemmas against a
Stanza analysis of the sentence, inflects the target lemma to match the
source form's morphological features (via humanize_pl/nlp/inflector.py),
and emits Candidate objects. The downstream agreement gate validates the
substitution; if it broke morphology, the gate rejects it.

This module is intentionally narrow: it only handles 1-to-1 lemma swaps
where source and target share UPOS and inflection paradigm. Frame rewrites
(structural changes) live elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import regex as re
import yaml

from humanize_pl.config import Mode
from humanize_pl.nlp.inflector import Inflector, load_default, parse_stanza_feats
from .base import Candidate

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "lemma_swaps.yaml"
WORD_RE = re.compile(r"\p{L}+")


@dataclass(frozen=True)
class LemmaSwapRule:
    id: str
    from_lemma: str
    to_lemma: str
    upos: str | None
    score: float
    risk: float
    stage: str
    operation_type: str
    targeted_issue: str | None
    modes: frozenset[str]
    forbid_left_lemmas: frozenset[str]
    forbid_right_lemmas: frozenset[str]
    require_context_lemmas: frozenset[str]


def lemma_swap_candidates(
    sentence: str,
    *,
    analysis: Any,
    mode: Mode,
    rules: list[LemmaSwapRule] | None = None,
    inflector: Inflector | None = None,
) -> list[Candidate]:
    """Generate candidates by swapping lemmas as declared in lemma_swaps.yaml.

    Requires a Stanza analysis to identify lemma + features. Returns an
    empty list when analysis is missing or inflector has no data — the
    engine degrades gracefully into a no-op.
    """
    if analysis is None:
        return []
    rules = rules if rules is not None else _load_rules_cached()
    if not rules:
        return []
    inflector = inflector if inflector is not None else load_default()
    if inflector.is_empty:
        return []
    tokens = list(getattr(analysis, "tokens", []))
    if not tokens:
        return []

    mode_name = mode.value if hasattr(mode, "value") else str(mode)
    out: list[Candidate] = []
    seen_texts: set[str] = set()
    for idx, token in enumerate(tokens):
        token_lemma = (token.lemma or "").lower()
        if not token_lemma:
            continue
        for rule in rules:
            if mode_name not in rule.modes:
                continue
            if rule.from_lemma != token_lemma:
                continue
            if rule.upos and (token.upos or "").upper() != rule.upos.upper():
                continue
            if not _guards_pass(rule, tokens, idx):
                continue
            replacement = _build_replacement(token, rule, inflector)
            if replacement is None:
                continue
            candidate_text = _splice_token(sentence, token, replacement)
            if candidate_text is None or candidate_text == sentence:
                continue
            if candidate_text in seen_texts:
                continue
            seen_texts.add(candidate_text)
            out.append(
                Candidate(
                    candidate_text,
                    rule.id,
                    rule.score,
                    stage=rule.stage,
                    operation_type=rule.operation_type,
                    risk=rule.risk,
                    targeted_issue=rule.targeted_issue,
                )
            )
    return out


def _guards_pass(rule: LemmaSwapRule, tokens: list[Any], idx: int) -> bool:
    if rule.forbid_left_lemmas:
        for offset in (1, 2):
            j = idx - offset
            if j < 0:
                break
            if (tokens[j].lemma or "").lower() in rule.forbid_left_lemmas:
                return False
    if rule.forbid_right_lemmas:
        for offset in (1, 2):
            j = idx + offset
            if j >= len(tokens):
                break
            if (tokens[j].lemma or "").lower() in rule.forbid_right_lemmas:
                return False
    if rule.require_context_lemmas:
        window = tokens[max(0, idx - 4) : min(len(tokens), idx + 5)]
        lemmas = {(tok.lemma or "").lower() for tok in window}
        if not (rule.require_context_lemmas & lemmas):
            return False
    return True


def _build_replacement(token: Any, rule: LemmaSwapRule, inflector: Inflector) -> str | None:
    feats = parse_stanza_feats(getattr(token, "feats", None))
    lookup = inflector.inflect(
        rule.to_lemma,
        feats,
        upos=rule.upos or (token.upos or None),
    )
    if lookup is None:
        return None
    return _match_case(token.text or "", lookup.form)


def _splice_token(sentence: str, token: Any, replacement: str) -> str | None:
    start = getattr(token, "start_char", None)
    end = getattr(token, "end_char", None)
    if start is not None and end is not None and 0 <= start < end <= len(sentence):
        return sentence[:start] + replacement + sentence[end:]
    # Fallback: locate the surface form by regex. Used only when Stanza does
    # not expose char offsets (older versions or stub analyzers in tests).
    surface = token.text or ""
    if not surface:
        return None
    pattern = re.compile(rf"\b{re.escape(surface)}\b", re.IGNORECASE)
    match = pattern.search(sentence)
    if not match:
        return None
    return sentence[: match.start()] + _match_case(match.group(0), replacement) + sentence[match.end() :]


def _match_case(source: str, replacement: str) -> str:
    if not source or not replacement:
        return replacement
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


@lru_cache(maxsize=1)
def _load_rules_cached() -> list[LemmaSwapRule]:
    return load_rules(DEFAULT_RULES_PATH)


def load_rules(path: Path | str | None) -> list[LemmaSwapRule]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    rules: list[LemmaSwapRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            rule = _coerce_rule(entry)
        except (KeyError, TypeError, ValueError):
            continue
        rules.append(rule)
    return rules


def _coerce_rule(entry: dict[str, Any]) -> LemmaSwapRule:
    guards = entry.get("guards") or {}
    modes_value = entry.get("modes") or ["conservative", "standard", "strong"]
    return LemmaSwapRule(
        id=str(entry["id"]),
        from_lemma=str(entry["from_lemma"]).lower(),
        to_lemma=str(entry["to_lemma"]).lower(),
        upos=entry.get("upos"),
        score=float(entry.get("score", 0.5)),
        risk=float(entry.get("risk", 0.1)),
        stage=str(entry.get("stage", "legal_rewrite")),
        operation_type=str(entry.get("operation_type", "lemma_swap")),
        targeted_issue=entry.get("targeted_issue"),
        modes=frozenset(str(m) for m in modes_value),
        forbid_left_lemmas=frozenset(
            str(x).lower() for x in (guards.get("forbid_left_lemmas") or [])
        ),
        forbid_right_lemmas=frozenset(
            str(x).lower() for x in (guards.get("forbid_right_lemmas") or [])
        ),
        require_context_lemmas=frozenset(
            str(x).lower() for x in (guards.get("require_context_lemmas") or [])
        ),
    )
