from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import regex as re
import yaml

from humanize_pl.config import Mode
from humanize_pl.nlp.morfeusz import MorfeuszAnalyzer, try_load_morfeusz
from humanize_pl.nlp.morphology import ger_to_infinitive
from .base import Candidate

_YAML = Path(__file__).parent / "nominalization.yaml"

_VERB_TAG_HEADS = frozenset({
    "inf", "fin", "praet", "imps", "impt", "pcon", "pant", "pact", "ppas", "winien", "pred",
})

# Dep relations indicating a noun complement/object of the light verb
_OBJ_DEP_RELS = frozenset({"obj", "obl", "iobj", "nmod"})
_RELATIVE_AFTER_NOUN_RE = re.compile(
    r"^\s*,?\s*(?:który|która|które|którzy|których|którym|którego|której|któremu|"
    r"którymi|którą)\b",
    re.IGNORECASE,
)


# ─── Data structures ─────────────────────────────────────────────────────────

class _RegexEntry(NamedTuple):
    pattern: re.Pattern
    replacement: str
    modes: frozenset[str]
    risk: float


class _NlpEntry(NamedTuple):
    replacement: str
    modes: frozenset[str]
    risk: float


@lru_cache(maxsize=1)
def _load_regex_entries() -> list[_RegexEntry]:
    with open(_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entries: list[_RegexEntry] = []
    for item in data.get("patterns", []):
        pat = re.compile(item["pattern"], re.IGNORECASE)
        modes = frozenset(item.get("modes", ["standard", "strong"]))
        risk = float(item.get("risk", 0.11))
        entries.append(_RegexEntry(pat, str(item["replacement"]), modes, risk))
    return entries


@lru_cache(maxsize=1)
def _load_nlp_table() -> dict[str, dict[str, _NlpEntry]]:
    """Returns {light_verb_lemma: {noun_lemma: NlpEntry}}."""
    with open(_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    table: dict[str, dict[str, _NlpEntry]] = {}
    for item in data.get("entries", []):
        verb = str(item["light_verb"]).lower()
        noun = str(item["noun"]).lower()
        repl = str(item["replacement"])
        modes = frozenset(item.get("modes", ["standard", "strong"]))
        risk = float(item.get("risk", 0.11))
        table.setdefault(verb, {})[noun] = _NlpEntry(repl, modes, risk)
    return table


# ─── Morfeusz form generation ─────────────────────────────────────────────────

def _generate_matching_form(
    light_verb_text: str,
    replacement_lemma: str,
    morfeusz: MorfeuszAnalyzer,
) -> str | None:
    """Return the form of replacement_lemma that matches light_verb_text morphologically.

    Strategy:
    1. Analyse light_verb_text with Morfeusz → collect its verb tags.
    2. Generate all forms of replacement_lemma → find exact tag match.
    3. Fall back to partial match on the first 3 tag components.
    """
    light_tags: set[str] = set()
    for a in morfeusz.analyses(light_verb_text.lower()):
        if a.tag.split(":")[0] in _VERB_TAG_HEADS:
            light_tags.add(a.tag)

    if not light_tags:
        return None

    repl_forms = morfeusz.generate(replacement_lemma)

    # Exact match
    for form in repl_forms:
        if form.tag in light_tags:
            return form.form

    # Partial match: first 3 tag components (verb_type : number/sg/pl : gender/person)
    light_partial = {":".join(t.split(":")[:3]) for t in light_tags}
    for form in repl_forms:
        if form.tag.split(":")[0] not in _VERB_TAG_HEADS:
            continue
        if ":".join(form.tag.split(":")[:3]) in light_partial:
            return form.form

    return None


# ─── NLP path ─────────────────────────────────────────────────────────────────

def _nlp_candidates(
    sentence: str,
    *,
    analysis,
    mode: Mode,
    morfeusz: MorfeuszAnalyzer,
) -> list[Candidate]:
    table = _load_nlp_table()
    active = mode.value
    candidates: list[Candidate] = []
    seen_verb_ids: set[int] = set()

    for tok in analysis.tokens:
        if tok.upos not in ("VERB", "AUX"):
            continue
        tok_id = getattr(tok, "id", None)
        if tok_id is None or tok_id in seen_verb_ids:
            continue
        tok_start = getattr(tok, "start_char", None)
        tok_end = getattr(tok, "end_char", None)
        if tok_start is None or tok_end is None:
            continue

        verb_lemma = (tok.lemma or tok.text).lower()
        noun_subtable = table.get(verb_lemma)
        if not noun_subtable:
            continue

        for dep in analysis.tokens:
            if getattr(dep, "head", None) != tok_id:
                continue
            if dep.upos != "NOUN":
                continue
            if (dep.deprel or "").lower() not in _OBJ_DEP_RELS:
                continue
            dep_start = getattr(dep, "start_char", None)
            dep_end = getattr(dep, "end_char", None)
            if dep_start is None or dep_end is None:
                continue
            if _has_relative_clause_after_removed_noun(sentence, dep_end):
                continue

            noun_lemma = (dep.lemma or dep.text).lower()
            entry = noun_subtable.get(noun_lemma)
            if entry is None:
                # Fallback: check if the surface noun form is a gerundive
                entry = _ger_auto_entry(dep.text, morfeusz, mode)
            if entry is None:
                continue
            if active not in entry.modes:
                continue

            repl_form = _generate_matching_form(tok.text, entry.replacement, morfeusz)
            if not repl_form:
                continue

            if tok.text[:1].isupper():
                repl_form = repl_form[:1].upper() + repl_form[1:]

            # Apply both edits from right to left to keep offsets valid
            ops = sorted(
                [
                    (tok_start, tok_end, repl_form),
                    (dep_start, dep_end, ""),
                ],
                key=lambda x: x[0],
                reverse=True,
            )
            result = sentence
            for start, end, new_text in ops:
                result = result[:start] + new_text + result[end:]
            result = re.sub(r"  +", " ", result).strip()

            if result == sentence:
                continue

            seen_verb_ids.add(tok_id)
            candidates.append(
                Candidate(
                    result,
                    f"nominalizacja:nlp:{verb_lemma}+{noun_lemma}",
                    0.64,
                    stage="legal_rewrite",
                    operation_type="debureaucratization",
                    risk=entry.risk,
                    targeted_issue="nominalization",
                )
            )
            break  # one substitution per verb token

    return candidates


def _has_relative_clause_after_removed_noun(sentence: str, noun_end: int) -> bool:
    """Avoid removing a noun that anchors a following relative clause.

    Example: "podejmować działania, które zakłócają..." cannot become
    "działać, które zakłócają...".
    """
    return bool(_RELATIVE_AFTER_NOUN_RE.match(sentence[noun_end:]))


# ─── Ger auto-detection helpers ──────────────────────────────────────────────

_W_CELU_RE = re.compile(r"\bw\s+celu\s+(\p{L}+)", re.IGNORECASE)


def _ger_auto_entry(
    noun_text: str, morfeusz: MorfeuszAnalyzer, mode: Mode
) -> _NlpEntry | None:
    """Auto-detect gerundive noun → return a synthetic NlpEntry using its infinitive.

    Only fires for standard/strong modes.  Risk is intentionally higher (0.16)
    than manual entries because the replacement quality is not hand-verified.
    """
    if mode == Mode.conservative:
        return None
    infinitive = ger_to_infinitive(noun_text, morfeusz)
    if not infinitive:
        return None
    return _NlpEntry(infinitive, frozenset({"standard", "strong"}), 0.16)


def _w_celu_ger_candidates(
    sentence: str, *, mode: Mode, morfeusz: MorfeuszAnalyzer
) -> list[Candidate]:
    """Replace 'w celu GERUNDIVE' → 'aby INFINITIVE'.

    Example: 'w celu przeprowadzenia kontroli' → 'aby przeprowadzić kontroli'
    (the noun complement stays in its original case — an acceptable approximation).
    """
    if mode == Mode.conservative:
        return []
    candidates: list[Candidate] = []
    for m in _W_CELU_RE.finditer(sentence):
        ger_word = m.group(1)
        infinitive = ger_to_infinitive(ger_word, morfeusz)
        if not infinitive:
            continue
        phrase_start = m.start()
        # Preserve capitalisation of original phrase start
        if sentence[phrase_start].isupper():
            replacement = "Aby " + infinitive
        else:
            replacement = "aby " + infinitive
        result = sentence[:phrase_start] + replacement + sentence[m.end():]
        result = re.sub(r"  +", " ", result).strip()
        if result == sentence:
            continue
        candidates.append(
            Candidate(
                result,
                "nominalizacja:w_celu_ger",
                0.58,
                stage="legal_rewrite",
                operation_type="debureaucratization",
                risk=0.14,
                targeted_issue="nominalization",
            )
        )
    return candidates


# ─── Regex fallback ───────────────────────────────────────────────────────────

def _preserve_case(original: str, replacement: str) -> str:
    if original and original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _regex_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    active = mode.value
    entries = [e for e in _load_regex_entries() if active in e.modes]
    if not entries:
        return []

    candidates: list[Candidate] = []
    combined = sentence
    combined_risk = 0.0

    for entry in entries:
        if not entry.pattern.search(combined):
            continue
        combined = entry.pattern.sub(
            lambda m, r=entry.replacement: _preserve_case(m.group(0), r), combined
        )
        combined_risk = max(combined_risk, entry.risk)

    if combined != sentence:
        candidates.append(
            Candidate(
                combined,
                "nominalizacja:combined",
                0.62,
                stage="legal_rewrite",
                operation_type="debureaucratization",
                risk=combined_risk,
                targeted_issue="nominalization",
            )
        )

    for entry in entries:
        if not entry.pattern.search(sentence):
            continue
        result = entry.pattern.sub(
            lambda m, r=entry.replacement: _preserve_case(m.group(0), r),
            sentence,
            count=1,
        )
        if result != sentence:
            candidates.append(
                Candidate(
                    result,
                    f"nominalizacja:{entry.pattern.pattern}",
                    0.56,
                    stage="legal_rewrite",
                    operation_type="debureaucratization",
                    risk=entry.risk,
                    targeted_issue="nominalization",
                )
            )

    return candidates


# ─── Public API ───────────────────────────────────────────────────────────────

def nominalization_candidates(
    sentence: str,
    *,
    mode: Mode,
    analysis=None,
) -> list[Candidate]:
    morfeusz = try_load_morfeusz()
    candidates: list[Candidate] = []
    if analysis is not None and morfeusz is not None:
        candidates.extend(
            _nlp_candidates(sentence, analysis=analysis, mode=mode, morfeusz=morfeusz)
        )
    if not candidates:
        candidates.extend(_regex_candidates(sentence, mode=mode))
    if morfeusz is not None:
        candidates.extend(_w_celu_ger_candidates(sentence, mode=mode, morfeusz=morfeusz))
    return candidates
