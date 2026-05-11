from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

import regex as re

from humanize_pl.nlp.morfeusz import MorfeuszAnalyzer, WHITELIST as MORF_WHITELIST
from .validators import GateCheck

WORD_RE = re.compile(r"\p{L}+")
PROTECTED_RE = re.compile(r"__PROTECTED_\d+__")
CONTEXT_TOKENS = 3

# Prepositions with unambiguous case government. Ambiguous ones (na, w, o, po,
# nad, pod, przed, za, między) intentionally excluded — they require semantic
# disambiguation we don't have without an LLM and would produce false alarms.
PREP_CASE: dict[str, str] = {
    # Genitive
    "bez": "Gen", "beze": "Gen", "dla": "Gen", "do": "Gen", "od": "Gen", "ode": "Gen",
    "u": "Gen", "z": "Gen", "ze": "Gen", "spod": "Gen", "sprzed": "Gen", "znad": "Gen",
    "spomiędzy": "Gen", "spośród": "Gen", "koło": "Gen", "obok": "Gen", "wokół": "Gen",
    "wzdłuż": "Gen", "oprócz": "Gen", "podczas": "Gen", "wśród": "Gen", "zamiast": "Gen",
    "naprzeciw": "Gen", "wedle": "Gen", "według": "Gen", "dookoła": "Gen", "naokoło": "Gen",
    "wewnątrz": "Gen", "mimo": "Gen", "pomimo": "Gen",
    # Dative
    "ku": "Dat", "dzięki": "Dat", "wbrew": "Dat", "przeciw": "Dat",
    "przeciwko": "Dat", "wobec": "Dat",
    # Accusative
    "przez": "Acc", "poprzez": "Acc",
}

# UD deprels that mark a modifier expected to agree with its head (NP-internal).
NP_AGREEING_DEPRELS = {
    "amod",
    "det",
    "det:poss",
    "nummod",
    "nummod:gov",
}

# UD deprels that mark a subject of a finite verb.
SUBJECT_DEPRELS = {"nsubj", "nsubj:pass", "csubj"}


@dataclass
class AgreementGateResult:
    checks: list[GateCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def agreement_gate(
    original: str,
    candidate: str,
    *,
    stanza_engine: Any = None,
    morfeusz: MorfeuszAnalyzer | None = None,
    analysis_cache: dict[str, Any] | None = None,
) -> list[GateCheck]:
    """Validate that `candidate` did not introduce Polish morphology bugs.

    The gate is diff-localized: it only inspects spans that changed between
    `original` and `candidate`, plus a small context window. Within each
    modified span we run three classes of checks:

      1. lexical validity (every alphabetic token is recognized),
      2. NP-internal agreement on Case/Number/Gender,
      3. subject-verb agreement on Number/Person/Gender,
      4. prepositional case government for unambiguous prepositions.

    The gate degrades gracefully: with no Stanza and no Morfeusz it is a
    no-op (returns a single passing check). With Morfeusz only it does
    lexical validity. With Stanza it does all of the above.
    """
    if not candidate.strip():
        return [GateCheck("agreement", False, "empty candidate")]
    if candidate == original:
        return [GateCheck("agreement", True)]
    if stanza_engine is None and morfeusz is None:
        return [GateCheck("agreement", True)]

    cand_words = list(WORD_RE.finditer(candidate))
    if not cand_words:
        return [GateCheck("agreement", True)]

    diff_spans = _modified_word_spans(original, candidate)
    if not diff_spans:
        return [GateCheck("agreement", True)]

    char_spans = _char_spans_from_word_spans(cand_words, diff_spans)

    checks: list[GateCheck] = []

    if morfeusz is not None:
        lex_check = _lexical_check(candidate, char_spans, morfeusz)
        checks.append(lex_check)
        if not lex_check.ok:
            return checks

    if stanza_engine is None:
        checks.append(GateCheck("agreement", True))
        return checks

    cand_analysis = _analyze(stanza_engine, candidate, analysis_cache)
    if cand_analysis is None or not getattr(cand_analysis, "tokens", None):
        checks.append(GateCheck("agreement", True))
        return checks
    orig_analysis = _analyze(stanza_engine, original, analysis_cache)

    np_failure = _np_agreement_failure(
        cand_analysis,
        orig_analysis,
        char_spans=char_spans,
    )
    if np_failure:
        checks.append(GateCheck("agreement_np", False, np_failure))
        return checks
    checks.append(GateCheck("agreement_np", True))

    sv_failure = _subject_verb_failure(
        cand_analysis,
        orig_analysis,
        char_spans=char_spans,
    )
    if sv_failure:
        checks.append(GateCheck("agreement_subject_verb", False, sv_failure))
        return checks
    checks.append(GateCheck("agreement_subject_verb", True))

    prep_failure = _preposition_government_failure(
        cand_analysis,
        orig_analysis,
        char_spans=char_spans,
    )
    if prep_failure:
        checks.append(GateCheck("agreement_preposition", False, prep_failure))
        return checks
    checks.append(GateCheck("agreement_preposition", True))

    return checks


def _analyze(stanza_engine: Any, text: str, cache: dict[str, Any] | None):
    if not text or not text.strip():
        return None
    if cache is not None and text in cache:
        return cache[text]
    try:
        analysis = stanza_engine.analyze_sentence(text)
    except Exception:
        return None
    if cache is not None:
        cache[text] = analysis
    return analysis


def _modified_word_spans(original: str, candidate: str) -> list[tuple[int, int]]:
    """Return inclusive (start_word_idx, end_word_idx) ranges of changes in candidate."""
    orig_words = [match.group(0).lower() for match in WORD_RE.finditer(original)]
    cand_words = [match.group(0).lower() for match in WORD_RE.finditer(candidate)]
    if orig_words == cand_words:
        return []
    matcher = SequenceMatcher(a=orig_words, b=cand_words, autojunk=False)
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            # nothing in candidate to inspect for a pure delete; we still
            # widen the context around the deletion point.
            spans.append((max(0, j1 - CONTEXT_TOKENS), min(len(cand_words), j1 + CONTEXT_TOKENS)))
            continue
        start = max(0, j1 - CONTEXT_TOKENS)
        end = min(len(cand_words), j2 + CONTEXT_TOKENS)
        spans.append((start, end))
    return _merge_spans(spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _char_spans_from_word_spans(
    word_matches: list[re.Match[str]],
    word_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start_idx, end_idx in word_spans:
        if start_idx >= len(word_matches):
            continue
        end_idx_clamped = min(end_idx, len(word_matches))
        if end_idx_clamped <= start_idx:
            continue
        start_char = word_matches[start_idx].start()
        end_char = word_matches[end_idx_clamped - 1].end()
        out.append((start_char, end_char))
    return out


def _token_in_spans(start: int | None, end: int | None, char_spans: list[tuple[int, int]]) -> bool:
    if start is None or end is None or not char_spans:
        return False
    for span_start, span_end in char_spans:
        if start < span_end and end > span_start:
            return True
    return False


def _lexical_check(
    candidate: str,
    char_spans: list[tuple[int, int]],
    morfeusz: MorfeuszAnalyzer,
) -> GateCheck:
    bad: list[str] = []
    for match in WORD_RE.finditer(candidate):
        if not _token_in_spans(match.start(), match.end(), char_spans):
            continue
        token = match.group(0)
        lower = token.lower()
        if lower in MORF_WHITELIST:
            continue
        if "PROTECTED" in token.upper():
            continue
        if morfeusz.has_form(token):
            continue
        # Treat single-letter tokens and short uppercase acronyms as benign:
        # SGJP regularly lacks these and Morfeusz has no good signal on them.
        if len(token) <= 2:
            continue
        if token.isupper() and len(token) <= 5:
            continue
        bad.append(token)
        if len(bad) >= 3:
            break
    if bad:
        return GateCheck(
            "agreement_lexical",
            False,
            f"unknown Polish form(s) introduced: {', '.join(bad)}",
        )
    return GateCheck("agreement_lexical", True)


def _feats(token: Any) -> dict[str, str]:
    raw = getattr(token, "feats", None) or ""
    out: dict[str, str] = {}
    for part in raw.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _token_by_id(analysis: Any, token_id: Any) -> Any:
    if token_id is None:
        return None
    for token in analysis.tokens:
        if token.id == token_id:
            return token
    return None


def _disagrees(
    modifier_feats: dict[str, str],
    head_feats: dict[str, str],
    *,
    keys: Iterable[str],
) -> str | None:
    """Return a description of the first feature mismatch, or None."""
    for key in keys:
        mod = modifier_feats.get(key)
        head = head_feats.get(key)
        if not mod or not head:
            continue
        if mod != head:
            return f"{key}={mod} vs head {key}={head}"
    return None


def _np_pair_signature(modifier: Any, head: Any) -> tuple[str, str]:
    mod_lemma = (modifier.lemma or modifier.text or "").lower()
    head_lemma = (head.lemma or head.text or "").lower()
    return (mod_lemma, head_lemma)


def _np_agreement_failure(
    candidate_analysis: Any,
    original_analysis: Any | None,
    *,
    char_spans: list[tuple[int, int]],
) -> str | None:
    original_pairs = _np_disagreement_pairs(original_analysis) if original_analysis else set()
    for token in candidate_analysis.tokens:
        deprel = (token.deprel or "").lower()
        if deprel not in NP_AGREEING_DEPRELS:
            continue
        upos = (token.upos or "").upper()
        if upos not in {"ADJ", "DET", "NUM", "PRON", "VERB"}:
            continue
        head = _token_by_id(candidate_analysis, token.head)
        if head is None or (head.upos or "").upper() not in {"NOUN", "PROPN", "PRON"}:
            continue
        if not _token_in_spans(token.start_char, token.end_char, char_spans) and not _token_in_spans(
            head.start_char, head.end_char, char_spans
        ):
            continue
        modifier_feats = _feats(token)
        head_feats = _feats(head)
        mismatch = _disagrees(modifier_feats, head_feats, keys=("Case", "Number", "Gender"))
        if not mismatch:
            continue
        signature = _np_pair_signature(token, head)
        if signature in original_pairs:
            # Same disagreement already existed in original; not introduced
            # by this rewrite — do no harm.
            continue
        return (
            f"NP agreement broken on '{token.text}' ↔ '{head.text}' ({mismatch})"
        )
    return None


def _np_disagreement_pairs(analysis: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if analysis is None:
        return pairs
    for token in analysis.tokens:
        deprel = (token.deprel or "").lower()
        if deprel not in NP_AGREEING_DEPRELS:
            continue
        head = _token_by_id(analysis, token.head)
        if head is None:
            continue
        if (head.upos or "").upper() not in {"NOUN", "PROPN", "PRON"}:
            continue
        if _disagrees(_feats(token), _feats(head), keys=("Case", "Number", "Gender")):
            pairs.add(_np_pair_signature(token, head))
    return pairs


def _subject_verb_failure(
    candidate_analysis: Any,
    original_analysis: Any | None,
    *,
    char_spans: list[tuple[int, int]],
) -> str | None:
    original_pairs = _sv_disagreement_pairs(original_analysis) if original_analysis else set()
    for verb in candidate_analysis.tokens:
        if (verb.upos or "").upper() not in {"VERB", "AUX"}:
            continue
        feats = _feats(verb)
        if feats.get("VerbForm") not in {"Fin", None}:
            continue
        subject = _subject_for_verb(candidate_analysis, verb)
        if subject is None:
            continue
        if not _token_in_spans(verb.start_char, verb.end_char, char_spans) and not _token_in_spans(
            subject.start_char, subject.end_char, char_spans
        ):
            continue
        sub_feats = _feats(subject)
        keys = ["Number", "Person"]
        if feats.get("Tense") == "Past" or feats.get("VerbForm") == "Part":
            keys.append("Gender")
        mismatch = _disagrees(sub_feats, feats, keys=keys)
        if not mismatch:
            continue
        signature = _sv_pair_signature(subject, verb)
        if signature in original_pairs:
            continue
        return (
            f"subject–verb agreement broken on '{subject.text}' ↔ '{verb.text}' ({mismatch})"
        )
    return None


def _subject_for_verb(analysis: Any, verb: Any) -> Any:
    for token in analysis.tokens:
        if (token.deprel or "").lower() in SUBJECT_DEPRELS and token.head == verb.id:
            return token
    return None


def _sv_pair_signature(subject: Any, verb: Any) -> tuple[str, str]:
    sub = (subject.lemma or subject.text or "").lower()
    v = (verb.lemma or verb.text or "").lower()
    return (sub, v)


def _sv_disagreement_pairs(analysis: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if analysis is None:
        return pairs
    for verb in analysis.tokens:
        if (verb.upos or "").upper() not in {"VERB", "AUX"}:
            continue
        feats = _feats(verb)
        if feats.get("VerbForm") not in {"Fin", None}:
            continue
        subject = _subject_for_verb(analysis, verb)
        if subject is None:
            continue
        keys = ["Number", "Person"]
        if feats.get("Tense") == "Past":
            keys.append("Gender")
        if _disagrees(_feats(subject), feats, keys=keys):
            pairs.add(_sv_pair_signature(subject, verb))
    return pairs


def _preposition_government_failure(
    candidate_analysis: Any,
    original_analysis: Any | None,
    *,
    char_spans: list[tuple[int, int]],
) -> str | None:
    original_pairs = (
        _prep_disagreement_pairs(original_analysis) if original_analysis else set()
    )
    for prep in candidate_analysis.tokens:
        if (prep.upos or "").upper() != "ADP":
            continue
        prep_form = (prep.text or "").lower()
        expected_case = PREP_CASE.get(prep_form)
        if expected_case is None:
            continue
        dependent = _prep_dependent(candidate_analysis, prep)
        if dependent is None:
            continue
        if not _token_in_spans(prep.start_char, prep.end_char, char_spans) and not _token_in_spans(
            dependent.start_char, dependent.end_char, char_spans
        ):
            continue
        actual_case = _feats(dependent).get("Case")
        if not actual_case or actual_case == expected_case:
            continue
        signature = (prep_form, (dependent.lemma or dependent.text or "").lower())
        if signature in original_pairs:
            continue
        return (
            f"preposition '{prep.text}' expects {expected_case} but governs "
            f"'{dependent.text}' in {actual_case}"
        )
    return None


def _prep_dependent(analysis: Any, prep: Any) -> Any:
    # In UD Polish, the preposition is attached as `case` to its NP head.
    if (prep.deprel or "").lower() != "case":
        return None
    head = _token_by_id(analysis, prep.head)
    if head is None:
        return None
    if (head.upos or "").upper() in {"NOUN", "PROPN", "PRON", "NUM", "ADJ"}:
        return head
    return None


def _prep_disagreement_pairs(analysis: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if analysis is None:
        return pairs
    for prep in analysis.tokens:
        if (prep.upos or "").upper() != "ADP":
            continue
        prep_form = (prep.text or "").lower()
        expected_case = PREP_CASE.get(prep_form)
        if expected_case is None:
            continue
        dependent = _prep_dependent(analysis, prep)
        if dependent is None:
            continue
        actual_case = _feats(dependent).get("Case")
        if actual_case and actual_case != expected_case:
            pairs.add((prep_form, (dependent.lemma or dependent.text or "").lower()))
    return pairs
