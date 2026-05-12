from __future__ import annotations

from dataclasses import dataclass

import pytest

from humanize_pl.nlp.stanza_engine import SentenceAnalysis, TokenInfo
from humanize_pl.safety.agreement import agreement_gate


@dataclass
class FakeStanza:
    """Stub stanza engine that returns scripted analyses keyed by sentence."""

    table: dict[str, list[TokenInfo]]

    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        if sentence not in self.table:
            return SentenceAnalysis(tokens=[])
        return SentenceAnalysis(tokens=self.table[sentence])


def _tokens(*specs: tuple) -> list[TokenInfo]:
    """Build a token list from compact specs.

    Each spec is (text, upos, feats, head, deprel) — id is assigned positionally
    starting at 1, and char offsets are computed from concatenated text plus a
    single space between tokens.
    """
    out: list[TokenInfo] = []
    cursor = 0
    for idx, spec in enumerate(specs, start=1):
        text, upos, feats, head, deprel = spec
        out.append(
            TokenInfo(
                text=text,
                lemma=text.lower(),
                upos=upos,
                feats=feats,
                head=head,
                deprel=deprel,
                id=idx,
                start_char=cursor,
                end_char=cursor + len(text),
            )
        )
        cursor += len(text) + 1
    return out


def test_no_nlp_no_morfeusz_is_noop():
    checks, _ = agreement_gate(
        "Pracownik wykonuje pracę.",
        "Pracownik wykonuje obowiązek.",
        stanza_engine=None,
        morfeusz=None,
    )
    assert len(checks) == 1
    assert checks[0].ok


def test_identical_inputs_pass():
    fake = FakeStanza({})
    checks, _ = agreement_gate(
        "Pracownik wykonuje pracę.",
        "Pracownik wykonuje pracę.",
        stanza_engine=fake,
        morfeusz=None,
    )
    assert all(check.ok for check in checks)


def test_np_agreement_break_is_caught():
    # candidate: "ważne decyzja" — adj is Nom Plur but noun is Nom Sing
    original = "duża decyzja zapadła."
    candidate = "ważne decyzja zapadła."
    fake = FakeStanza(
        {
            original: _tokens(
                ("duża", "ADJ", "Case=Nom|Gender=Fem|Number=Sing", 2, "amod"),
                ("decyzja", "NOUN", "Case=Nom|Gender=Fem|Number=Sing", 3, "nsubj"),
                ("zapadła", "VERB", "VerbForm=Fin|Number=Sing|Tense=Past|Gender=Fem", 0, "root"),
            ),
            candidate: _tokens(
                ("ważne", "ADJ", "Case=Nom|Gender=Neut|Number=Plur", 2, "amod"),
                ("decyzja", "NOUN", "Case=Nom|Gender=Fem|Number=Sing", 3, "nsubj"),
                ("zapadła", "VERB", "VerbForm=Fin|Number=Sing|Tense=Past|Gender=Fem", 0, "root"),
            ),
        }
    )
    checks, _ = agreement_gate(original, candidate, stanza_engine=fake, morfeusz=None)
    np_check = next((c for c in checks if c.name == "agreement_np"), None)
    assert np_check is not None
    assert not np_check.ok
    assert "decyzja" in np_check.reason


def test_np_agreement_do_no_harm_when_break_existed_in_original():
    # both texts share the same broken (modifier_lemma, head_lemma) pair —
    # rewrite did not introduce it, so the gate must pass.
    original = "duża decyzje są ważne."
    candidate = "duża decyzje pozostają ważne."
    bad_np = _tokens(
        ("duża", "ADJ", "Case=Nom|Gender=Fem|Number=Sing", 2, "amod"),
        ("decyzje", "NOUN", "Case=Nom|Gender=Fem|Number=Plur", 3, "nsubj"),
        ("są", "AUX", "VerbForm=Fin|Number=Plur|Person=3", 4, "cop"),
        ("ważne", "ADJ", "Case=Nom|Number=Plur", 0, "root"),
    )
    candidate_tokens = _tokens(
        ("duża", "ADJ", "Case=Nom|Gender=Fem|Number=Sing", 2, "amod"),
        ("decyzje", "NOUN", "Case=Nom|Gender=Fem|Number=Plur", 3, "nsubj"),
        ("pozostają", "VERB", "VerbForm=Fin|Number=Plur|Person=3", 0, "root"),
        ("ważne", "ADJ", "Case=Nom|Number=Plur", 3, "xcomp"),
    )
    fake = FakeStanza({original: bad_np, candidate: candidate_tokens})
    checks, _ = agreement_gate(original, candidate, stanza_engine=fake, morfeusz=None)
    np_check = next((c for c in checks if c.name == "agreement_np"), None)
    assert np_check is not None
    assert np_check.ok


def test_subject_verb_mismatch_caught():
    # Real bug class from current rule table: "decyzje mają istotne znaczenie"
    # → "decyzje jest ważne" (broken number agreement decyzje:Plur vs jest:Sing).
    original = "decyzje mają istotne znaczenie."
    candidate = "decyzje jest ważne."
    fake = FakeStanza(
        {
            original: _tokens(
                ("decyzje", "NOUN", "Case=Nom|Gender=Fem|Number=Plur", 2, "nsubj"),
                ("mają", "VERB", "VerbForm=Fin|Number=Plur|Person=3", 0, "root"),
                ("istotne", "ADJ", "Case=Acc|Gender=Neut|Number=Sing", 4, "amod"),
                ("znaczenie", "NOUN", "Case=Acc|Gender=Neut|Number=Sing", 2, "obj"),
            ),
            candidate: _tokens(
                ("decyzje", "NOUN", "Case=Nom|Gender=Fem|Number=Plur", 2, "nsubj"),
                ("jest", "AUX", "VerbForm=Fin|Number=Sing|Person=3", 0, "root"),
                ("ważne", "ADJ", "Case=Nom|Gender=Neut|Number=Sing", 2, "xcomp"),
            ),
        }
    )
    checks, _ = agreement_gate(original, candidate, stanza_engine=fake, morfeusz=None)
    sv_check = next((c for c in checks if c.name == "agreement_subject_verb"), None)
    assert sv_check is not None
    assert not sv_check.ok
    assert "decyzje" in sv_check.reason
    assert "jest" in sv_check.reason


def test_preposition_governs_wrong_case_caught():
    # "do" expects Genitive; a rewrite producing "do pracownik" (Nom) must fail.
    original = "umowa należy do pracownika."
    candidate = "umowa należy do pracownik."
    fake = FakeStanza(
        {
            original: _tokens(
                ("umowa", "NOUN", "Case=Nom|Number=Sing", 2, "nsubj"),
                ("należy", "VERB", "VerbForm=Fin|Number=Sing|Person=3", 0, "root"),
                ("do", "ADP", "AdpType=Prep", 4, "case"),
                ("pracownika", "NOUN", "Case=Gen|Number=Sing|Gender=Masc", 2, "obl"),
            ),
            candidate: _tokens(
                ("umowa", "NOUN", "Case=Nom|Number=Sing", 2, "nsubj"),
                ("należy", "VERB", "VerbForm=Fin|Number=Sing|Person=3", 0, "root"),
                ("do", "ADP", "AdpType=Prep", 4, "case"),
                ("pracownik", "NOUN", "Case=Nom|Number=Sing|Gender=Masc", 2, "obl"),
            ),
        }
    )
    checks, _ = agreement_gate(original, candidate, stanza_engine=fake, morfeusz=None)
    prep_check = next((c for c in checks if c.name == "agreement_preposition"), None)
    assert prep_check is not None
    assert not prep_check.ok
    assert "do" in prep_check.reason
    assert "Gen" in prep_check.reason


def test_diff_localization_ignores_unchanged_span():
    # Change is a single-token swap at position 1; the misagreement we plant
    # at position 10+ is well beyond the ±3 context window and must be ignored.
    common_tail = "ABC DEF GHI JKL MNO PQR STU VWX YZ_"
    original = f"Pracownik wykonuje pracę. {common_tail} złe pary tutaj."
    candidate = f"Wykonawca wykonuje pracę. {common_tail} złe pary tutaj."

    cand_tokens: list[TokenInfo] = []
    cursor = 0

    def add(text, upos, feats, head, deprel):
        nonlocal cursor
        tid = len(cand_tokens) + 1
        cand_tokens.append(
            TokenInfo(
                text=text,
                lemma=text.lower(),
                upos=upos,
                feats=feats,
                head=head,
                deprel=deprel,
                id=tid,
                start_char=cursor,
                end_char=cursor + len(text),
            )
        )
        cursor += len(text) + 1

    # Diff region:
    add("Wykonawca", "NOUN", "Case=Nom|Gender=Masc|Number=Sing", 2, "nsubj")
    add("wykonuje", "VERB", "VerbForm=Fin|Number=Sing|Person=3", 0, "root")
    add("pracę", "NOUN", "Case=Acc|Gender=Fem|Number=Sing", 2, "obj")
    # Filler tokens — neutral.
    for token in ("ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX", "YZ_"):
        add(token, "X", "", 0, "dep")
    # Misagreement WELL outside the diff context window:
    add("złe", "ADJ", "Case=Nom|Gender=Neut|Number=Plur", 14, "amod")
    add("pary", "NOUN", "Case=Nom|Gender=Fem|Number=Sing", 0, "root")
    add("tutaj", "ADV", "", 14, "advmod")

    fake = FakeStanza({candidate: cand_tokens, original: []})
    checks, _ = agreement_gate(original, candidate, stanza_engine=fake, morfeusz=None)
    assert all(check.ok for check in checks)


def test_lexical_check_flags_unknown_form():
    class StubMorfeusz:
        name = "stub"
        def has_form(self, word: str) -> bool:
            return word.lower() != "wykonawczacz"

    original = "Pracownik wykonuje pracę."
    candidate = "Pracownik wykonawczacz pracę."
    checks, _ = agreement_gate(
        original,
        candidate,
        stanza_engine=None,
        morfeusz=StubMorfeusz(),
    )
    lex = next((c for c in checks if c.name == "agreement_lexical"), None)
    assert lex is not None
    assert not lex.ok
    assert "wykonawczacz" in lex.reason


def test_lexical_check_ignores_protected_placeholders():
    class StubMorfeusz:
        name = "stub"
        def has_form(self, word: str) -> bool:
            return True

    original = "Pracownik wykonuje pracę zgodnie z __PROTECTED_0001__."
    candidate = "Wykonawca realizuje zadanie zgodnie z __PROTECTED_0001__."
    checks, _ = agreement_gate(
        original,
        candidate,
        stanza_engine=None,
        morfeusz=StubMorfeusz(),
    )
    lex = next((c for c in checks if c.name == "agreement_lexical"), None)
    assert lex is not None
    assert lex.ok


def test_empty_candidate_fails_fast():
    checks, _ = agreement_gate("Coś tam.", "", stanza_engine=None, morfeusz=None)
    assert len(checks) == 1
    assert not checks[0].ok


def test_correct_rewrite_passes_full_gate():
    original = "Pracodawca wypłaca wynagrodzenie."
    candidate = "Pracodawca przekazuje wynagrodzenie."

    class StubMorfeusz:
        name = "stub"
        def has_form(self, word: str) -> bool:
            return True

    fake = FakeStanza(
        {
            original: _tokens(
                ("Pracodawca", "NOUN", "Case=Nom|Gender=Masc|Number=Sing", 2, "nsubj"),
                ("wypłaca", "VERB", "VerbForm=Fin|Number=Sing|Person=3", 0, "root"),
                ("wynagrodzenie", "NOUN", "Case=Acc|Gender=Neut|Number=Sing", 2, "obj"),
            ),
            candidate: _tokens(
                ("Pracodawca", "NOUN", "Case=Nom|Gender=Masc|Number=Sing", 2, "nsubj"),
                ("przekazuje", "VERB", "VerbForm=Fin|Number=Sing|Person=3", 0, "root"),
                ("wynagrodzenie", "NOUN", "Case=Acc|Gender=Neut|Number=Sing", 2, "obj"),
            ),
        }
    )
    checks, _ = agreement_gate(
        original,
        candidate,
        stanza_engine=fake,
        morfeusz=StubMorfeusz(),
    )
    assert all(check.ok for check in checks)
    names = {c.name for c in checks}
    assert {"agreement_lexical", "agreement_np", "agreement_subject_verb", "agreement_preposition"} <= names


def test_humanize_text_works_with_gate_disabled_no_nlp():
    """Smoke test: with no NLP and gate flag toggled, basic engine still humanizes."""
    from humanize_pl import humanize_text

    result = humanize_text(
        "Należy zauważyć, że niniejszy dokument przedstawia zasady.",
        agreement_gate_enabled=False,
    )
    assert "warto zauważyć" in result.text.lower()


def test_humanize_text_basic_engine_with_gate_enabled_does_no_harm():
    """Without Morfeusz/Stanza available, gate must be inert."""
    from humanize_pl import humanize_text

    result = humanize_text(
        "Należy zauważyć, że niniejszy dokument przedstawia zasady.",
        agreement_gate_enabled=True,
    )
    # The classic kancelaryzm rewrite must still fire when nothing blocks it.
    assert "warto zauważyć" in result.text.lower()


def test_humanize_text_records_morfeusz_status():
    from humanize_pl import humanize_text

    result = humanize_text(
        "Pracownik wykonuje pracę.",
        agreement_gate_enabled=True,
    )
    assert "morfeusz" in result.model_status
    assert result.model_status["morfeusz"] in {"ready", "not_requested"} or result.model_status[
        "morfeusz"
    ].startswith("unavailable")


def test_np_agreement_auto_repaired_by_morfeusz():
    """When NP agreement fails but Morfeusz knows the correct form, the gate repairs it."""
    from humanize_pl.nlp.morfeusz import MorfeuszAnalysis

    original = "duża decyzja zapadła."
    # "ważne" is Nom/Neut/Plur but "decyzja" is Nom/Fem/Sing — mismatch.
    candidate = "ważne decyzja zapadła."

    # Token positions in candidate: "ważne "(0–5) "decyzja "(6–13) "zapadła."(14–22)
    cand_tokens = [
        TokenInfo(
            text="ważne", lemma="ważny", upos="ADJ",
            feats="Case=Nom|Gender=Neut|Number=Plur",
            head=2, deprel="amod", id=1, start_char=0, end_char=5,
        ),
        TokenInfo(
            text="decyzja", lemma="decyzja", upos="NOUN",
            feats="Case=Nom|Gender=Fem|Number=Sing",
            head=3, deprel="nsubj", id=2, start_char=6, end_char=13,
        ),
        TokenInfo(
            text="zapadła", lemma="zapadać", upos="VERB",
            feats="VerbForm=Fin|Number=Sing|Tense=Past|Gender=Fem",
            head=0, deprel="root", id=3, start_char=14, end_char=21,
        ),
    ]
    orig_tokens = _tokens(
        ("duża", "ADJ", "Case=Nom|Gender=Fem|Number=Sing", 2, "amod"),
        ("decyzja", "NOUN", "Case=Nom|Gender=Fem|Number=Sing", 3, "nsubj"),
        ("zapadła", "VERB", "VerbForm=Fin|Number=Sing|Tense=Past|Gender=Fem", 0, "root"),
    )
    fake = FakeStanza({original: orig_tokens, candidate: cand_tokens})

    class RepairMorfeusz:
        name = "repair_stub"

        def has_form(self, word: str) -> bool:
            return True

        def generate(self, lemma: str) -> list[MorfeuszAnalysis]:
            if lemma == "ważny":
                return [
                    MorfeuszAnalysis("ważna", "ważny", "adj:sg:nom:f:pos"),
                    MorfeuszAnalysis("ważne", "ważny", "adj:sg:nom:n:pos"),
                    MorfeuszAnalysis("ważny", "ważny", "adj:sg:nom:m1:pos"),
                    MorfeuszAnalysis("ważnych", "ważny", "adj:pl:gen:m1:pos"),
                ]
            return []

    checks, repaired_text = agreement_gate(
        original, candidate, stanza_engine=fake, morfeusz=RepairMorfeusz()
    )
    np_check = next((c for c in checks if c.name == "agreement_np"), None)
    assert np_check is not None
    assert np_check.ok, f"Expected NP check to pass after repair; reason: {np_check.reason}"
    assert "repaired" in np_check.reason.lower()
    assert repaired_text == "ważna decyzja zapadła."


def test_require_morfeusz_raises_when_unavailable(monkeypatch):
    import humanize_pl.core as core_mod
    from humanize_pl import humanize_text

    class BoomAnalyzer:
        def __init__(self) -> None:
            raise RuntimeError("morfeusz2 not installed in this env")

    monkeypatch.setattr(core_mod, "MorfeuszAnalyzer", BoomAnalyzer)

    with pytest.raises(RuntimeError, match="Required Morfeusz"):
        humanize_text("Pracownik wykonuje pracę.", require_morfeusz=True)
