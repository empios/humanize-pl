from __future__ import annotations

from dataclasses import dataclass

from humanize_pl.config import Mode
from humanize_pl.nlp.inflector import Inflector, parse_stanza_feats
from humanize_pl.nlp.stanza_engine import SentenceAnalysis, TokenInfo
from humanize_pl.rules.lemma_engine import (
    LemmaSwapRule,
    lemma_swap_candidates,
    load_rules,
)


@dataclass
class FakeStanza:
    table: dict[str, list[TokenInfo]]

    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        return SentenceAnalysis(tokens=self.table.get(sentence, []))


def _t(
    text: str,
    *,
    lemma: str,
    upos: str,
    feats: str,
    head: int,
    deprel: str,
    start: int,
    end: int,
    idx: int,
) -> TokenInfo:
    return TokenInfo(
        text=text,
        lemma=lemma,
        upos=upos,
        feats=feats,
        head=head,
        deprel=deprel,
        id=idx,
        start_char=start,
        end_char=end,
    )


# --- inflector unit tests -------------------------------------------------


def test_inflector_exact_match():
    inflector = Inflector(
        {
            "ten": {
                "upos": "DET",
                "forms": {
                    "Case=Nom|Gender=Masc|Number=Sing": "ten",
                    "Case=Gen|Gender=Fem|Number=Sing": "tej",
                },
            }
        }
    )
    lookup = inflector.inflect("ten", {"Case": "Gen", "Gender": "Fem", "Number": "Sing"})
    assert lookup is not None
    assert lookup.form == "tej"


def test_inflector_relaxes_animacy_first():
    # Caller asks for masc+anim+sg in Acc; data only has Acc/Masc/Sing/Inan
    # and a generic Case=Acc|Gender=Masc|Number=Sing fallback. We expect the
    # generic fallback only after Animacy is dropped.
    inflector = Inflector(
        {
            "ten": {
                "upos": "DET",
                "forms": {
                    "Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing": "ten",
                    "Case=Acc|Gender=Masc|Number=Sing": "ten",
                },
            }
        }
    )
    lookup = inflector.inflect(
        "ten",
        {"Case": "Acc", "Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
    )
    assert lookup is not None
    assert lookup.form == "ten"


def test_inflector_returns_none_for_unknown_lemma():
    inflector = Inflector({"ten": {"upos": "DET", "forms": {}}})
    assert inflector.inflect("nieznany", {"Case": "Nom"}) is None


def test_parse_stanza_feats():
    parsed = parse_stanza_feats("Case=Nom|Gender=Fem|Number=Sing")
    assert parsed == {"Case": "Nom", "Gender": "Fem", "Number": "Sing"}
    assert parse_stanza_feats(None) == {}


# --- lemma_engine unit tests ---------------------------------------------


NINIEJSZY_RULE = LemmaSwapRule(
    id="kancelaryzm:niniejszy_to_ten",
    from_lemma="niniejszy",
    to_lemma="ten",
    upos="ADJ",
    score=0.6,
    risk=0.06,
    stage="legal_rewrite",
    operation_type="debureaucratization",
    targeted_issue="bureaucratic_demonstrative",
    modes=frozenset({"conservative", "standard", "strong"}),
    forbid_left_lemmas=frozenset({"art", "paragraf"}),
    forbid_right_lemmas=frozenset({"kodeks", "ustawa"}),
    require_context_lemmas=frozenset(),
)


def _inflector():
    return Inflector(
        {
            "ten": {
                "upos": "DET",
                "forms": {
                    "Case=Nom|Gender=Masc|Number=Sing": "ten",
                    "Case=Nom|Gender=Fem|Number=Sing": "ta",
                    "Case=Nom|Gender=Neut|Number=Sing": "to",
                    "Case=Gen|Gender=Masc|Number=Sing": "tego",
                    "Case=Gen|Gender=Fem|Number=Sing": "tej",
                    "Case=Acc|Gender=Fem|Number=Sing": "tę",
                },
            }
        }
    )


def test_lemma_swap_substitutes_with_agreement_preservation():
    sentence = "Niniejsza umowa obowiązuje."
    fake = FakeStanza(
        {
            sentence: [
                _t(
                    "Niniejsza",
                    lemma="niniejszy",
                    upos="ADJ",
                    feats="Case=Nom|Gender=Fem|Number=Sing",
                    head=2,
                    deprel="amod",
                    start=0,
                    end=9,
                    idx=1,
                ),
                _t(
                    "umowa",
                    lemma="umowa",
                    upos="NOUN",
                    feats="Case=Nom|Gender=Fem|Number=Sing",
                    head=3,
                    deprel="nsubj",
                    start=10,
                    end=15,
                    idx=2,
                ),
                _t(
                    "obowiązuje",
                    lemma="obowiązywać",
                    upos="VERB",
                    feats="VerbForm=Fin|Number=Sing|Person=3",
                    head=0,
                    deprel="root",
                    start=16,
                    end=26,
                    idx=3,
                ),
            ]
        }
    )
    analysis = fake.analyze_sentence(sentence)
    candidates = lemma_swap_candidates(
        sentence,
        analysis=analysis,
        mode=Mode.conservative,
        rules=[NINIEJSZY_RULE],
        inflector=_inflector(),
    )
    assert len(candidates) == 1
    assert candidates[0].text == "Ta umowa obowiązuje."
    assert candidates[0].rule == "kancelaryzm:niniejszy_to_ten"


def test_lemma_swap_handles_masculine_noun():
    sentence = "Niniejszy dokument wskazuje."
    fake = FakeStanza(
        {
            sentence: [
                _t(
                    "Niniejszy",
                    lemma="niniejszy",
                    upos="ADJ",
                    feats="Case=Nom|Gender=Masc|Number=Sing|Animacy=Inan",
                    head=2,
                    deprel="amod",
                    start=0,
                    end=9,
                    idx=1,
                ),
                _t(
                    "dokument",
                    lemma="dokument",
                    upos="NOUN",
                    feats="Case=Nom|Gender=Masc|Number=Sing|Animacy=Inan",
                    head=3,
                    deprel="nsubj",
                    start=10,
                    end=18,
                    idx=2,
                ),
                _t(
                    "wskazuje",
                    lemma="wskazywać",
                    upos="VERB",
                    feats="VerbForm=Fin|Number=Sing|Person=3",
                    head=0,
                    deprel="root",
                    start=19,
                    end=27,
                    idx=3,
                ),
            ]
        }
    )
    cands = lemma_swap_candidates(
        sentence,
        analysis=fake.analyze_sentence(sentence),
        mode=Mode.conservative,
        rules=[NINIEJSZY_RULE],
        inflector=_inflector(),
    )
    assert len(cands) == 1
    assert cands[0].text == "Ten dokument wskazuje."


def test_lemma_swap_respects_forbid_left_lemmas_guard():
    sentence = "Art. niniejszy zawiera."
    fake = FakeStanza(
        {
            sentence: [
                _t(
                    "Art",
                    lemma="art",
                    upos="NOUN",
                    feats="",
                    head=0,
                    deprel="root",
                    start=0,
                    end=3,
                    idx=1,
                ),
                _t(
                    "niniejszy",
                    lemma="niniejszy",
                    upos="ADJ",
                    feats="Case=Nom|Gender=Masc|Number=Sing",
                    head=1,
                    deprel="amod",
                    start=5,
                    end=14,
                    idx=2,
                ),
                _t(
                    "zawiera",
                    lemma="zawierać",
                    upos="VERB",
                    feats="VerbForm=Fin|Number=Sing|Person=3",
                    head=0,
                    deprel="root",
                    start=15,
                    end=22,
                    idx=3,
                ),
            ]
        }
    )
    cands = lemma_swap_candidates(
        sentence,
        analysis=fake.analyze_sentence(sentence),
        mode=Mode.conservative,
        rules=[NINIEJSZY_RULE],
        inflector=_inflector(),
    )
    assert cands == []


def test_lemma_swap_returns_empty_without_analysis():
    cands = lemma_swap_candidates(
        "Niniejsza umowa.",
        analysis=None,
        mode=Mode.conservative,
        rules=[NINIEJSZY_RULE],
        inflector=_inflector(),
    )
    assert cands == []


def test_lemma_swap_returns_empty_with_empty_inflector():
    sentence = "Niniejsza umowa."
    fake = FakeStanza({sentence: []})
    cands = lemma_swap_candidates(
        sentence,
        analysis=fake.analyze_sentence(sentence),
        mode=Mode.conservative,
        rules=[NINIEJSZY_RULE],
        inflector=Inflector(None),
    )
    assert cands == []


def test_load_rules_reads_real_yaml():
    rules = load_rules(None)
    assert rules == []

    from humanize_pl.rules.lemma_engine import DEFAULT_RULES_PATH

    rules = load_rules(DEFAULT_RULES_PATH)
    assert any(r.id == "kancelaryzm:niniejszy_to_ten" for r in rules)


def test_real_inflections_json_loads():
    from humanize_pl.nlp.inflector import load_default

    inflector = load_default.__wrapped__()  # bypass cache
    assert inflector.has_lemma("ten")
    lookup = inflector.inflect(
        "ten",
        {"Case": "Nom", "Gender": "Fem", "Number": "Sing"},
    )
    assert lookup is not None
    assert lookup.form == "ta"


def test_case_preservation_in_substitution():
    sentence = "NINIEJSZA UMOWA."
    fake = FakeStanza(
        {
            sentence: [
                _t(
                    "NINIEJSZA",
                    lemma="niniejszy",
                    upos="ADJ",
                    feats="Case=Nom|Gender=Fem|Number=Sing",
                    head=2,
                    deprel="amod",
                    start=0,
                    end=9,
                    idx=1,
                ),
                _t(
                    "UMOWA",
                    lemma="umowa",
                    upos="NOUN",
                    feats="Case=Nom|Gender=Fem|Number=Sing",
                    head=0,
                    deprel="root",
                    start=10,
                    end=15,
                    idx=2,
                ),
            ]
        }
    )
    cands = lemma_swap_candidates(
        sentence,
        analysis=fake.analyze_sentence(sentence),
        mode=Mode.conservative,
        rules=[NINIEJSZY_RULE],
        inflector=_inflector(),
    )
    assert len(cands) == 1
    assert cands[0].text.startswith("TA ") or cands[0].text.startswith("Ta ")
