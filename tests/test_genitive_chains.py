import pytest

from humanize_pl.config import Mode
from humanize_pl.nlp.stanza_engine import StanzaEngine
from humanize_pl.rules.genitive_chains import genitive_chain_candidates

@pytest.fixture(scope="module")
def engine():
    return StanzaEngine()

def test_genitive_chains_are_simplified(engine):
    sentence = "W związku z tym powód wskazał cel zapewnienia możliwości realizacji projektu."
    analysis = engine.analyze_sentence(sentence)
    
    candidates = genitive_chain_candidates(sentence, analysis, mode=Mode.standard)
    
    # We should have a candidate matching the yaml
    assert any("aby umożliwić realizację" in c.text for c in candidates)

def test_genitive_chains_respect_mode(engine):
    sentence = "Skutek wejścia życia ustawy będzie zauważalny od razu."
    analysis = engine.analyze_sentence(sentence)
    
    # Assuming standard/strong mode enables it
    cands_standard = genitive_chain_candidates(sentence, analysis, mode=Mode.standard)
    assert len(cands_standard) > 0
    
    # Assuming conservative mode does not (as per YAML, modes are ["standard", "strong"])
    cands_conservative = genitive_chain_candidates(sentence, analysis, mode=Mode.conservative)
    assert len(cands_conservative) == 0

def test_no_genitive_chains_returns_empty(engine):
    sentence = "To jest proste zdanie."
    analysis = engine.analyze_sentence(sentence)
    
    candidates = genitive_chain_candidates(sentence, analysis, mode=Mode.standard)
    assert len(candidates) == 0
