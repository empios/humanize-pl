import pytest
from humanize_pl import humanize_text
from humanize_pl.config import Mode

def test_em_dash_is_normalized_to_en_dash():
    result = humanize_text("Sąd ustalił—wbrew twierdzeniom—że powództwo jest zasadne.", mode=Mode.standard)
    # the em-dash should be replaced by space + en-dash + space
    assert "Sąd ustalił – wbrew twierdzeniom – że" in result.text


def test_a_markdown_rule_is_not_turned_into_a_stray_dash():
    """Measured on real model output: "---" became "– -" in 25 of 27
    documents, because the first two hyphens were read as a dash."""
    from humanize_pl.rules.cleanup import cleanup_candidates

    assert all("– -" not in row.text for row in cleanup_candidates("---"))
    assert any(row.text == "Sąd – jak wskazano – orzekł." for row in cleanup_candidates("Sąd -- jak wskazano -- orzekł."))


def test_a_blank_to_fill_in_keeps_its_space():
    """ "zawarta w dniu ……… r." glued into "w dniu………" breaks the form."""
    from humanize_pl.rules.cleanup import cleanup_candidates

    for sentence in ("zawarta w dniu .......... r. w Warszawie", "NIP: .........."):
        assert all(".........." in row.text and " .........." in row.text for row in cleanup_candidates(sentence))
    # An ordinary stray space before a full stop still goes.
    assert any(row.text == "Umowa wygasa." for row in cleanup_candidates("Umowa wygasa ."))
