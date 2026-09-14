import pytest
from humanize_pl import humanize_text
from humanize_pl.config import Mode

def test_em_dash_is_normalized_to_en_dash():
    result = humanize_text("Sąd ustalił—wbrew twierdzeniom—że powództwo jest zasadne.", mode=Mode.standard)
    # the em-dash should be replaced by space + en-dash + space
    assert "Sąd ustalił – wbrew twierdzeniom – że" in result.text
