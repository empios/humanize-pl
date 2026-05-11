from humanize_pl import humanize_text


def test_no_english_markers():
    result = humanize_text("Szczególną rolę w systemie źródeł prawa pracy odgrywają układy zbiorowe pracy.")
    assert "surprisingly" not in result.text.lower()


def test_kancelaryzm():
    result = humanize_text("Należy zauważyć, że niniejszy dokument przedstawia zasady.")
    assert "warto zauważyć" in result.text.lower()
    assert "ten dokument" in result.text.lower()


def test_passive():
    result = humanize_text("Została przeprowadzona analiza danych.")
    assert result.text == "Przeprowadzono analizę danych."


def test_protected_article():
    result = humanize_text("Zgodnie z art. 22 § 1 Kodeksu pracy należy zauważyć, że obowiązek istnieje.")
    assert "art. 22 § 1" in result.text
