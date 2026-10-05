from humanize_pl.detect.lexical import connective_density, mtld

def test_connective_density_counts_target_connectives():
    text = "Zatem sąd uznał, że roszczenie jest zasadne. Ponadto, chociaż powód nie stawił się na rozprawie, to jednakże dowody były jasne."
    # Total words: 20
    # Connectives: Zatem, Ponadto, chociaż, jednakże (4)
    # Density: (4 / 20) * 1000 = 200.0
    density = connective_density(text)
    assert density == 200.0

def test_connective_density_is_zero_for_empty_text():
    assert connective_density("") == 0.0

def test_connective_density_ignores_similar_words():
    # "więc" is a connective, "więcej" is not (unless it's "co więcej").
    # "dlatego" is, "dla" is not.
    text = "Więcej danych dla sądu."
    density = connective_density(text)
    assert density == 0.0

def test_mtld_robustness():
    # Simple smoke test for MTLD
    text = "To jest krótki test. To jest drugi test."
    val = mtld(text)
    assert isinstance(val, float)
    assert val > 0.0


def test_polish_syllables_skip_a_softening_i():
    from humanize_pl.detect.lexical import syllables

    assert [syllables(w) for w in ("nie", "się", "pierwszy", "umowa", "Wynagrodzenie")] == [
        1, 1, 2, 3, 5
    ]


def test_fog_pl_rises_with_sentence_length_and_long_words():
    from humanize_pl.detect.lexical import fog_pl

    plain = [["Ala", "ma", "kota"], ["Kot", "ma", "Alę"]]
    assert fog_pl(plain) == 1.2  # 0.4 x (3 words per sentence + no hard words)
    legal = [["Odpowiedzialność", "odszkodowawcza", "przedsiębiorstwa", "obejmuje", "szkodę"]]
    assert fog_pl(legal) == 34.0  # 0.4 x (5 + 100 x 4/5): only "szkodę" is short
    assert fog_pl([]) == 0.0
