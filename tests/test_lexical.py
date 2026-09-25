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
