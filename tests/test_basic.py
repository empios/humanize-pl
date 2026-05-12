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


def test_passive_morfeusz_extended():
    """Morfeusz oracle handles participles not in the hardcoded fast-path table."""
    from humanize_pl.rules.passive_voice import passive_candidates
    from humanize_pl.config import Mode

    cases = [
        # Feminine noun: nom → acc via Morfeusz
        ("Została podpisana umowa.", "Podpisano umowę."),
        # Masculine inanimate / neuter: nom == acc, stays unchanged
        ("Zostało wydane orzeczenie w tej sprawie.", "Wydano orzeczenie w tej sprawie."),
        # New verb (not in fast-path table)
        ("Zostało uchwalone rozporządzenie.", "Uchwalono rozporządzenie."),
    ]
    for sentence, expected in cases:
        cands = passive_candidates(sentence, mode=Mode.standard)
        assert cands, f"No candidate for: {sentence}"
        assert cands[0].text == expected, f"Got: {cands[0].text!r}"


def test_passive_accusative_morfeusz():
    """accusative_form_for_noun uses Morfeusz for feminine nouns, identity for others."""
    from humanize_pl.nlp.morfeusz import accusative_form_for_noun

    assert accusative_form_for_noun("analiza") == "analizę"
    assert accusative_form_for_noun("umowa") == "umowę"
    assert accusative_form_for_noun("ustawa") == "ustawę"
    # Masc inanimate and neuter: nom == acc
    assert accusative_form_for_noun("raport") == "raport"
    assert accusative_form_for_noun("badanie") == "badanie"
    assert accusative_form_for_noun("wyrok") == "wyrok"


def test_passive_np_adj_agreement():
    """passive_candidates inflects the full NP (adj+noun and noun+adj) to accusative."""
    from humanize_pl.rules.passive_voice import passive_candidates
    from humanize_pl.config import Mode

    cases = [
        # prenominal adj + feminine noun
        ("Zostało podpisane ważna umowa.", "Podpisano ważną umowę."),
        # postnominal adj after feminine noun (key legal pattern: "decyzja administracyjna")
        ("Została zatwierdzona decyzja administracyjna.", "Zatwierdzono decyzję administracyjną."),
        # prenominal adj + noun + postnominal adj (full NP)
        ("Została przygotowana obszerna analiza finansowa.", "Przygotowano obszerną analizę finansową."),
        ("Została przygotowana ważna ustawa budżetowa.", "Przygotowano ważną ustawę budżetową."),
        # neuter: nom == acc for both noun and adj — must stay unchanged
        ("Zostało wydane orzeczenie administracyjne.", "Wydano orzeczenie administracyjne."),
        ("Zostało wdrożone nowe rozwiązanie techniczne.", "Wdrożono nowe rozwiązanie techniczne."),
    ]
    for sentence, expected in cases:
        cands = passive_candidates(sentence, mode=Mode.standard)
        assert cands, f"No candidate for: {sentence}"
        assert cands[0].text == expected, f"Got: {cands[0].text!r}"


def test_passive_accusative_adj_morfeusz():
    """accusative_form_for_adj returns correct form based on noun gender."""
    from humanize_pl.nlp.morfeusz import accusative_form_for_adj

    # Feminine: adj acc ends in -ą
    assert accusative_form_for_adj("ważna", "decyzja") == "ważną"
    assert accusative_form_for_adj("obszerna", "analiza") == "obszerną"
    assert accusative_form_for_adj("administracyjna", "decyzja") == "administracyjną"
    assert accusative_form_for_adj("budżetowa", "ustawa") == "budżetową"
    # Neuter: nom == acc, unchanged
    assert accusative_form_for_adj("administracyjne", "orzeczenie") == "administracyjne"
    assert accusative_form_for_adj("nowe", "rozwiązanie") == "nowe"
    # Masc inanimate (m3): nom == acc, unchanged
    assert accusative_form_for_adj("ważny", "raport") == "ważny"


def test_protected_article():
    result = humanize_text("Zgodnie z art. 22 § 1 Kodeksu pracy należy zauważyć, że obowiązek istnieje.")
    assert "art. 22 § 1" in result.text


def test_nominalization_dokonac():
    from humanize_pl.rules.nominalization import nominalization_candidates
    from humanize_pl.config import Mode

    cases = [
        # perfective infinitive
        ("Należy dokonać analizy dokumentów.", "przeanalizować", Mode.standard),
        ("Należy dokonać wyboru spośród kandydatów.", "wybrać", Mode.standard),
        # impersonal past
        ("Dokonano ustalenia stanu faktycznego.", "ustalono", Mode.standard),
        ("Dokonano zmiany w umowie.", "zmieniono", Mode.standard),
        # imperfective infinitive
        ("Organ może dokonywać analizy akt.", "analizować", Mode.standard),
    ]
    for sentence, expected_frag, mode in cases:
        cands = nominalization_candidates(sentence, mode=mode)
        assert cands, f"No candidate for: {sentence!r}"
        assert any(expected_frag in c.text.lower() for c in cands), (
            f"Expected {expected_frag!r} in candidates for {sentence!r}; got: {[c.text for c in cands]}"
        )


def test_nominalization_przeprowadzic():
    from humanize_pl.rules.nominalization import nominalization_candidates
    from humanize_pl.config import Mode

    cases = [
        ("Należy przeprowadzić analizę ryzyka.", "przeanalizować", Mode.standard),
        ("Przeprowadzono kontrolę dokumentów.", "skontrolowano", Mode.standard),
        ("Należy przeprowadzić badanie sprawy.", "zbadać", Mode.standard),
        ("Organ może przeprowadzać ocenę ofert.", "oceniać", Mode.standard),
    ]
    for sentence, expected_frag, mode in cases:
        cands = nominalization_candidates(sentence, mode=mode)
        assert cands, f"No candidate for: {sentence!r}"
        assert any(expected_frag in c.text.lower() for c in cands), (
            f"Expected {expected_frag!r} in candidates for {sentence!r}; got: {[c.text for c in cands]}"
        )


def test_nominalization_no_conservative():
    """Nominalization rules are inactive in conservative mode."""
    from humanize_pl.rules.nominalization import nominalization_candidates
    from humanize_pl.config import Mode

    cands = nominalization_candidates("Organ dokonał analizy dokumentów.", mode=Mode.conservative)
    assert cands == []


def test_nominalization_nlp_path():
    """NLP path handles all conjugated forms via Stanza dep-parse + Morfeusz generate."""
    from humanize_pl.rules.nominalization import nominalization_candidates
    from humanize_pl.config import Mode

    class _T:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _A:
        def __init__(self, tokens):
            self.tokens = tokens

    # Build a fake Stanza analysis for each test case
    cases = [
        # (sentence, verb_text, verb_lemma, noun_text, noun_lemma, expected_frag)
        ("Komisja dokonała analizy.", "dokonała", "dokonać", "analizy", "analiza", "przeanalizowała"),
        ("Organ dokonał wyboru.", "dokonał", "dokonać", "wyboru", "wybór", "wybrał"),
        ("Oni dokonali ustalenia.", "dokonali", "dokonać", "ustalenia", "ustalenie", "ustalili"),
        ("Należy przeprowadzić kontrolę.", "przeprowadzić", "przeprowadzić", "kontrolę", "kontrola", "skontrolować"),
        ("Dokonano analizy.", "Dokonano", "dokonać", "analizy", "analiza", "przeanalizowano"),
    ]

    for sentence, v_text, v_lemma, n_text, n_lemma, expected_frag in cases:
        # Compute rough char offsets
        v_start = sentence.index(v_text)
        v_end = v_start + len(v_text)
        n_start = sentence.index(n_text)
        n_end = n_start + len(n_text)

        analysis = _A([
            _T(id=1, text=v_text, lemma=v_lemma, upos="VERB", deprel="root", head=0,
               start_char=v_start, end_char=v_end),
            _T(id=2, text=n_text, lemma=n_lemma, upos="NOUN", deprel="obj", head=1,
               start_char=n_start, end_char=n_end),
        ])

        cands = nominalization_candidates(sentence, mode=Mode.standard, analysis=analysis)
        assert cands, f"No NLP candidate for: {sentence!r}"
        assert any(expected_frag.lower() in c.text.lower() for c in cands), (
            f"Expected {expected_frag!r} in candidates for {sentence!r}; got: {[c.text for c in cands]}"
        )


def test_lix_score():
    from humanize_pl.nlp.morphology import lix_score

    # Simple text → low LIX
    assert lix_score("Ala ma kota.") < 30
    # Legal text → high LIX
    score = lix_score(
        "Postępowanie administracyjne zostało przeprowadzone z naruszeniem przepisów."
    )
    assert score > 55


def test_ger_to_infinitive():
    from humanize_pl.nlp.morphology import ger_to_infinitive
    from humanize_pl.nlp.morfeusz import try_load_morfeusz

    m = try_load_morfeusz()
    assert ger_to_infinitive("analizowania", m) == "analizować"
    assert ger_to_infinitive("przeprowadzenia", m) == "przeprowadzić"
    assert ger_to_infinitive("kontrolowania", m) == "kontrolować"
    # Pure noun — no ger analysis
    assert ger_to_infinitive("weryfikacji", m) is None


def test_w_celu_ger_candidates():
    from humanize_pl.rules.nominalization import nominalization_candidates
    from humanize_pl.config import Mode

    cands = nominalization_candidates(
        "W celu przeprowadzenia kontroli należy złożyć wniosek.", mode=Mode.standard
    )
    w_celu = [c for c in cands if "w_celu" in c.rule]
    assert w_celu, "No w_celu_ger candidate generated"
    assert any("aby przeprowadzić" in c.text.lower() for c in w_celu)


def test_lix_in_sentence_features():
    from humanize_pl.rules.features import analyze_sentence_features

    f_simple = analyze_sentence_features("Ala ma kota.")
    f_legal = analyze_sentence_features(
        "Postępowanie administracyjne zostało przeprowadzone z naruszeniem przepisów."
    )
    assert f_simple.lix < 30
    assert f_legal.lix > 55
    # MDD is None until enriched with Stanza analysis
    assert f_simple.mdd is None


def test_ger_auto_detection_nlp_path():
    """Ger auto-detection fires when light verb is known but noun is a gerundive not in table."""
    from humanize_pl.rules.nominalization import nominalization_candidates
    from humanize_pl.config import Mode

    class _T:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _A:
        def __init__(self, tokens):
            self.tokens = tokens

    sentence = "Organ dokonał analizowania dokumentów."
    v_start = sentence.index("dokonał")
    n_start = sentence.index("analizowania")
    analysis = _A([
        _T(id=1, text="dokonał", lemma="dokonać", upos="VERB", deprel="root", head=0,
           start_char=v_start, end_char=v_start + 7),
        _T(id=2, text="analizowania", lemma="analizowanie", upos="NOUN", deprel="obj",
           head=1, start_char=n_start, end_char=n_start + 12),
    ])
    cands = nominalization_candidates(sentence, mode=Mode.standard, analysis=analysis)
    nlp_cands = [c for c in cands if "nlp" in c.rule]
    assert nlp_cands, "No NLP candidate from ger auto-detection"
    assert any("analizował" in c.text.lower() for c in nlp_cands)


def test_co_do_zasady():
    from humanize_pl.rules.kancelaryzmy import kancelaryzm_candidates
    from humanize_pl.config import Mode

    cases = [
        ("Co do zasady wierzyciel może dochodzić naprawienia szkody.", "zasadniczo"),
        ("co do zasady pracownik nie może decydować samodzielnie.", "zasadniczo"),
    ]
    for sentence, expected in cases:
        cands = kancelaryzm_candidates(sentence, mode=Mode.standard)
        assert cands, f"No candidate for: {sentence!r}"
        assert any(expected in c.text.lower() for c in cands), (
            f"Expected {expected!r} in {[c.text for c in cands]}"
        )


def test_drop_intro_with_new_verbs():
    """drop_discourse_intro fires when remainder has verbs newly added to FINITE_VERB_WORDS."""
    from humanize_pl import humanize_text
    from humanize_pl.config import Mode

    cases = [
        "Warto wskazać, że platforma umożliwia użytkownikom zakładanie konta.",
        "Warto podkreślić, że blokada konta nie pozbawia użytkownika prawa.",
        "Warto podkreślić, że ocena odpowiedzialności wymaga ustalenia treści umowy.",
    ]
    for sentence in cases:
        result = humanize_text(sentence, mode=Mode.standard)
        assert result.changed, f"Expected change for: {sentence!r}"
        assert "Warto" not in result.text, (
            f"Expected discourse intro removed, got: {result.text!r}"
        )


def test_kancelaryzmy_b_additions():
    from humanize_pl.rules.kancelaryzmy import kancelaryzm_candidates
    from humanize_pl.config import Mode

    cases = [
        ("Decyzja jest słuszna, albowiem spełnia wymogi.", "ponieważ"),
        ("Wniosek jest zasadny, aczkolwiek wymaga uzupełnienia.", "choć"),
        ("Przepis stosuje się każdorazowo do nowych spraw.", "zawsze"),
        ("Sprawa w przedmiocie skargi podatkowej.", "w sprawie"),
        ("Należy postąpić jednakowoż zgodnie z przepisami.", "jednak"),
    ]
    for sentence, expected_frag in cases:
        cands = kancelaryzm_candidates(sentence, mode=Mode.standard)
        assert cands, f"No candidate for: {sentence!r}"
        assert any(expected_frag in c.text for c in cands), (
            f"Expected {expected_frag!r} in candidates for {sentence!r}; got: {[c.text for c in cands]}"
        )


def test_split_przy_czym_long_sentence():
    from humanize_pl.rules.sentence_flow import sentence_flow_candidates
    from humanize_pl.config import Mode

    # 33 words, "przy czym" in the middle, both halves have finite verbs, no legal refs near split
    sentence = (
        "Pracodawca ma obowiązek zapewnienia pracownikom odpowiednich warunków bezpieczeństwa "
        "i higieny pracy, a niedopełnienie tego obowiązku może prowadzić do odpowiedzialności "
        "cywilnej i karnej, przy czym pracownik zachowuje prawo do dochodzenia odszkodowania "
        "na drodze sądowej."
    )
    cands = sentence_flow_candidates(sentence, mode=Mode.standard)
    assert cands, "Expected a split_przy_czym candidate"
    assert cands[0].rule == "split_przy_czym"
    assert "Przy czym" in cands[0].text
    assert ". Przy czym" in cands[0].text


def test_no_split_przy_czym_short_sentence():
    from humanize_pl.rules.sentence_flow import sentence_flow_candidates
    from humanize_pl.config import Mode

    # Under 32 words → no split
    sentence = "Organ jest zobowiązany do rozpatrzenia wniosku, przy czym ma prawo do informacji."
    cands = sentence_flow_candidates(sentence, mode=Mode.standard)
    assert cands == []


def test_no_split_przy_czym_near_legal_ref():
    from humanize_pl.rules.sentence_flow import sentence_flow_candidates
    from humanize_pl.config import Mode

    # "Kodeksu pracy" is within 6 words of the split point → guard fires
    sentence = (
        "Pracownik nabywa prawo do urlopu w wymiarze proporcjonalnym do przepracowanego "
        "okresu, a zasady ustalania wymiaru urlopu są określone w przepisach Kodeksu pracy, "
        "przy czym strona ma prawo do wystąpienia z roszczeniem o ekwiwalent pieniężny."
    )
    cands = sentence_flow_candidates(sentence, mode=Mode.standard)
    assert cands == [], f"Expected no split near legal ref, got: {[c.text for c in cands]}"


def test_split_ponieważ_standard_mode():
    from humanize_pl.rules.sentence_flow import sentence_flow_candidates
    from humanize_pl.config import Mode

    # 32 words, ponieważ split available in standard (not just strong)
    sentence = (
        "Strona postępowania administracyjnego jest zobowiązana do niezwłocznego przedłożenia "
        "wszelkich dokumentów istotnych dla prawidłowego rozstrzygnięcia sprawy, ponieważ "
        "organ administracji publicznej nie może wydać decyzji administracyjnej bez kompletnego "
        "materiału dowodowego zebranego w danej sprawie."
    )
    cands_standard = sentence_flow_candidates(sentence, mode=Mode.standard)
    cands_conservative = sentence_flow_candidates(sentence, mode=Mode.conservative)
    assert cands_standard, "Expected ponieważ split in standard mode"
    assert cands_standard[0].rule == "split_causal"
    assert "Wynika to z tego, że" in cands_standard[0].text
    assert cands_conservative == [], "Conservative mode should not split"
