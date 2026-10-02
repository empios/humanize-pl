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
    # Masc personal: acc == gen, tagged "gen.acc"
    assert accusative_form_for_noun("prezes") == "prezesa"


def test_passive_np_adj_agreement():
    """passive_candidates inflects the full NP (adj+noun and noun+adj) to accusative."""
    from humanize_pl.rules.passive_voice import passive_candidates
    from humanize_pl.config import Mode

    cases = [
        # prenominal adj + feminine noun
        ("Została podpisana ważna umowa.", "Podpisano ważną umowę."),
        # postnominal adj after feminine noun (key legal pattern: "decyzja administracyjna")
        ("Została zatwierdzona decyzja administracyjna.", "Zatwierdzono decyzję administracyjną."),
        # prenominal adj + noun + postnominal adj (full NP)
        ("Została przygotowana obszerna analiza finansowa.", "Przygotowano obszerną analizę finansową."),
        ("Została przygotowana ważna ustawa budżetowa.", "Przygotowano ważną ustawę budżetową."),
        # masculine personal: acc == gen for both noun and adj
        ("Został powołany nowy prezes zarządu.", "Powołano nowego prezesa zarządu."),
        # Morfeusz reads "umowa" and "nowa" as nouns and adjectives
        ("Została zawarta umowa następującej treści.", "Zawarto umowę następującej treści."),
        ("Została zawarta nowa umowa handlowa.", "Zawarto nową umowę handlową."),
        # Subject before the verb that reads the same in the accusative
        ("Raport został sporządzony zgodnie z umową.", "Raport sporządzono zgodnie z umową."),
        ("Towar odebrano, co zostało potwierdzone protokołem.",
         "Towar odebrano, co potwierdzono protokołem."),
        # Final punctuation is kept as it was, never added
        ("Została zawarta umowa najmu:", "Zawarto umowę najmu:"),
        ("(miejscowość, gdzie została podpisana umowa)", "(miejscowość, gdzie podpisano umowę)"),
        # neuter: nom == acc for both noun and adj — must stay unchanged
        ("Zostało wydane orzeczenie administracyjne.", "Wydano orzeczenie administracyjne."),
        ("Zostało wdrożone nowe rozwiązanie techniczne.", "Wdrożono nowe rozwiązanie techniczne."),
    ]
    for sentence, expected in cases:
        cands = passive_candidates(sentence, mode=Mode.standard)
        assert cands, f"No candidate for: {sentence}"
        assert cands[0].text == expected, f"Got: {cands[0].text!r}"


def test_passive_needs_an_agreeing_subject_after_the_participle(monkeypatch):
    """With the subject before the verb, conversion would leave it nominative
    ("Umowa sporządzono"), so no candidate is offered."""
    from humanize_pl.rules import passive_voice
    from humanize_pl.config import Mode

    for sentence in [
        "Umowa została sporządzona w dwóch jednobrzmiących egzemplarzach.",
        "Fundacja została ustanowiona przez fundatora aktem notarialnym.",
        "Uchwała została podjęta jednogłośnie.",
        # "poza" also reads as a noun ("pose"); the preposition reading wins
        "Ta umowa została zawarta poza lokalem przedsiębiorstwa.",
        # Coordination: "umowa użyczenia" would stay nominative
        "Została zawarta umowa najmu i umowa użyczenia.",
        "Raport został sporządzony i przekazany zarządowi.",
        # An impersonal form takes no agent
        "Została podpisana umowa przez obie strony.",
        # Masculine personal subject before the verb: "prezesa" would be needed
        "Prezes został powołany uchwałą zarządu.",
    ]:
        assert passive_voice.passive_candidates(sentence, mode=Mode.standard) == [], sentence

    # Without Morfeusz the subject cannot be checked or inflected
    monkeypatch.setattr(passive_voice, "try_load_morfeusz", lambda: None)
    assert passive_voice.passive_candidates(
        "Została przeprowadzona analiza danych.", mode=Mode.standard
    ) == []


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
        ("Należy niezwłocznie dokonać analizy.", "przeanalizować", Mode.standard),
        ("Należy dokonać wyboru spośród kandydatów.", "wybrać", Mode.standard),
        # impersonal past
        ("Dokonano ustalenia w toku postępowania.", "ustalono", Mode.standard),
        ("Dokonano zmiany w umowie.", "zmieniono", Mode.standard),
        # imperfective infinitive
        ("Organ może dokonywać analizy w każdym czasie.", "analizować", Mode.standard),
    ]
    # The deleted noun's genitive would stay behind: "przeanalizować dokumentów"
    assert not nominalization_candidates("Należy dokonać analizy dokumentów.", mode=Mode.standard)
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
        ("Należy przeprowadzić analizę w terminie 7 dni.", "przeanalizować", Mode.standard),
        ("Przeprowadzono kontrolę w siedzibie spółki.", "skontrolowano", Mode.standard),
        ("Należy przeprowadzić badanie.", "zbadać", Mode.standard),
        ("Organ może przeprowadzać ocenę corocznie.", "oceniać", Mode.standard),
    ]
    # The deleted noun's genitive would stay behind: "skontrolowano dokumentów"
    assert not nominalization_candidates("Przeprowadzono kontrolę dokumentów.", mode=Mode.standard)
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
    """'w celu GER' → ', aby INF' only where the result is correct Polish."""
    from humanize_pl.rules.nominalization import nominalization_candidates
    from humanize_pl.config import Mode

    def w_celu(sentence: str) -> list[str]:
        cands = nominalization_candidates(sentence, mode=Mode.standard)
        return [c.text for c in cands if c.rule == "nominalizacja:w_celu_ger"]

    expected = "Strona składa wniosek, aby skorzystać z ulgi."
    assert w_celu("Strona składa wniosek w celu skorzystania z ulgi.") == [expected]
    assert w_celu("Strona składa wniosek, w celu skorzystania z ulgi.") == [expected]
    for sentence in [
        # The object would stay genitive: "aby przeprowadzić kontroli"
        "Należy złożyć wniosek w celu przeprowadzenia kontroli.",
        # No main clause before it, so "Aby ..." would need a closing comma
        "W celu skorzystania z ulgi strona składa wniosek.",
        # An infinitive takes no agent: "aby usunąć przez Wynajmującego"
        "Najemca zgłasza usterkę w celu usunięcia przez Wynajmującego.",
    ]:
        assert w_celu(sentence) == [], sentence


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


def test_w_oparciu_o_takes_the_genitive():
    from humanize_pl.rules.kancelaryzmy import kancelaryzm_candidates
    from humanize_pl.safety.protectors import protect_text
    from humanize_pl.config import Mode

    cases = {
        "Administrator przetwarza dane w oparciu o przepisy rozporządzenia.":
            "Administrator przetwarza dane na podstawie przepisów rozporządzenia.",
        "Decyzję wydano w oparciu o obowiązujące przepisy prawa.":
            "Decyzję wydano na podstawie obowiązujących przepisów prawa.",
        "W oparciu o umowę Wykonawca wystawia fakturę.":
            "Na podstawie umowy Wykonawca wystawia fakturę.",
        "Spółka działa w oparciu o dane udostępnione przez Klienta.":
            "Spółka działa na podstawie danych udostępnionych przez Klienta.",
        # A protected reference ("art. 104") reads the same after both phrases
        protect_text("Decyzję wydano w oparciu o art. 104 k.p.a.").text:
            "Decyzję wydano na podstawie __PROTECTED_0000__ k.p.a.",
    }
    for sentence, expected in cases.items():
        texts = [c.text for c in kancelaryzm_candidates(sentence, mode=Mode.standard)]
        assert expected in texts, texts

    for sentence in [
        # A word Morfeusz cannot inflect
        "Działamy w oparciu o insighty klientów.",
        # Coordination: every conjunct would need the genitive
        "Spółka działa w oparciu o dane udostępnione i wytyczne udzielone przez Klienta.",
    ]:
        texts = [c.text for c in kancelaryzm_candidates(sentence, mode=Mode.standard)]
        assert not any("na podstawie" in text for text in texts), texts


def test_light_verb_reduction_keeps_genitive_chains_without_stanza():
    """"dokonać oceny ryzyka" → "ocenić ryzyka" would strand the genitive; with
    no Stanza analysis Morfeusz decides."""
    from humanize_pl.rules.kancelaryzmy import kancelaryzm_candidates
    from humanize_pl.config import Mode

    texts = [
        c.text
        for c in kancelaryzm_candidates("Zespół musi dokonać oceny ryzyka w ciągu tygodnia.", mode=Mode.standard)
    ]
    assert not any("ocenić ryzyka" in text for text in texts), texts
    texts = [
        c.text
        for c in kancelaryzm_candidates("Zespół musi dokonać oceny w ciągu tygodnia.", mode=Mode.standard)
    ]
    assert any("musi ocenić w ciągu tygodnia" in text for text in texts), texts


def test_light_verb_reductions_keep_the_deleted_nouns_dependents():
    """Deleting the complement noun must not strand its dependents: a genitive,
    an adjective or a "nad" phrase, with or without a Stanza parse."""
    from humanize_pl.config import Mode
    from humanize_pl.nlp.stanza_engine import SentenceAnalysis, TokenInfo
    from humanize_pl.rules.kancelaryzmy import kancelaryzm_candidates
    from humanize_pl.rules.nominalization import nominalization_candidates

    def texts(sentence: str, analysis=None) -> list[str]:
        cands = nominalization_candidates(sentence, mode=Mode.standard, analysis=analysis)
        cands += kancelaryzm_candidates(sentence, mode=Mode.standard, analysis=analysis)
        return [c.text for c in cands]

    for sentence, stranded in [
        ("Zarząd musi dokonać zmiany umowy w terminie.", "zmienić umowy"),
        ("Należy przeprowadzić analizę prawną sprawy.", "przeanalizować prawną"),
        ("Rada sprawuje nadzór nad spółką.", "nadzoruje nad"),
    ]:
        assert not any(stranded in text for text in texts(sentence)), sentence
    assert "Należy zmienić w terminie 7 dni." in texts("Należy dokonać zmiany w terminie 7 dni.")
    # "bez" is also a noun (lilac), but here it opens a prepositional phrase
    assert "Organ może oceniać bez zbędnej zwłoki." in texts(
        "Organ może przeprowadzać ocenę bez zbędnej zwłoki."
    )

    def parsed(words: list[tuple[str, str, str, int, str, str]]) -> SentenceAnalysis:
        tokens, start = [], 0
        for index, (text, lemma, upos, head, deprel, feats) in enumerate(words, start=1):
            tokens.append(TokenInfo(text, lemma, upos, feats, head, deprel, index, start, start + len(text)))
            start += len(text) + 1
        return SentenceAnalysis(tokens=tokens)

    # Stanza attaches the adjective to the noun the reduction deletes
    sentence = "Należy dokonać oceny prawnej"
    analysis = parsed([
        ("Należy", "należy", "VERB", 0, "root", ""),
        ("dokonać", "dokonać", "VERB", 1, "xcomp", ""),
        ("oceny", "ocena", "NOUN", 2, "obj", "Case=Gen"),
        ("prawnej", "prawny", "ADJ", 3, "amod", "Case=Gen"),
    ])
    assert not any("ocenić prawnej" in text for text in texts(sentence, analysis))
    sentence = "Należy przeprowadzić analizę prawną"
    analysis = parsed([
        ("Należy", "należy", "VERB", 0, "root", ""),
        ("przeprowadzić", "przeprowadzić", "VERB", 1, "xcomp", ""),
        ("analizę", "analiza", "NOUN", 2, "obj", "Case=Acc"),
        ("prawną", "prawny", "ADJ", 3, "amod", "Case=Acc"),
    ])
    assert not any("przeanalizować prawną" in text for text in texts(sentence, analysis))


def test_cleanup_spacing_keeps_layout_and_numbers():
    from humanize_pl.rules.cleanup import cleanup_candidates

    for sentence in [
        "Zamieszkały w Krakowie przy ul. Długiej.",  # non-breaking space
        "DARCZYŃCA\tOBDAROWANY",  # signature columns
        "Zamieszkały w ................... (adres).",  # fill-in blank
        "Kwota ……………… zł zostanie zapłacona w terminie ................ .",
        "Rodzaj pracy .……………………………",
        "Wynajmujący                    Najemca",  # aligned signature columns
        "............\t\t   ............",
        "Spotkanie o godz. 12:00, kwota 1,5 tys. zł.",  # time and decimal comma
    ]:
        assert cleanup_candidates(sentence) == [], sentence

    cands = cleanup_candidates("Strony ustalają  , że umowa obowiązuje.Od dziś.")
    assert cands[0].text == "Strony ustalają, że umowa obowiązuje.Od dziś."
    cands = cleanup_candidates("Strony ustalają,że umowa obowiązuje.")
    assert cands[0].text == "Strony ustalają, że umowa obowiązuje."


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


def test_split_keeps_coordinated_objects_together():
    """", a także o ..." joins two objects of one verb. Split off, the second
    half is a fragment whose only verb sits in a relative clause."""
    from humanize_pl.rules.sentence_flow import sentence_flow_candidates
    from humanize_pl.config import Mode

    sentence = (
        "Podmiot przetwarzający niezwłocznie informuje Administratora o każdym naruszeniu "
        "ochrony danych osobowych w swojej organizacji, a także o wszelkich planowanych "
        "zmianach w sposobie przetwarzania, które mogą wpłynąć na bezpieczeństwo "
        "przetwarzania danych osobowych przez Podmiot."
    )
    assert sentence_flow_candidates(sentence, mode=Mode.strong) == []


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
