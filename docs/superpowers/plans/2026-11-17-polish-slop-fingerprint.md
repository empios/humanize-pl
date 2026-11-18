# Polski „odcisk slopu” — plan implementacji

> **Dla agentów:** WYMAGANY SUB-SKILL: superpowers:subagent-driven-development (zalecany) lub superpowers:executing-plans. Kroki w formacie `- [ ]`.

**Cel:** zastąpić zgadywane listy tików AI listą zmierzoną na korpusie, osobno dla gatunku prawnego i `general`, przejrzaną ręcznie i podpiętą do `detect/` jako sygnał tylko do diagnozy.

**Architektura:** `humanize_pl/corpus/fingerprint.py` wybiera kandydatów z istniejącego log-odds (`corpus/logodds.py`). Do odcięcia słownictwa tematycznego służą trzy filtry: zgodność między modelami, udział dokumentów i kontrola na odłożonym ludzkim korpusie. `tools/derive_patterns.py` scala kandydatów z wersjonowanym plikiem `humanize_pl/data/slop_fingerprint.yaml` i nie rusza decyzji recenzenta. `detect/` czyta wyłącznie wpisy `accepted` właściwego gatunku i zgłasza je jako rodzinę `slop_ngram` z `rewritable=False`.

**Stos:** Python 3.10+, `regex`, PyYAML, `corpus/logodds.py` (Monroe 2008), pytest.

**Spec:** raport „Kierunki rozwoju humanize pl 2026” (scratchpad sesji), sekcje „Ocena bez detektora i diagnoza zależna od gatunku” oraz „Mapa drogowa → Teraz”, pozycja 1.

## Stan wyjściowy (zmierzony 2026-11-17)

- `tools/derive_patterns.py` liczy już log-odds AI wobec SAOS, ale jego wynik nie trafia do żadnego modułu. `derived_v2.json` (27 dokumentów AI) pokazuje w czołówce temat, nie styl: „danych osobowych”, „kara umowna”, „ust lit”.
- Korpus AI: `docs_tests/corpus/ai_models/{haiku,opus,sonnet}` po 74 pliki (szkice `draft_*` i przepisania `rewrite_saos_*`). Ścieżka `general` ma tylko `docs_tests/corpus/general/haiku`, czyli 24 pliki jednego modelu.
- Korpus ludzki: `saos_train.jsonl` (1804) i `saos_holdout.jsonl` (592) dla gatunku prawnego. Dla `general`: `wiki_talk_train.jsonl` (5874), `allegro_reviews_train.jsonl` (8771), a jako odłożone `wiki_talk.jsonl` i `allegro_reviews.jsonl`.
- Cały `docs_tests/corpus/` jest w `.gitignore`. Testy budują więc syntetyczne korpusy w pamięci, a do repo trafia tylko `slop_fingerprint.yaml`.
- `humanize-pl-eval` liczy `detector_auc_drafts` na tych samych szkicach `ai_models`. Wyprowadzenie na całym korpusie zawyżyłoby AUC (in-sample), stąd podział na połowy w Task 1.

## Ograniczenia globalne

- Detektor nie trafia do bramek, do ewaluacji humanizera ani do wyboru checkpointu. `slop_ngram` to informacja dla użytkownika w `detect/` i nic więcej.
- Kandydat dotykający modalności deontycznej, odwołań (art, ust, pkt, lit, nr, poz, dz, kc, kpc, kp) lub liczb ma `protected: true` i nie może dostać statusu `accepted`. Loader odrzuca taki plik.
- Wpis `rejected` wymaga niepustego `note`.
- `classify_document` nigdy nie zwraca `general`. Gatunek `general` w diagnozie bierze się wyłącznie z `DocumentType.general` wybranego przez użytkownika.
- Wartości: `Z_THRESHOLD = 1.96` (bez zmian), `MIN_DOC_SHARE = 0.10`, `MIN_MODELS = 2` (dla korpusu jednego modelu: 1 plus flaga `indicative`), `MIN_HOLDOUT_RATIO = 2.0`, waga znaleziska `0.4`.
- Tekst dla użytkownika, notatki w YAML i README po polsku. Identyfikatory po angielsku. Commity w stylu `area: summary`.
- Żadnych plików tymczasowych w repo. Pomiary robić w `python -c` albo w scratchpadzie.
- Praca na gałęzi `detect/slop-fingerprint` od `main`. Push tylko na prośbę.

## Na co patrzeć w przeglądzie

1. **Fraza poprawna gatunkowo w prawie**, np. „na podstawie”, „w związku z”, „stanowi”. Kontrola odłożona musi ją odrzucić, bo w `saos_holdout` jest częsta. Test: `test_phrase_common_in_human_holdout_is_flagged` (Task 1).
2. **Temat jednego scenariusza**, np. „danych osobowych” z polityk RODO. Odpada przez udział dokumentów. Test: `test_topic_term_in_few_documents_is_dropped` (Task 1).
3. **Wpis zaakceptowany dla `legal` wyskakuje w e-mailu `general`**. Matcher musi filtrować po gatunku. Test: `test_accepted_entry_fires_only_in_its_genre` (Task 3).
4. **Dwa liczenia jednego trafienia**, gdy fraza pokrywa się z istniejącym `discourse_frame`. Test: `test_slop_hit_inside_existing_finding_is_not_doubled` (Task 3).
5. **Ponowny pomiar kasuje decyzje recenzenta**. Scalanie musi zachować `status` i `note`. Test: `test_merge_keeps_reviewer_decisions` (Task 2).

---

### Task 1: Wybór kandydatów z filtrami tematu

**Pliki:**
- Utwórz: `humanize_pl/corpus/fingerprint.py`
- Test: `tests/test_fingerprint.py`

**Interfejsy:**
- Korzysta z: `count_terms`, `log_odds_with_prior`, `Z_THRESHOLD`, `tokenize`, `ngrams`, `sentence_openings` z `humanize_pl.corpus.logodds` oraz `extract_deontic_profile` z `humanize_pl.safety.deontic`.
- Dostarcza:
  ```python
  KINDS = {"unigram": (1, False), "bigram": (2, False), "trigram": (3, False),
           "opening_bigram": (2, True), "opening_trigram": (3, True)}

  @dataclass(frozen=True)
  class SlopCandidate:
      term: str; kind: str; genre: str
      z_score: float; ai_per_1000: float; human_per_1000: float
      holdout_per_1000: float | None
      ai_doc_share: float
      models: tuple[str, ...]
      protected: bool
      indicative: bool

  def scenario_half(file_name: str) -> str            # "train" | "holdout"
  def is_protected(term: str) -> bool
  def slop_candidates(ai_by_model: dict[str, list[str]], human: list[str], *, genre: str,
                      human_holdout: list[str] | None = None,
                      min_count: int = 5) -> list[SlopCandidate]
  ```

- [ ] **Krok 1: Napisz testy, które na razie nie przechodzą.** Korpusy syntetyczne, po 20 dokumentów na model, zbudowane w helperze testu.
  - `test_shared_style_phrase_is_a_candidate`: „warto podkreślić” w 60% dokumentów obu modeli, nieobecna u ludzi → jest w wyniku z `models == ("a", "b")`.
  - `test_term_used_by_one_model_only_is_not_a_candidate`: fraza tylko u modelu „a” → brak w wyniku.
  - `test_topic_term_in_few_documents_is_dropped`: „danych osobowych” gęsto w 1 z 40 dokumentów → brak w wyniku.
  - `test_phrase_common_in_human_holdout_is_flagged`: fraza rzadka w `human`, a częsta w `human_holdout` → brak w wyniku, bo stosunek AI do odłożonego < `MIN_HOLDOUT_RATIO`.
  - `test_deontic_and_reference_terms_are_protected`: `is_protected("strona powinna")`, `is_protected("ust lit")` i `is_protected("art kc")` dają True, a `is_protected("warto podkreślić")` daje False.
  - `test_single_model_corpus_is_indicative`: `ai_by_model` z jednym kluczem → kandydaci mają `indicative=True` i wystarcza im 1 model.
  - `test_scenario_half_is_stable_across_models`: `scenario_half("opus/draft_opinia_prawna.txt") == scenario_half("haiku/draft_opinia_prawna.txt")`, a wśród 40 nazw `draft_{i}` są obie połowy.

- [ ] **Krok 2:** `.venv/bin/pytest tests/test_fingerprint.py -q`. Oczekiwane: FAIL (ImportError).

- [ ] **Krok 3: Zaimplementuj.**
  - Log-odds liczony na całości AI wobec `human` daje ranking i `z_score`.
  - Model liczy się do `models`, gdy jego własny log-odds wobec `human` > 0 i ma co najmniej 2 wystąpienia.
  - `ai_doc_share` to odsetek dokumentów AI zawierających termin: `count_terms` na każdym dokumencie osobno.
  - Holdout: odrzuć, gdy `holdout_per_1000 > 0` i `ai_per_1000 / holdout_per_1000 < MIN_HOLDOUT_RATIO`.
  - `scenario_half`: `zlib.crc32` nazwy pliku bez katalogu modelu, parzysta suma → `"train"`.
  - `is_protected`: niepusty `extract_deontic_profile(term)` albo token z `REFERENCE_TOKENS`.
  - Wynik posortowany malejąco po `z_score`.

- [ ] **Krok 4:** `.venv/bin/pytest tests/test_fingerprint.py -q && .venv/bin/ruff check humanize_pl/corpus tests/test_fingerprint.py`. Oczekiwane: PASS, brak uwag.

- [ ] **Krok 5:** commit `corpus: pick slop candidates across models, documents and a human holdout`.

### Task 2: Plik odcisku, loader i scalanie

**Pliki:**
- Utwórz: `humanize_pl/data/slop_fingerprint.yaml` (pusty szkielet: `version: 1`, `measurements: []`, `entries: []`)
- Utwórz: `humanize_pl/detect/fingerprint.py`
- Zmień: `humanize_pl/corpus/fingerprint.py` (dodaj `merge_candidates`)
- Test: `tests/test_fingerprint.py`

**Interfejsy:**
- Korzysta z: `SlopCandidate` z Zadania 1.
- Dostarcza:
  ```python
  # humanize_pl/detect/fingerprint.py
  DEFAULT_PATH: Path  # humanize_pl/data/slop_fingerprint.yaml
  STATUSES = ("candidate", "accepted", "rejected")

  @dataclass(frozen=True)
  class SlopEntry:
      term: str; kind: str; genre: str; status: str; note: str
      protected: bool; indicative: bool

  def load_fingerprint(path: Path = DEFAULT_PATH) -> list[SlopEntry]   # ValueError z polskim komunikatem
  def accepted_patterns(genre: str) -> list[tuple[SlopEntry, re.Pattern]]  # lru_cache

  # humanize_pl/corpus/fingerprint.py
  def merge_candidates(document: dict, fresh: list[SlopCandidate], *, measurement: dict) -> dict
  ```
- Kształt wpisu w YAML: `term, kind, genre, status, note, protected, indicative, z_score, ai_per_1000, human_per_1000, holdout_per_1000, ai_doc_share, models`. Kluczem jest trójka `(term, kind, genre)`. Pozycja `measurement` zapisuje datę, modele, liczności, użyty podział i progi.

- [ ] **Krok 1: Napisz testy, które na razie nie przechodzą.**
  - `test_merge_keeps_reviewer_decisions`: wpis `accepted` z notatką przechodzi przez ponowny pomiar i zachowuje `status` oraz `note`, a liczby się odświeżają.
  - `test_merge_adds_new_terms_as_candidates`.
  - `test_merge_keeps_entries_missing_from_new_measurement`: wpis zostaje bez zmian, a w `measurements[-1]["not_remeasured"]` pojawia się jego termin.
  - `test_loader_rejects_accepted_protected_entry`: `ValueError`, komunikat zawiera „chroniony”.
  - `test_loader_rejects_rejected_without_note`.
  - `test_opening_pattern_matches_only_sentence_start`: `accepted_patterns` dla `opening_bigram` „warto podkreślić” trafia w „Warto podkreślić, że…”, a nie w „…i warto podkreślić…”.
  - `test_ngram_pattern_respects_word_boundaries`: „w kontekście” nie trafia w „w kontekścieX”.
  - `test_shipped_fingerprint_loads`: `load_fingerprint()` działa na pliku z repo.

- [ ] **Krok 2:** `.venv/bin/pytest tests/test_fingerprint.py -q`. Oczekiwane: FAIL dla nowych testów.

- [ ] **Krok 3: Zaimplementuj.**
  - Wzorzec n-gramu: tokeny złączone `\s+`, z `(?<!\p{L})` i `(?!\p{L})`, flaga IGNORECASE.
  - Wzorzec otwarcia: na początku `^[\W\d_]*`.
  - Dopasowanie po formach powierzchniowych. Morfologię celowo pomijamy: lista mierzy formy, które modele faktycznie piszą.

- [ ] **Krok 4:** `.venv/bin/pytest tests/test_fingerprint.py -q`. Oczekiwane: PASS.

- [ ] **Krok 5:** commit `detect: load the reviewed slop fingerprint and merge new measurements`.

### Task 3: Sygnał `slop_ngram` w diagnozie, zależny od gatunku

**Pliki:**
- Zmień: `humanize_pl/detect/signals.py` (`sentence_findings` i nowe `_slop_ngrams`)
- Zmień: `humanize_pl/detect/engine.py` (`detect_document`)
- Zmień: `humanize_pl/flows/base.py:471,557,602` (przekazać gatunek)
- Test: `tests/test_detection.py`

**Interfejsy:**
- Korzysta z: `accepted_patterns(genre)` z Zadania 2.
- Dostarcza:
  - `sentence_findings(sentence, *, paragraph_index, sentence_index, genre: str = "legal")`
  - `detect_document(text, *, profile=None, calibrate_against_default=True, genre: str = "legal")`
  - Finding ma `family="slop_ngram"`, `rule=f"detect:slop.{kind}.{term.replace(' ', '_')}"`, `weight=0.4` i `rewritable=False`.

- [ ] **Krok 1: Napisz testy, które na razie nie przechodzą.** Każdy podmienia `accepted_patterns` przez `monkeypatch`, więc nie zależy od zawartości pliku z repo.
  - `test_accepted_entry_fires_only_in_its_genre`: wpis `general` „mam nadzieję że” trafia przy `genre="general"`, a przy `"legal"` nie.
  - `test_slop_hit_inside_existing_finding_is_not_doubled`: zdanie „Warto podkreślić, że umowa wiąże.” przy wpisie „warto podkreślić” ma jedno znalezisko `discourse_frame` i zero `slop_ngram`.
  - `test_slop_finding_is_detect_only`: `rewritable is False`.
  - `test_general_flow_diagnoses_with_general_genre`: `run_all_layers` z `DocumentType.general` widzi wpis `general`.

- [ ] **Krok 2:** `.venv/bin/pytest tests/test_detection.py -q`. Oczekiwane: FAIL.

- [ ] **Krok 3: Zaimplementuj.**
  - `_slop_ngrams` liczony na końcu `sentence_findings`. Pomija trafienia, których zakres nakłada się na znalezisko innej rodziny w tym samym zdaniu.
  - W `flows/base.py`: `genre = "general" if settings.document_type == DocumentType.general else "legal"`.
  - Pozostali wołający `detect_document` (cli, benchmark, evaluation, docx_io, replay, gate, core, profile) zostają przy domyślnym `"legal"`.

- [ ] **Krok 4:** `.venv/bin/pytest -q && .venv/bin/ruff check .`. Oczekiwane: wszystko PASS (dziś 629 passed).

- [ ] **Krok 5:** commit `detect: report reviewed slop n-grams per genre as detect-only signals`.

### Task 4: Narzędzie pomiaru i pierwszy pomiar

**Pliki:**
- Przepisz: `tools/derive_patterns.py` (cienka warstwa CLI nad `slop_candidates` i `merge_candidates`; stary zapis `--out` JSON znika)
- Zmień: `humanize_pl/data/slop_fingerprint.yaml` (kandydaci)
- Zmień: `README.md:885-904` (sekcja o derive_patterns, po polsku)

**Interfejsy:**
- Korzysta z: `slop_candidates`, `scenario_half` i `merge_candidates` (Task 1–2) oraz `load_texts` z obecnego narzędzia, które zostaje.
- CLI:
  ```bash
  .venv/bin/python tools/derive_patterns.py --genre legal \
    --ai haiku=docs_tests/corpus/ai_models/haiku --ai opus=docs_tests/corpus/ai_models/opus \
    --ai sonnet=docs_tests/corpus/ai_models/sonnet --split train \
    --human docs_tests/corpus/saos_train.jsonl --human-holdout docs_tests/corpus/saos_holdout.jsonl \
    --update humanize_pl/data/slop_fingerprint.yaml
  .venv/bin/python tools/derive_patterns.py --genre general \
    --ai haiku=docs_tests/corpus/general/haiku --split all \
    --human docs_tests/corpus/general/wiki_talk_train.jsonl \
    --human docs_tests/corpus/general/allegro_reviews_train.jsonl \
    --human-holdout docs_tests/corpus/general/wiki_talk.jsonl \
    --human-holdout docs_tests/corpus/general/allegro_reviews.jsonl \
    --update humanize_pl/data/slop_fingerprint.yaml
  ```
  Flagi `--ai` i `--human*` są powtarzalne. `--split train` bierze tylko szkice z połowy `train`. Ostrzeżenia `UWAGA` po polsku pojawiają się przy < 25 dokumentach i przy jednym modelu.

- [ ] **Krok 1:** przepisz narzędzie. Usuń z README stary przykład `--out derived_patterns.json` i opisz nowy obieg: pomiar → przegląd → `accepted`.
- [ ] **Krok 2:** uruchom oba polecenia. Oczekiwane: w YAML są kandydaci obu gatunków, `measurements` ma 2 pozycje, a `load_fingerprint()` przechodzi.
- [ ] **Krok 3:** `.venv/bin/pytest -q && .venv/bin/ruff check .`. Oczekiwane: PASS.
- [ ] **Krok 4:** commit `tools: measure the slop fingerprint into the shipped YAML` (sami kandydaci, jeszcze bez decyzji).

### Task 5: Przegląd kandydatów (decyzja użytkownika)

**Pliki:** `humanize_pl/data/slop_fingerprint.yaml`

- [ ] **Krok 1:** Claude proponuje `status` i `note` dla każdego kandydata. Dla każdej propozycji podaje 2 przykładowe zdania AI i liczbę trafień w `saos_holdout`. Reguły propozycji:
  - Odrzucić frazę normalną w gatunku, nawet przy wysokim z.
  - Odrzucić ślad szablonu promptu, np. „jeśli dotyczy”, „dnia data”.
  - Przyjąć tylko formuły stylu, nie terminy.
- [ ] **Krok 2:** **użytkownik zatwierdza albo poprawia statusy.** Bez tego wpisy zostają `candidate` i nie działają.
- [ ] **Krok 3:** commit `detect: accept the reviewed slop fingerprint`.

### Task 6: Pomiar skutku i bramki

Bez nowych plików. Pomiary w `python -c` albo w scratchpadzie. Stary kod do porównania: `git worktree` w scratchpadzie z `PYTHONPATH=<worktree>`.

- [ ] **Krok 1: AUC na połowie odłożonej.** Policz AUC `ai_signal_score` dla szkiców z połowy `holdout` wobec ludzkich szablonów (te same reguły co w `evaluation.detector_metrics`), przed zmianą i po niej. Jest to metryka decyzyjna. Cel: wzrost względem 0,5748. Spadek oznacza, że wpisy wracają do przeglądu.
- [ ] **Krok 2: Fałszywe trafienia na ludzkim tekście prawnym.** Policz trafienia `slop_ngram` per termin na każdym `text`/`document` z `humanize_pl/data/blueprints/*.yaml` i `data/templates/*/*.yaml`. Termin, który trafia w więcej niż 3 szablony, wraca do przeglądu.
- [ ] **Krok 3: Profil referencyjny.**
  ```bash
  .venv/bin/python tools/build_reference_profile.py --corpus docs_tests/corpus/saos_train.jsonl --name saos_train_1804 --genre court_reasoning
  ```
  Uruchom z tymi samymi argumentami, którymi powstał obecny `saos_train_1804.json` (sprawdzić w jego polach). Cel: odświeżyć `signal_score` p50 i p95 ludzi. Diff ma dotyczyć tylko pól zależnych od nowej rodziny.
- [ ] **Krok 4: Bramki wydania.** `.venv/bin/humanize-pl-benchmark --engines basic --mode standard --allow-fallback --fail-on-status` kończy się kodem 0 (`style_compliance` reaguje na wzrost sygnału o > 0,02). Następnie `.venv/bin/humanize-pl-eval`. Jeśli zmienia się `detector_auc_drafts`, nową wartość nagrywa się w `eval_baseline.json` dopiero za zgodą użytkownika, z dopiskiem, że zawiera połowę in-sample.
- [ ] **Krok 5:** `.venv/bin/pytest -q && .venv/bin/ruff check .`, a potem przegląd martwego kodu: `grep -rn "derived_patterns\|--out" tools README.md`.
- [ ] **Krok 6:** commit `reference_profiles: re-measure human scores with the slop family`.

## Poza zakresem tego planu

- **Korpus AI dla `general`:** tylko 24 teksty jednego modelu, więc wpisy `general` są `indicative`. Dogenerowanie szkiców innymi modelami wymaga wywołań LLM i decyzji użytkownika.
- **Reguły przepisujące dla przyjętych fraz:** dochodzą osobno, regułą na znalezisko, z walidatorami. Ten plan dotyczy tylko diagnozy.
- **Druga pozycja z raportu** (jawna baza gatunkowa i lista metryk wykluczonych) oraz **trzecia** (rejestr użycia AI): osobne plany.
