# humanize-pl

Deterministyczny silnik kontroli i redakcji AI-generowanych polskich tekstów
prawniczych.

Projekt **nie używa `texthumanize`** i nie używa generatywnego LLM. Nie jest
parafrazerem ani narzędziem do obchodzenia detektorów AI. Jego celem jest
bezpieczne przepisanie roboczego tekstu AI na precyzyjny język prawny:
umowy, opinie, analizy, pisma, regulaminy i podobne dokumenty prawnicze.

Działa warstwowo:

1. ochrona fragmentów wrażliwych,
2. segmentacja zdań,
3. analiza cech prawnych i artefaktów AI,
4. generowanie kandydatów przez reguły polskie,
5. walidacja bezpieczeństwa i normatywności,
6. opcjonalna analiza Stanza,
7. opcjonalny filtr semantyczny sentence-transformers,
8. zapis DOCX/TXT + raport JSON.

## Co nowego (niewydane)

- wykrywanie predykacji (`has_finite_verb`) oparte na morfologii Morfeusz2/SGJP
  zamiast zamkniętej listy ~90 czasowników; poprzednia heurystyka odrzucała 32%
  zdań realnej prozy prawniczej i po cichu blokowała generowanie kandydatów,
- nowa warstwa `humanize_pl/detect/` — diagnoza sygnałów AI niezależna od trybu
  i od tego, czy istnieje reguła przepisująca,
- sekcja `detection` w raporcie JSON, widoczna także przy zerze zmian,
- flaga `--detect-only` dla tekstu, pliku DOCX i folderu,
- powtórzone otwarcia zdań liczone w skali dokumentu, nie akapitu.

## Co nowego w 0.2.2

- wyłączone ryzykowne dzielenie zdań po `oraz`,
- dodany walidator: nowo utworzone zdanie musi mieć czasownik osobowy,
- dodany walidator blokujący konstrukcje typu `działać, które` po zbyt agresywnej nominalizacji,
- dodane reguły stylu formalnego/prawniczego,
- dodane wykrywanie monotonnych otwarć i ram AI, np. powtarzanego `Warto wskazać` oraz `ma istotne znaczenie`,
- raport JSON pokazuje zaakceptowane zmiany i odrzucone kandydaty,
- raport JSON pokazuje metryki monotonii akapitów w sekcji `quality.paragraph_monotony`,
- benchmark zapisuje czas przetwarzania, sygnały bezpieczeństwa i blokady bramek w `review.md`, `summary.csv` i `summary.json`,
- domyślny profil `legal_ai_review` dla AI-generowanych tekstów prawniczych,
- metryki `legal_review` w raporcie JSON,
- przetwarzanie DOCX używa jednej sesji humanizatora na dokument, więc modele NLP nie są ładowane osobno dla każdego akapitu,
- usunięte reguły, które generowały niepoprawne konstrukcje typu `Nie oznacza to jednak, że nie występuje podporządkowania`,
- test regresji dla błędu `Ponadto za wynagrodzeniem`,
- wyniki benchmarków pod `docs_tests/results/` są traktowane jako artefakty lokalne i ignorowane przez Git.

## Detekcja niezależna od przepisywania

Silnik rozdziela dwie rzeczy, które wcześniej były sklejone: **wykrycie**
sygnału AI i **bezpieczne przepisanie** go. Dokument bez ani jednej
zaakceptowanej zmiany nadal dostaje pełną diagnozę — „nic nie przepisano" i
„nic nie znaleziono" to różne wyniki.

Sama diagnoza, bez zapisu dokumentu:

```bash
humanize-pl input.docx --detect-only --report detection.json
```

Folder dokumentów:

```bash
humanize-pl docs/ --detect-only --report detection.json
```

Detekcja działa w każdym trybie, także w `conservative`, gdzie żadna reguła
przepisująca nie jest uruchamiana. Sekcja `detection` w raporcie JSON zawiera:

- `ai_signal_score` — nasycająca się ważona gęstość sygnałów (0–1),
- `score_is_calibrated` — dziś zawsze `false`; próg wymaga ludzkiego korpusu
  referencyjnego polskich tekstów prawniczych,
- `families` — rodziny sygnałów z gęstością na 1000 słów,
- `findings` — pojedyncze znaleziska ze spanami znakowymi i dowodem,
- `findings_rewritable` / `findings_detect_only` — ile znalezisk ma za sobą
  regułę przepisującą, a ile silnik potrafi tylko wskazać.

Rozróżnienie `rewritable` jest celowe: znalezisko bez reguły przepisującej ma
być widoczne, a nie przemilczane.

Wykrywanie powtórzonych otwarć zdań liczone jest w skali **dokumentu**, nie
akapitu — teksty AI różnicują otwarcia wewnątrz akapitu, powtarzając te same
ramy w całym dokumencie.

## Instalacja

Minimalnie:

```bash
python -m pip install -e .
```

Z NLP:

```bash
python -m pip install -e ".[nlp]"
python -m humanize_pl.download_models --stanza
```

Z walidacją transformerową:

```bash
python -m pip install -e ".[nlp,transformers]"
python -m humanize_pl.download_models --stanza --transformers --fluency
```

Pełny lokalny zestaw NLP, razem z warstwą Morfeusz2, jeżeli system ma dostępne
natywne zależności Morfeusza:

```bash
python -m pip install -e ".[nlp,transformers,morfeusz]"
python -m humanize_pl.download_models --stanza --transformers --fluency --morfeusz
```

Dla pracy developerskiej:

```bash
python -m pip install -e ".[dev]"
```

## Użycie

DOCX:

```bash
humanize-pl input.docx -o output.docx --mode conservative --engine basic --report report.json
```

Tryb standard:

```bash
humanize-pl input.docx -o output.docx --mode standard --engine basic --report report.json
```

Folder dokumentów DOCX:

```bash
humanize-pl docs/ -o output/ --mode standard --engine nlp --report batch-report.json
```

Tryb folderowy przetwarza wszystkie pliki `.docx` bezpośrednio w podanym
folderze (bez przeszukiwania podfolderów). Pliki tymczasowe Worda zaczynające
się od `~$` są pomijane. Wyniki otrzymują przyrostek `_humanized.docx`.
Jeżeli `-o` nie zostanie podane, powstanie folder obok wejściowego, np.
`docs_humanized/`.

Dla folderu `--report batch-report.json` zapisuje raport zbiorczy z sumami i
krótkim podsumowaniem każdego dokumentu. Pełne raporty dokumentów trafiają do
folderu `batch-report_details/`. Błąd jednego dokumentu nie zatrzymuje
pozostałych; zostaje zapisany w raporcie zbiorczym, a komenda kończy się kodem
`1`, jeżeli co najmniej jeden plik nie został przetworzony.

Profil prawny AI jest domyślny, ale można go jawnie wskazać:

```bash
humanize-pl input.docx -o output.docx --mode standard --legal-review-profile legal_ai_review
```

Z analizą Stanza:

```bash
humanize-pl input.docx -o output.docx --engine nlp --mode standard --report report.json
```

Z filtrem semantycznym sentence-transformers i scorerem płynności:

```bash
humanize-pl input.docx -o output.docx --engine hybrid --mode standard --report report.json
```

Własne modele i twardy wymóg ich dostępności:

```bash
humanize-pl input.docx -o output.docx --engine hybrid --mode standard \
  --semantic-model sdadas/st-polish-paraphrase-from-distilroberta \
  --fluency-model allegro/herbert-base-cased \
  --require-models \
  --offline-models
```

Tekst:

```bash
humanize-pl "Podsumowując źródła prawa pracy tworzą system."
```

Wersja:

```bash
humanize-pl --version
```

## Tryby

- `conservative` — dla prawa i dokumentów formalnych; mało zmian.
- `standard` — więcej zmian stylistycznych, nadal bezpieczne walidatory.
- `strong` — eksperymentalnie, niezalecane dla prawa.

## Silniki

- `basic` — reguły + walidatory, bez modeli.
- `nlp` — reguły + opcjonalna Stanza dla składni, lematów i zależności.
- `hybrid` — reguły + Stanza + sentence-transformer jako walidator semantyczny + masked-LM jako scorer płynności.

Jeżeli model nie jest dostępny lokalnie, silnik domyślnie zapisze ostrzeżenie i
wróci do dostępnych warstw. Flaga `--require-models` zmienia to w błąd, a
`--offline-models` wymusza ładowanie wyłącznie z lokalnego cache bez prób
odświeżania zasobów Stanza/HuggingFace.

## Benchmark i bramki wydania

Podstawowa bramka bezpieczeństwa bez modeli zewnętrznych:

```bash
make benchmark-basic
```

Równoważna komenda:

```bash
humanize-pl-benchmark --engines basic --mode standard --allow-fallback --fail-on-status
```

Komenda zapisuje artefakty pod `docs_tests/results/latest/` i kończy się kodem
`1`, jeżeli którykolwiek dokument ma status inny niż `ok`.

Pełniejsza, ręczna walidacja silników opcjonalnych wymaga lokalnie pobranych
modeli:

```bash
python -m humanize_pl.download_models --stanza --transformers --fluency --morfeusz
make benchmark-optional
```

`benchmark-optional` uruchamia `nlp` i `hybrid` z `--offline-models`,
`--require-models` oraz `--fail-on-status`, więc nadaje się jako lokalna
checklista przed wydaniem, ale nie zakłada dostępu do sieci w trakcie testu.

Pełna lokalna kontrola przed wydaniem:

```bash
humanize-pl-release-check
```

Obejmuje testy, lint, podstawowy benchmark i budowę wheel w trybie
`--no-isolation`, czyli z zależnościami zainstalowanymi przez `.[dev]`.
Jeżeli w systemie jest `make`, równoważnym skrótem jest `make release-check`.

## Ważne

Silnik nie jest generatywnym parafrazerem. To kontrolowany edytor formalnej polszczyzny: lepiej odrzucić zmianę niż wygenerować nienaturalne lub nieprecyzyjne zdanie.

Walidatory blokują m.in. zmianę normatywności (`może`/`musi`/`powinien`),
utratę stron, świadczeń, kwot, dat, cytatów i podstaw prawnych, zdania bez
orzeczenia, osierocone zdania względne po bezokoliczniku, angielskie wstawki
oraz wycieki placeholderów ochronnych.
