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

## Użycie

DOCX:

```bash
humanize-pl input.docx -o output.docx --mode conservative --engine basic --report report.json
```

Tryb standard:

```bash
humanize-pl input.docx -o output.docx --mode standard --engine basic --report report.json
```

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

## Ważne

Silnik nie jest generatywnym parafrazerem. To kontrolowany edytor formalnej polszczyzny: lepiej odrzucić zmianę niż wygenerować nienaturalne lub nieprecyzyjne zdanie.

Walidatory blokują m.in. zmianę normatywności (`może`/`musi`/`powinien`),
utratę stron, świadczeń, kwot, dat, cytatów i podstaw prawnych, zdania bez
orzeczenia, osierocone zdania względne po bezokoliczniku, angielskie wstawki
oraz wycieki placeholderów ochronnych.
