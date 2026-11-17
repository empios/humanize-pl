# Rejestr użycia modelu językowego w raportach — plan implementacji

> **Dla agentów:** WYMAGANY SUB-SKILL: superpowers:executing-plans. Kroki w formacie `- [ ]`.

**Cel:** każdy raport mówi wprost, które zmiany zrobił model językowy, które reguły, ile propozycji modelu odrzuciły lokalne kontrole i że zatwierdzenie należy do osoby odpowiedzialnej za dokument.

**Architektura:** dane już istnieją: `ItemOutcome.llm` (model, liczniki) oraz zmiany z regułą `llm_rewrite` w `applied_changes`, która przeżywa `collapse_visible_changes`. Właściwość `ItemOutcome.ai_use` zestawia je dla jednej pozycji i trafia do `to_json`. `summarise` agreguje je w `summary["ai_use"]`. PDF dostaje sekcję „Udział modelu językowego”.

**Spec:** raport „Kierunki rozwoju humanize pl 2026”, mapa drogowa, pozycja „Teraz” nr 3 (rejestr użycia AI, komunikacja „redaktor zachowujący sens”, art. 50 AI Act).

## Ograniczenia globalne

- Raport nie formułuje twierdzeń prawnych o AI Act. Opisuje fakty przebiegu.
- Tekst dla użytkownika po polsku, bez długich myślników jako wtrąceń (pilnuje tego `test_the_report_avoids_the_phrasing_it_flags`).
- Stare raporty bez `ai_use` renderują się dalej i mówią, że rejestru w nich nie ma.
- Zmiana zaczęta regułą i dokończona przez model liczy się jako zmiana modelu.

## Na co patrzeć w przeglądzie

1. Przebieg bez modelu (`rewrite_backend=rules`) nie może sugerować, że model był użyty. Test: `test_rules_only_run_records_no_model`.
2. Model zamówiony, ale niedostępny (`rules_fallback`), jest liczony osobno. Test: `test_summary_aggregates_model_use`.
3. Raport sprzed rejestru. Test: `test_pdf_old_report_has_no_register`.

---

### Task 1: `ItemOutcome.ai_use` i `summary["ai_use"]`

**Pliki:** `humanize_pl/flows/base.py`, test `tests/test_ai_use.py`

**Interfejsy:**
- `ItemOutcome.ai_use -> dict`: `{"model": str | None, "model_segments": int, "rule_segments": int, "model_rejections": int}`. W `to_json` pod kluczem `ai_use`.
- `summarise(...)["ai_use"]`: `{"models": list[str], "documents_with_model_edits": int, "model_segments": int, "rule_segments": int, "model_rejections": int, "model_unavailable": int}`.

- [ ] Testy: `test_item_records_model_and_rule_segments`, `test_a_rule_edit_finished_by_the_model_counts_as_model`, `test_rules_only_run_records_no_model`, `test_summary_aggregates_model_use`, `test_ai_use_reaches_the_json_row`. RED.
- [ ] Implementacja, GREEN, `pytest -q`, `ruff check .`, commit `flows: record what the language model changed per item and per run`.

### Task 2: sekcja PDF „Udział modelu językowego”

**Pliki:** `humanize_pl/reports/pdf_pl.py`, test `tests/test_pdf_report.py`

**Interfejsy:** `_Report.ai_use_rows() -> list[tuple[str, str]]`, `_Report.ai_use_sentence() -> str`, sekcja `_Report.ai_use` w `story()` przed `caveats`.

- [ ] Testy: `test_pdf_names_what_the_model_changed`, `test_pdf_says_when_no_model_was_used`, `test_pdf_old_report_has_no_register`. RED.
- [ ] Implementacja, GREEN, pełny zestaw, commit `reports: show the language model's share of the changes`.
