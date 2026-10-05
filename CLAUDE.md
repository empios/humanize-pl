# humanize-pl

Polish legal-text QA and redaction engine. Rule-based, with an optional hosted
LLM (OpenAI-compatible) for leftover issues. Not an AI-detector evasion tool:
the goal is precise legal Polish, with legally sensitive content preserved.
README.md is in Polish and holds full feature docs.

## Commands
```bash
python -m pip install -e ".[dev]"   # setup (extras: nlp, transformers, morfeusz, xlsx, pdf, ui, all)
pytest -q                           # tests
ruff check .                        # lint (line-length 100, py310 target)
make release-check                  # test + lint + benchmark-basic + eval + build
```
CLI entry points: `humanize-pl`, `humanize-pl-flow` (DOCX/XLSX batches), `humanize-pl-benchmark`, `humanize-pl-eval` (quality vs `docs_tests/eval_baseline.json`),
`humanize-pl-release-check`, `humanize-pl-ui` (Gradio).

## Layout
- `humanize_pl/pipeline.py`, `core.py`: layered pipeline (protect → segment → features → rule candidates → safety validation → optional Stanza/ST/LLM → legal-sensitivity check)
- `rules/`: Polish rewrite rules (YAML tables + Python). `safety/`: protectors, anchors, deontic, agreement checks
- `detect/`: AI-signal diagnosis, independent of rewrite mode. Calibrated on the SAOS human corpus
- `flows/`: DOCX/XLSX batch flows (resume via details JSON). `io/`, `reports/`: DOCX I/O, XLSX/PDF/JSON reports
- `blueprint*.py`, `categories.py`, `data/categories.yaml`: document categories and structure checks. `data/blueprints/<category>.yaml`: general blueprint (section checklist + the category's general fill-in `document`); `data/templates/<category>/*.yaml`: precise fill-in templates. `layout_ir.py`: template IR (extract/fill/evaluate). Check runs general first, then the matched template, which decides. Filling templates into DOCX lives in `document-studio/` (its only consumer), not here
- Source DOCX of shipped templates are protected: never read, ship or hardcode their path; work from the YAML
- `tools/`: corpus tooling. `docs_tests/`: sample/corpus docs. `document-studio/`: separate sub-project with its own README and pyproject
- `outputs/`: generated, don't hand-edit

## Rules
- Legal safety first: never alter deontic modality (must/may/shall), numbers, dates, references or defined terms. Safety validators must reject unsafe candidates.
- DOCX edits preserve runs, hyperlinks, bookmarks and mixed formatting. Fields, comments, tracked changes, footnotes and content controls are protected and reported as warnings.
- Predicate detection uses Morfeusz2/SGJP morphology, not word lists. Don't reintroduce closed verb lists.
- Metrics must be computed before and after redaction (`*_before` / `*_after`). Don't drop metrics distorted by genre; list them as excluded.
- LLM config comes from `.env` (see `.env.example`). Never commit `.env` or keys.
- Keep user-facing text, reports and docs in Polish. Code and identifiers in English.
- Commit style: `area: summary` (e.g. `blueprints: ...`, `flows: ...`).

## Working agreement
- Stay inside this project's scope. Don't touch files outside the repo, and don't refactor or add features unless asked.
- Don't create temp files (scratch scripts, debug dumps, `*.tmp`, ad-hoc outputs). Use inline commands or pytest. If a file is truly needed, delete it before finishing.
- After every change, hunt for dead code and fix it:
  - Remove unused imports, functions, params, constants and YAML keys that the change orphaned.
  - Grep for remaining references before deleting anything.
  - Run `ruff check .` and `pytest -q`. Both must pass before you call the work done.
- Keep quality up. Match surrounding style, no speculative abstractions, no commented-out code.
