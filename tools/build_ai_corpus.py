"""Generate the AI side of the corpus from tools/corpus_prompts.yaml.

The prompt grid has been in the repository since the corpus was first
described, and nothing read it: `{zadanie}` was never filled in by any code,
so the documents it was supposed to produce were written by hand instead.
Eight of the fifteen shipped fixtures carry `generated_by: unknown`, and the
folder's own README says they were written to contain the patterns the
detector looks for. Measuring a detector against text authored to be caught
is circular, and it inflated the reported separation: those eight score
0.318-0.682 while real model output scores 0.192-0.335.

This closes that loop. Every document here comes from the endpoint, every
manifest entry records which model wrote it, under which register and at
which temperature, and the grid is walked deterministically so two runs with
the same arguments ask for the same things.

Usage:
    python tools/build_ai_corpus.py --per-category 4 --dry-run
    python tools/build_ai_corpus.py --per-category 4
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from humanize_pl.llm import (
    LlmConfigurationError,
    LlmEndpointError,
    LlmSettings,
    OpenAICompatibleRewriter,
    _safe_error,
)

DEFAULT_PROMPTS = Path("tools/corpus_prompts.yaml")
DEFAULT_OUT = Path("docs_tests/ai_generated")

# Room for a full contract or statement of claim, plus whatever the model
# spends thinking before it starts one.
#
# 4000 was not enough and failed in a way that looked like the model refusing.
# A reasoning model emits its trace first: measured on this endpoint, a
# 4000-token budget went entirely on ~12k characters of `reasoning_content`,
# came back with `finish_reason: length` and an empty answer, on four of eight
# categories. The four that survived were the ones it happened to think about
# briefly - so the budget was silently selecting which documents entered the
# corpus, which is the provenance artifact this whole tool exists to remove.
MAX_TOKENS = 16000

# A document shorter than this is a refusal, a preamble or a truncation, not
# a legal document. Same floor the human corpus uses in
# humanize_pl/corpus/normalize.py, so both sides are filtered alike.
MIN_WORDS = 150


def load_grid(path: Path) -> tuple[list[dict[str, str]], list[float], dict[str, list[str]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    styles = payload.get("styles") or []
    temperatures = payload.get("temperatures") or [0.7]
    categories = payload.get("categories") or {}
    if not styles or not categories:
        raise ValueError(f"{path}: brak `styles` albo `categories`.")
    return styles, [float(value) for value in temperatures], categories


def plan(
    categories: dict[str, list[str]],
    styles: list[dict[str, str]],
    temperatures: list[float],
    *,
    per_category: int,
) -> list[dict[str, Any]]:
    """Walk the grid so a category's documents differ along every axis.

    Taking the first N scenarios and pairing them all with the first style
    would vary only the subject; stepping style and temperature alongside the
    scenario means four documents of one category use four scenarios, and
    rotate through the registers and sampling settings as they go.
    """
    jobs: list[dict[str, Any]] = []
    for category, scenarios in sorted(categories.items()):
        if not scenarios:
            continue
        for index in range(per_category):
            style = styles[index % len(styles)]
            jobs.append(
                {
                    "category": category,
                    "scenario": scenarios[index % len(scenarios)],
                    "style": style,
                    "temperature": temperatures[index % len(temperatures)],
                    "ordinal": index + 1,
                }
            )
    return jobs


def messages_for(job: dict[str, Any]) -> list[dict[str, str]]:
    style = job["style"]
    return [
        {"role": "system", "content": style["system"]},
        {"role": "user", "content": style["user"].format(zadanie=job["scenario"])},
    ]


def next_index(out_dir: Path) -> int:
    """Continue the existing numbering instead of overwriting it.

    The hand-written fixtures stay: six test modules and the benchmark read
    them by name, and they remain valid regression material. What changes is
    which documents feed calibration and evaluation, not which files exist.
    """
    highest = 0
    for path in out_dir.glob("ai_legal_*.txt"):
        head = path.stem.split("_")[2:3]
        if head and head[0].isdigit():
            highest = max(highest, int(head[0]))
    return highest + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--per-category",
        type=int,
        default=4,
        help="Ile dokumentów na kategorię (8 kategorii × N)",
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wypisz, o co runner zapyta, i nie wywołuj modelu",
    )
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)

    styles, temperatures, categories = load_grid(args.prompts)
    jobs = plan(categories, styles, temperatures, per_category=args.per_category)

    print(f"Siatka: {len(categories)} kategorii × {args.per_category} = {len(jobs)} dokumentów")
    print(f"Rejestry: {[s['id'] for s in styles]}   temperatury: {temperatures}")

    if args.dry_run:
        for job in jobs:
            print(
                f"  [{job['category']}/{job['ordinal']}] "
                f"styl={job['style']['id']} temp={job['temperature']}\n"
                f"      {job['scenario']}"
            )
        print("\n--dry-run: nie wywołano modelu, nic nie zapisano.")
        return 0

    if len(jobs) < 25:
        # tools/derive_patterns.py refuses to rank below this, and for the
        # same reason: fewer documents describe the handful that were drawn,
        # not the model.
        print(
            f"[uwaga] {len(jobs)} dokumentów to poniżej progu 25 stosowanego "
            "przez derive_patterns.py. Rozważ wyższe --per-category."
        )

    try:
        settings = LlmSettings.from_environment(args.env_file)
    except LlmConfigurationError as exc:
        parser.error(str(exc))

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    index = next_index(args.out)
    written = skipped = 0
    today = datetime.now(timezone.utc).date().isoformat()

    with OpenAICompatibleRewriter(settings) as client:
        if not client.probe():
            parser.error(
                "Endpoint niedostępny: " + "; ".join(client.metadata.warnings or ["brak szczegółów"])
            )
        model_name = client.metadata.model

        for job in jobs:
            label = f"{job['category']}/{job['ordinal']}"
            # An empty body is retried once; a short one is not.
            #
            # They look alike in the output and are not alike at all. A model
            # that answers a drafting request with three sentences is telling
            # us something about the prompt, and asking again would hide it.
            # An empty 200 is the server, not the model - measured on the
            # first probe batch, where the first five calls came back empty
            # and the last three succeeded, in plain alphabetical order. That
            # is a cold model finishing its load, and one retry covers it.
            text = None
            for attempt in (1, 2):
                try:
                    text = client.complete_text(
                        messages_for(job),
                        max_tokens=MAX_TOKENS,
                        temperature=job["temperature"],
                    )
                    break
                except LlmEndpointError as exc:
                    message = _safe_error(exc)
                    retryable = any(w in message.lower() for w in ("pust", "rozumowanie", "ucięta"))
                    if attempt == 1 and retryable:
                        print(f"  [{message[:48]}] {label}: ponawiam")
                        time.sleep(2.0)
                        continue
                    print(f"  [błąd] {label}: {message}")
                    break
            if text is None:
                skipped += 1
                continue

            words = len(text.split())
            if words < MIN_WORDS:
                # Reported rather than retried: a model that answers a
                # drafting request with three sentences is telling us
                # something about the prompt, and silently asking again
                # would hide it.
                print(f"  [za krótki] {label}: {words} słów (< {MIN_WORDS})")
                skipped += 1
                continue

            name = f"ai_legal_{index:02d}_{job['category']}.txt"
            (args.out / name).write_text(text.strip() + "\n", encoding="utf-8")
            manifest.append(
                {
                    "id": Path(name).stem,
                    "file": name,
                    "type": job["category"],
                    "category": job["category"],
                    "focus": [],
                    "generated_by": model_name,
                    "generated_on": today,
                    "prompt_style": job["style"]["id"],
                    "temperature": job["temperature"],
                    "scenario": job["scenario"],
                    # Recorded because the calibrated score is not stable
                    # below roughly 600 words: truncating one document to 150
                    # words doubled its score, and the same happens to human
                    # text. Any analysis of this corpus has to be able to
                    # control for length rather than discover it later.
                    "words": words,
                }
            )
            print(f"  [{words:>5} słów] {name}  styl={job['style']['id']} temp={job['temperature']}")
            index += 1
            written += 1
            if args.delay:
                time.sleep(args.delay)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nZapisano {written} dokumentów, pominięto {skipped}.")
    print(f"Manifest: {manifest_path}  (łącznie {len(manifest)} wpisów)")
    print(
        "\nKorpus pochodzi z jednego endpointu, więc mierzy jeden model — "
        "zapisz to jako ograniczenie w README i w publikacji."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
