"""Held-out, blinded evaluation packets. Human ratings are never inferred from scores."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import typer

from humanize_pl.flow import humanize
from humanize_pl.flows.base import FlowSettings
from humanize_pl.flows.resume import file_digest
from humanize_pl.io.atomic import ensure_distinct_paths, write_text_atomic

app = typer.Typer(help="Zaślepiona ocena redakcji na odłożonym korpusie.")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def prepare_packet(manifest: Path, output: Path, *, settings: FlowSettings, seed: int = 1) -> dict:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest musi zawierać niepustą listę dokumentów odłożonych do oceny.")
    ids, hashes, sources, kinds, genres, generators = set(), set(), [], set(), set(), set()
    for row in rows:
        if row.get("split") != "heldout" or row.get("track") != settings.track.value:
            raise ValueError("Wszystkie próbki muszą mieć split=heldout i wybraną ścieżkę track.")
        if row.get("source") not in {"human", "ai"} or not row.get("genre") or not row.get("id"):
            raise ValueError("Wymagane pola próbki: id, genre, source (human/ai), file, track, split.")
        source = (manifest.parent / row["file"]).resolve()
        fingerprint = file_digest(source)
        if row["id"] in ids or fingerprint in hashes:
            raise ValueError("Powtórzony identyfikator lub tekst w korpusie.")
        ids.add(row["id"])
        hashes.add(fingerprint)
        sources.append(source)
        kinds.add(row["source"])
        genres.add(row["genre"])
        if row['source'] == 'ai':
            if not row.get('generator'):
                raise ValueError("Próbka AI wymaga nazwy generatora.")
            generators.add(row['generator'])
        elif row.get('approved_control') is not True:
            raise ValueError("Kontrola ludzka wymaga jawnego approved_control=true w manifeście.")
    if kinds != {'human', 'ai'} or len(genres) < 3 or len(generators) < 2:
        raise ValueError("Ocena wymaga kontroli ludzkich, co najmniej trzech gatunków i dwóch generatorów AI.")
    targets = [output / name for name in ('blind.json', 'key.json', 'ratings.json', 'technical.json')]
    ensure_distinct_paths([manifest, *sources], targets)
    if any(path.exists() for path in targets):
        raise ValueError("Pakiet oceny już istnieje; wybierz pusty katalog, aby nie nadpisać ocen.")
    randomizer = random.Random(seed)
    order = list(zip(rows, sources))
    randomizer.shuffle(order)
    blind, key, technical, ratings = [], [], [], []
    for i, (row, source) in enumerate(order, 1):
        original = source.read_text(encoding='utf-8')
        started = perf_counter()
        result = humanize(original, settings=settings, pdf=False)
        seconds = perf_counter() - started
        if not result.ok:
            raise ValueError(f"Nie udało się przygotować pozycji {row['id']}.")
        edited_side = randomizer.choice(('A', 'B'))
        packet_id = f"sample-{i:04d}"
        variants = {edited_side: result.text, 'B' if edited_side == 'A' else 'A': original}
        blind.append({'id': packet_id, 'A': variants['A'], 'B': variants['B']})
        key.append({'id': packet_id, 'document_id': row['id'], 'edited_side': edited_side,
                    'source': row['source'], 'genre': row['genre'], 'generator': row.get('generator'),
                    'source_sha256': file_digest(source)})
        technical.append({'id': packet_id, 'seconds': round(seconds, 4), 'changes': result.changes_applied,
                          'cost': None, 'cost_note': 'Brak potwierdzonego kosztu dostawcy; nie oszacowano.',
                          'readiness': result.readiness_status, 'warnings': result.warnings,
                          'model': result.payload.get('layers', {}).get('hosted_model', {})})
        ratings.append({'id': packet_id, 'reviewer': '', 'qualification': '', 'preference': None,
                        'A': {'content_preserved': None, 'language_errors': None, 'unnecessary_edits': None},
                        'B': {'content_preserved': None, 'language_errors': None, 'unnecessary_edits': None},
                        'notes': ''})
    package_root = Path(__file__).parent
    code_hash = hashlib.sha256(''.join(file_digest(path) for path in sorted(package_root.rglob('*.py'))).encode()).hexdigest()
    key_payload = {'schema': 1, 'track': settings.track.value, 'seed': seed,
                   'manifest_sha256': file_digest(manifest), 'implementation_sha256': code_hash,
                   'settings': json.loads(json.dumps(asdict(settings), default=str)),
                   'blind_sha256': hashlib.sha256(_json(blind).encode()).hexdigest(), 'items': key}
    for rating in ratings:
        rating['packet_sha256'] = key_payload['blind_sha256']
    for target, value in zip(targets, (blind, key_payload, ratings, technical)):
        write_text_atomic(target, _json(value), sources=[manifest, *sources])
    return {'samples': len(blind), 'track': settings.track.value, 'human_evaluation': 'pending', 'output': str(output)}


def evaluate_packet(key: dict, ratings: list[dict]) -> dict:
    if key.get('track') not in {'legal', 'general'}:
        raise ValueError("Nieznana ścieżka oceny.")
    expected = {row['id']: row for row in key['items']}
    if len(expected) != len(key['items']) or any(row.get('edited_side') not in {'A', 'B'} for row in expected.values()):
        raise ValueError("Klucz zawiera powtórzone próbki lub nieprawidłowy wariant.")
    seen, complete = set(), []
    for rating in ratings:
        if key.get('blind_sha256') and rating.get('packet_sha256') != key['blind_sha256']:
            raise ValueError("Oceny pochodzą z innego pakietu zaślepionego.")
        identifier = rating.get('id')
        if identifier not in expected or identifier in seen:
            raise ValueError("Nieznana lub powtórzona ocena próbki.")
        seen.add(identifier)
        if rating.get('preference') is None:
            continue
        if rating.get('preference') not in {'A', 'B', 'tie'} or not rating.get('reviewer'):
            raise ValueError("Ocena wymaga recenzenta i preferencji A/B/tie.")
        if key['track'] == 'legal' and rating.get('qualification') != 'lawyer':
            raise ValueError("Ocena ścieżki prawnej wymaga recenzenta z qualification=lawyer.")
        for side in ('A', 'B'):
            scores = rating.get(side, {})
            if not isinstance(scores.get('content_preserved'), bool):
                raise TypeError("Ocena zachowania treści musi być wartością true/false.")
            for field in ('language_errors', 'unnecessary_edits'):
                if type(scores.get(field)) is not int or scores[field] < 0:
                    raise ValueError("Liczby błędów i zbędnych zmian muszą być nieujemnymi liczbami całkowitymi.")
        complete.append((expected[identifier], rating))
    harmful = sum(not rating[item['edited_side']]['content_preserved'] for item, rating in complete)
    preferred = sum(rating['preference'] == item['edited_side'] for item, rating in complete)
    rejected = sum(rating['preference'] not in {item['edited_side'], 'tie'} for item, rating in complete)
    language_regressions = sum(rating[item['edited_side']]['language_errors'] > rating['B' if item['edited_side'] == 'A' else 'A']['language_errors'] for item, rating in complete)
    excessive_controls = sum(rating[item['edited_side']]['unnecessary_edits'] > 0 for item, rating in complete if item['source'] == 'human')
    full = len(complete) == len(expected) and bool(expected)
    passed = full and harmful == 0 and language_regressions == 0 and excessive_controls == 0
    if key['track'] == 'general':
        passed = passed and preferred > rejected
    return {'track': key['track'], 'rated': len(complete), 'total': len(expected), 'coverage_complete': full,
            'edited_preferred': preferred, 'original_preferred': rejected, 'ties': len(complete)-preferred-rejected,
            'harmful_edits': harmful, 'language_regressions': language_regressions,
            'human_controls_with_unnecessary_edits': excessive_controls,
            'release_criteria_passed': passed,
            'note': 'Wynik dotyczy ocenionej próbki i zadeklarowanych recenzentów; nie jest gwarancją jakości wszystkich dokumentów.'}


@app.command('prepare')
def prepare_command(manifest: Path, output: Path, track: str = 'general', seed: int = 1,
                    backend: str = 'rules'):
    from humanize_pl.document import RewriteBackend

    result = prepare_packet(manifest, output, settings=FlowSettings(track=track,
        rewrite_backend=RewriteBackend(backend), require_llm=backend == 'hybrid'), seed=seed)
    typer.echo(_json(result))


@app.command('score')
def score_command(key: Path, ratings: Path):
    typer.echo(_json(evaluate_packet(json.loads(key.read_text(encoding='utf-8')),
                                    json.loads(ratings.read_text(encoding='utf-8')))))


if __name__ == '__main__':
    app()
