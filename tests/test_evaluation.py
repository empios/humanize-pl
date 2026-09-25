import json
from copy import deepcopy

import pytest

from humanize_pl.evaluation import evaluate_packet, prepare_packet
from humanize_pl.flows.base import FlowSettings


def corpus(tmp_path):
    # Synthetic fixture exercises the packet protocol, not a quality benchmark.
    rows = []
    for index, (source, genre, generator) in enumerate([
        ('human', 'email', None), ('ai', 'article', 'generator-a'), ('ai', 'product', 'generator-b')
    ]):
        path = tmp_path / f'{index}.txt'
        path.write_text(f'Ogród ma {index + 1} drzewa.', encoding='utf-8')
        rows.append({'id': f'private-id-{index}', 'file': path.name, 'source': source,
                     'genre': genre, 'generator': generator, 'approved_control': True,
                     'track': 'general', 'split': 'heldout'})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(rows), encoding='utf-8')
    return manifest


def test_packet_hides_variant_identity_and_has_no_invented_ratings(tmp_path):
    manifest = corpus(tmp_path)
    output = tmp_path / 'packet'
    prepared = prepare_packet(manifest, output, settings=FlowSettings(track='general'))
    assert prepared['human_evaluation'] == 'pending'
    blind = (output / 'blind.json').read_text(encoding='utf-8')
    assert 'generator-' not in blind and 'private-id' not in blind and 'edited' not in blind
    key = json.loads((output / 'key.json').read_text(encoding='utf-8'))
    ratings = json.loads((output / 'ratings.json').read_text(encoding='utf-8'))
    result = evaluate_packet(key, ratings)
    assert result['rated'] == 0 and not result['release_criteria_passed']
    wrong_packet = deepcopy(ratings)
    wrong_packet[0]['packet_sha256'] = 'different packet'
    with pytest.raises(ValueError, match='innego pakietu'):
        evaluate_packet(key, wrong_packet)
    with pytest.raises(ValueError, match='już istnieje'):
        prepare_packet(manifest, output, settings=FlowSettings(track='general'))


def test_training_or_unapproved_control_is_refused(tmp_path):
    manifest = corpus(tmp_path)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    rows[0]['split'] = 'training'
    manifest.write_text(json.dumps(rows), encoding='utf-8')
    with pytest.raises(ValueError, match='heldout'):
        prepare_packet(manifest, tmp_path / 'packet', settings=FlowSettings(track='general'))


def rating():
    return {'id': 'sample-0001', 'reviewer': 'reviewer-1', 'qualification': 'lawyer', 'preference': 'A',
            'A': {'content_preserved': True, 'language_errors': 0, 'unnecessary_edits': 0},
            'B': {'content_preserved': True, 'language_errors': 0, 'unnecessary_edits': 0}}


def test_release_criteria_read_harm_and_excessive_edits_not_style_score():
    key = {'track': 'general', 'items': [{'id': 'sample-0001', 'edited_side': 'A', 'source': 'human'}]}
    good = rating()
    assert evaluate_packet(key, [good])['release_criteria_passed']
    for field, value in [('content_preserved', False), ('language_errors', 1), ('unnecessary_edits', 1)]:
        bad = deepcopy(good)
        bad['A'][field] = value
        assert not evaluate_packet(key, [bad])['release_criteria_passed']
    assert not evaluate_packet(key, [])['release_criteria_passed']


def test_legal_evaluation_requires_lawyer_and_allows_tie():
    key = {'track': 'legal', 'items': [{'id': 'sample-0001', 'edited_side': 'A', 'source': 'human'}]}
    row = rating()
    row['qualification'] = 'reader'
    with pytest.raises(ValueError, match='lawyer'):
        evaluate_packet(key, [row])
    row['qualification'], row['preference'] = 'lawyer', 'tie'
    assert evaluate_packet(key, [row])['release_criteria_passed']
    with pytest.raises(ValueError, match='powtórzona'):
        evaluate_packet(key, [row, row])
