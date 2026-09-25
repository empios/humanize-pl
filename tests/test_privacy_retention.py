import json
import os

import httpx
import pytest

from humanize_pl.llm import LlmSettings, OpenAICompatibleRewriter
from humanize_pl.privacy import RequestMask, processing_location
from humanize_pl.retention import MARKER, delete_run, expire_runs, finish_run, register_run, runs
from humanize_pl.safety.protectors import protect_text


@pytest.mark.parametrize('secret', ['Kowalski', 'Kowalskiego', 'Pani Nowak', 'Jan Nowak',
    'ul. Lipowa 7/9, Kraków', 'al. Jana Pawła 12, 00-001 Warszawa', '31-001 Kraków',
    'jan@example.pl', '+48 123 456 789', 'PL12345678901234567890123456',
    '12 3456 7890 1234 5678 9012 3456', 'PESEL: 12345678901'])
def test_sensitive_forms_are_masked_and_round_trip(secret):
    protected = protect_text(f'Dane: {secret}.', include_sensitive=True)
    assert secret not in protected.text
    assert protected.restore(protected.text) == f'Dane: {secret}.'


def test_nested_quote_restoration():
    source = 'Powiedział: „Jan Kowalski otrzymał 100 zł”.'
    protected = protect_text(source, include_sensitive=True)
    assert protected.restore(protected.text) == source


def test_every_message_field_passes_through_request_mask():
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'cytat': payload['messages'][1]['content']})}}]})
    client = OpenAICompatibleRewriter(LlmSettings('http://127.0.0.1:8000/v1', 'test'),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    source = 'Kowalski, ul. Lipowa 7/9, Kraków'
    reply = client.complete_json([{'role': 'system', 'content': 'JSON; profil: '+source},
                                  {'role': 'user', 'content': source},
                                  {'role': 'assistant', 'content': source}])
    sent = json.dumps(requests, ensure_ascii=False)
    assert all(value not in sent for value in ('Kowalski', 'Lipowa', 'Kraków'))
    assert reply['cytat'] == source
    assert requests[0]['messages'][0]['content'].startswith('JSON;')
    assert client.metadata.to_report()['processing_location'] == 'local_loopback'


def test_one_mapping_for_all_fields_and_json_escaping():
    mask = RequestMask()
    messages = mask.messages([{'role': 'system', 'content': 'Jan Kowalski'}, {'role': 'user', 'content': 'Jan Kowalski'}])
    assert messages[0]['content'] == messages[1]['content']
    data = {'choices': [{'message': {'content': json.dumps({'quote': messages[0]['content']})}}]}
    assert json.loads(mask.response(data)['choices'][0]['message']['content'])['quote'] == 'Jan Kowalski'
    with pytest.raises(ValueError):
        mask.restore('__PRIVATE_unknown__')


@pytest.mark.parametrize('url,expected', [('http://localhost:8000', 'local_loopback'),
    ('http://[::1]:8000', 'local_loopback'), ('http://127.0.0.2', 'local_loopback'),
    ('http://localhost.example.com', 'external_or_network'), ('http://192.168.1.2', 'external_or_network'),
    ('https://model.test/v1', 'external_or_network')])
def test_location_does_not_guess_that_network_hosts_are_local(url, expected):
    assert processing_location(url) == expected


def test_retention_only_removes_owned_finished_runs(tmp_path):
    old, active, unowned = [tmp_path / name for name in ('old', 'active', 'documents')]
    for path in (old, active, unowned):
        path.mkdir()
        (path / 'document.txt').write_text('treść', encoding='utf-8')
    register_run(old)
    register_run(active)
    finish_run(old)
    for path in (old, active):
        os.utime(path / MARKER, (1, 1))
    assert expire_runs(tmp_path, now=8*86400) == ['old']
    assert active.exists() and unowned.exists()
    assert not runs(tmp_path)
    with pytest.raises(ValueError):
        delete_run(tmp_path, 'active')
    with pytest.raises(ValueError):
        delete_run(tmp_path, 'documents')
    with pytest.raises(ValueError):
        delete_run(tmp_path, '..')
    finish_run(active)
    (tmp_path / 'active.zip').write_bytes(b'archive')
    delete_run(tmp_path, 'active')
    assert not active.exists() and not (tmp_path / 'active.zip').exists()
