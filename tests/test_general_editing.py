import json
import re
from dataclasses import replace

import httpx
import pytest
from typer.testing import CliRunner

from humanize_pl import GeneralOptions, GeneralProfile, humanize
from humanize_pl.cli import app
from humanize_pl.document import DocumentType, RewriteBackend
from humanize_pl.flows.base import FlowSettings, _rewrite_general_paragraphs, run_all_layers
from humanize_pl.general import editorial_findings, protected_lines
from humanize_pl.llm import LlmSettings, OpenAICompatibleRewriter


def model(propose, requests):
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        user = payload['messages'][1]['content']
        source = re.search(r'Fragment do redakcji:\n(.*?)\nZauważone problemy:', user, re.DOTALL)
        text = source.group(1) if source else 'To jest test połączenia.'
        identifier = re.search(r'fragment_id: (\S+)', user)
        body = {'fragment_id': identifier.group(1) if identifier else 'capability-test',
                    'source': text, 'proposal': propose(text) if source else text, 'rationale': 'Redakcja.'}
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(body)}}]})
    return OpenAICompatibleRewriter(LlmSettings('https://model.test/v1', 'test', '', 2),
                                    client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize('profile', list(GeneralProfile))
def test_profiles_reach_model_and_report(profile):
    requests = []
    options = GeneralOptions(profile=profile, audience='początkujący', tone='warm', formality='casual')
    settings = FlowSettings(track='general', rewrite_backend=RewriteBackend.hybrid, general_options=options)
    rewriter = model(lambda text: text, requests)
    outcome, _ = run_all_layers('Ogród zimą odpoczywa. Rośliny potrzebują mniej wody.', name='test', settings=settings, rewriter=rewriter, llm_prepared=True)
    assert outcome.to_json()['general_options']['profile'] == profile.value
    assert 'początkujący' in requests[-1]['messages'][0]['content']
    assert 'warm' in requests[-1]['messages'][0]['content']


def test_paragraph_context_includes_neighbours_but_not_their_facts():
    requests = []
    rewriter = model(lambda text: text, requests)
    source = 'Pierwszy akapit.\n\nDrugi akapit. To jego dalsza część.\nOstatni akapit.'
    text, changes, _ = _rewrite_general_paragraphs(source, rewriter=rewriter, options=GeneralOptions(), protected=set())
    assert text == source and not changes
    middle = requests[2]['messages'][1]['content']
    assert 'Fragment do redakcji:\nDrugi akapit. To jego dalsza część.' in middle
    assert 'Pierwszy akapit.' in middle and 'Ostatni akapit.' in middle


@pytest.mark.parametrize('intensity', ['light', 'style', 'rewrite'])
@pytest.mark.parametrize('proposal', ['Pacjent nie ma gorączki.', 'Pacjent ma gorączkę i kaszel.', 'Lekarz ma gorączkę.'])
def test_all_intensities_reject_changed_claims(intensity, proposal):
    rewriter = model(lambda text: proposal, [])
    result = rewriter.rewrite_fragment('Pacjent ma gorączkę.', fragment_id='p1', document_type=DocumentType.general,
                                      general_options=GeneralOptions(intensity=intensity, max_shortening=50))
    assert not result.accepted


def test_shortening_and_intensity_are_independent_limits():
    source = 'Warto podkreślić, że ogród zimą potrzebuje mniej pracy, niż się wydaje.'
    shorter = 'Ogród zimą potrzebuje mniej pracy, niż się wydaje.'
    assert GeneralOptions(intensity='rewrite', max_shortening=0).rejection(source, shorter)
    assert GeneralOptions(intensity='style', max_shortening=50).rejection(source, shorter) is None
    assert GeneralOptions(intensity='light').rejection('Kot spał spokojnie pod wysokim drzewem.', 'Pod wysokim drzewem spokojnie spał kot.')
    assert GeneralOptions(protected_terms=('ogrodu',)).rejection('Dotyczy ogrodu.', 'Dotyczy parku.')


@pytest.mark.parametrize('source', [
    '— Warto podkreślić, że czekałem! — powiedział.\n— Czekałem i czekałem.',
    '```python\nWarto podkreślić, że to kod.\n\nprint("dwa  odstępy")\n```',
    '„Warto podkreślić,\nże taki był początek”.',
    '`warto podkreślić` to nazwa polecenia.',
    'Tak, tak. Jeszcze raz. Jeszcze raz.\nRośliny zostały posadzone przez sąsiadów.',
])
def test_intentional_style_survives_the_actual_flow(source):
    result = humanize(source, track='general', pdf=False)
    assert result.text == source
    assert not result.applied_changes


def test_blank_lines_do_not_shift_protection():
    source = 'Pierwszy akapit.\n\n— Warto podkreślić, że to celowy dialog.\n\n```\nwarto podkreślić\n```'
    assert set(protected_lines(source)) == {2, 4, 5, 6}
    result = humanize(source, track='general', pdf=False)
    assert result.text == source


@pytest.mark.parametrize('prefix', ['\n', 'Pierwszy akapit.\n\n', 'Pierwszy akapit.\n \n'])
def test_reverted_edits_after_blank_lines_are_not_reported(prefix):
    source = prefix + 'Warto podkreślić, że ogród zimą potrzebuje mniej pracy, niż się wydaje.'
    result = humanize(source, track='general', general_options={'max_shortening': 0}, pdf=False)
    assert result.text == source
    assert result.changes_applied == 0
    assert result.applied_changes == result.outcomes[0].examples == []
    assert not result.outcomes[0].review['proposals']


def test_protection_filters_only_the_actual_protected_paragraph():
    sentence = 'Warto podkreślić, że ogród zimą potrzebuje mniej pracy, niż się wydaje.'
    source = f'Pierwszy akapit.\n\n    {sentence}\n{sentence}'
    result = humanize(source, track='general', pdf=False)
    assert result.text.split('\n')[2] == '    ' + sentence
    assert result.text.split('\n')[3] == 'Ogród zimą potrzebuje mniej pracy, niż się wydaje.'
    assert result.changes_applied == 1
    assert result.applied_changes[0]['paragraph_index'] == 3
    assert result.outcomes[0].review['proposals'][0]['paragraph_index'] == 3


@pytest.mark.parametrize('max_shortening,expected_words', [(30, 11), (50, 8)])
def test_shortening_budget_is_shared_by_rules_and_model(max_shortening, expected_words):
    from humanize_pl.safety.meaning import assertion

    source = 'Warto podkreślić, że warto zauważyć, że ogród zimą potrzebuje mniej pracy, niż się wydaje.'
    def propose(text):
        remaining = assertion(text)
        return remaining[:1].upper() + remaining[1:]

    rewriter = model(propose, [])
    outcome, _ = run_all_layers(source, name='test', settings=FlowSettings(
        track='general', rewrite_backend=RewriteBackend.hybrid,
        general_options=GeneralOptions(max_shortening=max_shortening),
    ), rewriter=rewriter, llm_prepared=True)
    assert len(re.findall(r'\w+', outcome.text_out)) == expected_words
    assert GeneralOptions(max_shortening=max_shortening).rejection(source, outcome.text_out) is None
    assert outcome.llm['accepted'] == (1 if max_shortening == 50 else 0)
    if max_shortening == 30:
        assert outcome.llm['decision_reasons']['rejected:general_edit_limits'] == 1
        assert not any(c['issue'] == 'general_paragraph_edit' for c in outcome.applied_changes)


def test_editorial_findings_are_separate_observations():
    findings = editorial_findings('Na chwilę obecną to to działa. To ostatnie wymaga wyjaśnienia.')
    assert {row['kind'] for row in findings} >= {'wordiness', 'reference_review', 'adjacent_repetition'}
    assert editorial_findings('— To to działa!\n`to to`') == []
    assert not any('ai' in row for row in findings)


def test_short_text_is_edited_without_score_claim():
    result = humanize('Warto podkreślić, że ogród zimą potrzebuje mniej pracy, niż się wydaje.',
                      track='general', general_options=GeneralOptions(max_shortening=50), pdf=False)
    assert result.text.startswith('Ogród')
    assert result.payload['documents'][0]['signal_interpretable'] is False
    assert not result.payload['summary']['signal_interpretable']


def test_legal_track_rejects_general_configuration():
    with pytest.raises(ValueError, match='general'):
        FlowSettings(general_options=GeneralOptions(profile='prose'))
    assert replace(FlowSettings(), track='general').general_options == GeneralOptions()


def test_cli_options_and_invalid_shortening():
    runner = CliRunner()
    result = runner.invoke(app, ['Ogród odpoczywa.', '--track', 'general', '--general-profile', 'email',
                                 '--intensity', 'light', '--tone', 'warm', '--no-pdf', '--no-report'])
    assert result.exit_code == 0, result.output
    bad = runner.invoke(app, ['Tekst.', '--track', 'general', '--max-shortening', '99'])
    assert bad.exit_code != 0


@pytest.mark.parametrize('options', [{'tone': 'invalid'}, {'formality': 'invalid'}, {'max_shortening': 1.5}, {'audience': 'x'*301}])
def test_invalid_preferences_fail_before_processing(options):
    with pytest.raises(ValueError):
        GeneralOptions(**options)
