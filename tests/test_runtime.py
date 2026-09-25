import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from humanize_pl import RunCancelled, RunControl, humanize
from humanize_pl.document import RewriteBackend
from humanize_pl.flows.base import FlowSettings, run_all_layers
from humanize_pl.llm import LlmEndpointError, LlmSettings, OpenAICompatibleRewriter
from humanize_pl.runtime import controlled


def response():
    return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
        'fragment_id': 'capability-test', 'source': 'To jest test połączenia.',
        'proposal': 'To jest test połączenia.', 'rationale': 'test'})}}]})


def test_cancelled_run_does_not_touch_output(tmp_path):
    control = RunControl()
    control.cancel()
    target = tmp_path / 'result.txt'
    target.write_text('Poprzedni wynik.', encoding='utf-8')
    with pytest.raises(RunCancelled):
        humanize('Nowy tekst.', output=target, track='general', pdf=False, control=control)
    assert target.read_text(encoding='utf-8') == 'Poprzedni wynik.'


def test_cancel_before_publication_removes_staged_file(tmp_path):
    control = RunControl()
    stages = []
    def progress(event):
        stages.append(event['stage'])
        if event['stage'] == 'publikowanie pliku':
            control.cancel()
    control.on_progress = progress
    target = tmp_path / 'result.txt'
    target.write_text('Poprzedni wynik.', encoding='utf-8')
    with pytest.raises(RunCancelled):
        humanize('Ogród odpoczywa.', output=target, track='general', pdf=False, control=control)
    assert 'analiza źródła' in stages and 'weryfikacja wyniku' in stages
    assert target.read_text(encoding='utf-8') == 'Poprzedni wynik.'
    assert not list(tmp_path.glob('.humanize-*'))


def test_cancellation_during_request_discards_reply_and_closes_client():
    control = RunControl()
    client = None
    def handler(request):
        control.cancel()
        return response()
    @controlled
    def run():
        nonlocal client
        client = OpenAICompatibleRewriter(LlmSettings('https://model.test', 'test'),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return client.probe()
    with pytest.raises(RunCancelled):
        run(control=control)
    assert client.metadata.status != 'ready'


def test_require_llm_rejects_runtime_error_after_successful_probe(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        return response() if len(requests) == 1 else httpx.Response(500, json={})
    rewriter = OpenAICompatibleRewriter(LlmSettings('https://model.test', 'test', timeout_seconds=2),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert rewriter.probe()
    monkeypatch.setattr(rewriter, '_wait_retry', lambda _: None)
    with pytest.raises(RuntimeError, match='podczas pracy'):
        run_all_layers('Ogród odpoczywa.', name='test',
            settings=FlowSettings(track='general', rewrite_backend=RewriteBackend.hybrid, require_llm=True),
            rewriter=rewriter, llm_prepared=True)


def test_optional_llm_error_is_reported_as_degradation(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout('private endpoint details', request=request)
    rewriter = OpenAICompatibleRewriter(LlmSettings('https://model.test', 'test', timeout_seconds=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    rewriter._probed, rewriter.metadata.status = True, 'ready'
    monkeypatch.setattr(rewriter, '_wait_retry', lambda _: None)
    outcome, _ = run_all_layers('Ogród odpoczywa.', name='test', settings=FlowSettings(track='general',
        rewrite_backend=RewriteBackend.hybrid), rewriter=rewriter, llm_prepared=True)
    assert outcome.llm['status'] == 'ready_with_errors'
    assert outcome.readiness_status == 'ready_with_warnings'
    assert any('zapytań do modelu' in warning for warning in outcome.warnings)
    assert outcome.text_out == 'Ogród odpoczywa.'
    assert 'private endpoint details' not in str(outcome.to_json())


def test_total_timeout_covers_late_success_not_only_socket_inactivity():
    def handler(request):
        time.sleep(.02)
        return response()
    rewriter = OpenAICompatibleRewriter(LlmSettings('https://model.test', 'test', timeout_seconds=.005),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert not rewriter.probe()
    assert any('limit czasu' in warning for warning in rewriter.metadata.warnings)


@pytest.fixture
def slow_http_endpoint():
    started, disconnected = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            mode = request['messages'][0]['content']
            body = json.dumps({'choices': [{'message': {'content': '{"ok": true}'}}]}).encode()
            headers = (f'HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n'
                       'Content-Type: application/json\r\nConnection: close\r\n\r\n').encode()
            if mode == 'fast':
                self.connection.sendall(headers + body)
                return
            try:
                if mode == 'body':
                    self.connection.sendall(headers)
                    slow_bytes = body
                else:
                    slow_bytes = headers + body
                started.set()
                for value in slow_bytes:
                    self.connection.sendall(bytes([value]))
                    time.sleep(.02)
            except OSError:
                disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.02), daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', started, disconnected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize('stage', ['headers', 'body'])
def test_total_deadline_interrupts_a_trickling_http_response(slow_http_endpoint, stage):
    endpoint, _, disconnected = slow_http_endpoint
    with OpenAICompatibleRewriter(LlmSettings(endpoint, 'test', timeout_seconds=.15)) as client:
        started = time.monotonic()
        with pytest.raises(LlmEndpointError, match='limit czasu'):
            client.complete_json([{'role': 'user', 'content': stage}])
        assert time.monotonic() - started < .8
        assert disconnected.wait(timeout=.8), 'Timed-out connection must actually close'
        assert client.complete_json([{'role': 'user', 'content': 'fast'}]) == {'ok': True}


def test_cancellation_interrupts_active_http_and_keeps_other_requests_usable(slow_http_endpoint):
    endpoint, started, disconnected = slow_http_endpoint
    control = RunControl()

    @controlled
    def run():
        with OpenAICompatibleRewriter(LlmSettings(endpoint, 'test', timeout_seconds=5)) as client:
            return client.complete_json([{'role': 'user', 'content': 'body'}])

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(run, control=control)
        assert started.wait(timeout=2)
        cancelled_at = time.monotonic()
        control.cancel()
        with pytest.raises(RunCancelled):
            pending.result(timeout=1)
        assert time.monotonic() - cancelled_at < .8
        assert disconnected.wait(timeout=.8)
    # Cancelling one task must not stop the shared HTTP event loop.
    with OpenAICompatibleRewriter(LlmSettings(endpoint, 'test', timeout_seconds=1)) as client:
        assert client.complete_json([{'role': 'user', 'content': 'fast'}]) == {'ok': True}


def test_auxiliary_model_errors_are_counted_for_require_llm():
    rewriter = OpenAICompatibleRewriter(LlmSettings('https://model.test', 'test'),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200,
            json={'choices': [{'message': {'content': 'nie JSON'}}]}))))
    with pytest.raises(LlmEndpointError):
        rewriter.complete_json([{'role': 'user', 'content': 'test'}])
    assert rewriter.metadata.status == 'ready_with_errors'


def test_xlsm_rejected_before_reading_or_writing(tmp_path):
    source, target = tmp_path / 'input.xlsm', tmp_path / 'output.xlsx'
    source.write_bytes(b'VBA must not be lost')
    with pytest.raises(ValueError, match='XLSM'):
        humanize(source, output=target, column='A', pdf=False)
    assert source.read_bytes() == b'VBA must not be lost'
    assert not target.exists()


def test_unsupported_formats_fail_in_api_cli_and_direct_workbook_flow(tmp_path):
    from typer.testing import CliRunner

    from humanize_pl.cli import app
    from humanize_pl.flows.xlsx_flow import run_xlsx_flow

    source = tmp_path / 'source.xlsm'
    source.write_bytes(b'original')
    result = CliRunner().invoke(app, [str(source), '--no-pdf'])
    assert result.exit_code != 0 and 'XLSM' in result.output
    with pytest.raises(ValueError, match='XLSM'):
        run_xlsx_flow(source, tmp_path / 'output.xlsx', column='A', settings=FlowSettings())
    source_pdf = tmp_path / 'source.pdf'
    source_pdf.write_bytes(b'original PDF')
    with pytest.raises(ValueError, match='PDF'):
        humanize(source_pdf, pdf=False)
    xlsx = tmp_path / 'source.xlsx'
    xlsx.write_bytes(b'not read before validation')
    with pytest.raises(ValueError, match='XLSM'):
        run_xlsx_flow(xlsx, tmp_path / 'output.xlsm', column='A', settings=FlowSettings())
    assert sorted(p.name for p in tmp_path.iterdir()) == ['source.pdf', 'source.xlsm', 'source.xlsx']
