import json
import os
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api import create_app
from app.observability import Tracer
from app.service import ChatService


def wait_for(predicate):
    deadline = monotonic() + 3
    while not predicate():
        if monotonic() >= deadline:
            raise AssertionError('Background exporter did not finish')
        sleep(.01)


class TraceTests(unittest.TestCase):
    def make_service(self, tracer):
        retriever = Mock()
        passages = [{'text': 'PRIVATE passage', 'source': 'source.pdf', 'page': 18, 'score': .7},
                    {'text': 'X' * 4000, 'source': 'source.pdf', 'page': 19, 'score': .6}]
        retriever.search.return_value = passages
        retriever.passages = [dict(p, id=i) for i, p in enumerate(passages, 451)]
        generator = Mock()
        generator.generate.return_value = 'PRIVATE answer'
        generator.last_usage = {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30}
        generator.last_messages = [{'role': 'user', 'content': 'PRIVATE question'}]
        return ChatService(lambda _: retriever, lambda: generator, tracer=tracer), generator

    def test_disabled_does_not_construct_client(self):
        factory = Mock()
        tracer = Tracer(client_factory=factory)
        service, _ = self.make_service(tracer)
        with TestClient(create_app(lambda: service, database_factory=lambda: None)) as client:
            r = client.post('/chat', json={'question': 'PRIVATE question'})
            self.assertEqual(r.status_code, 200)
            self.assertIsNone(r.json()['trace_id'])
            self.assertEqual(r.json()['request_id'], r.headers['x-request-id'])
        factory.assert_not_called()

    def test_metadata_only_tree_and_usage(self):
        exporter = Mock()
        tracer = Tracer(enabled=True, client_factory=lambda: exporter)
        service, _ = self.make_service(tracer)
        with TestClient(create_app(lambda: service, database_factory=lambda: None)) as client:
            r = client.post('/chat', json={'question': 'PRIVATE question'})
            wait_for(lambda: tracer.exported == 1)
            self.assertEqual(r.json()['trace_id'], r.headers['x-request-id'])
            records = [call.kwargs for call in exporter.create_run.call_args_list]
            self.assertEqual([x['name'] for x in records],
                             ['chat', 'load_resources', 'retrieval', 'context_construction', 'generation'])
            self.assertNotIn('PRIVATE', json.dumps(records, default=str))
            root = records[0]
            for child in records[1:]:
                self.assertEqual(child['parent_run_id'], root['id'])
                self.assertTrue(child['dotted_order'].startswith(root['dotted_order'] + '.'))
                self.assertGreaterEqual(child['end_time'], child['start_time'])
            self.assertEqual(records[2]['outputs']['passages'][0]['ids'], [451])
            self.assertEqual(records[3]['outputs']['included_ranks'], [1])
            llm = records[-1]
            self.assertEqual(llm['run_type'], 'llm')
            self.assertEqual(llm['extra']['metadata']['ls_model_name'], 'gpt-4o-mini')
            self.assertEqual(llm['outputs']['usage_metadata']['total_tokens'], 30)
            self.assertEqual(client.get('/health').json()['tracing']['exported'], 1)

    def test_content_opt_in(self):
        exporter = Mock()
        tracer = Tracer(enabled=True, content_enabled=True, client_factory=lambda: exporter)
        service, _ = self.make_service(tracer)
        try:
            service.chat('PRIVATE question', 'single_nq')
            wait_for(lambda: tracer.exported == 1)
            records = [call.kwargs for call in exporter.create_run.call_args_list]
            self.assertEqual(records[0]['inputs']['question'], 'PRIVATE question')
            self.assertEqual(records[-1]['outputs']['messages'][0]['content'], 'PRIVATE answer')
            self.assertEqual(records[-1]['inputs']['messages'][0]['content'], 'PRIVATE question')
        finally:
            service.close()

    def test_error_types_only_and_correlation(self):
        exporter = Mock()
        tracer = Tracer(enabled=True, client_factory=lambda: exporter)
        service, generator = self.make_service(tracer)
        generator.generate.side_effect = RuntimeError('PRIVATE secret provider body')
        with TestClient(create_app(lambda: service, database_factory=lambda: None)) as client:
            response = client.post('/chat', json={'question': 'PRIVATE question'})
            self.assertEqual(response.status_code, 500)
            wait_for(lambda: tracer.exported == 1)
            records = [call.kwargs for call in exporter.create_run.call_args_list]
            self.assertEqual(records[0]['id'], response.headers['x-request-id'])
            self.assertEqual(records[0]['error'], 'RuntimeError')
            self.assertEqual(records[-1]['error'], 'RuntimeError')
            self.assertNotIn('PRIVATE', json.dumps(records, default=str))

    def test_export_failure_does_not_change_answer(self):
        exporter = Mock()
        exporter.create_run.side_effect = TimeoutError('PRIVATE service failure')
        tracer = Tracer(enabled=True, client_factory=lambda: exporter)
        service, _ = self.make_service(tracer)
        with TestClient(create_app(lambda: service, database_factory=lambda: None)) as client:
            r = client.post('/chat', json={'question': 'question'})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['answer'], 'PRIVATE answer')
            wait_for(lambda: tracer.failed == 1)

    def test_slow_export_is_off_request_path_and_queue_bounded(self):
        entered, release = Event(), Event()
        def slow(**kwargs):
            entered.set()
            release.wait(timeout=3)
        exporter = Mock()
        exporter.create_run.side_effect = slow
        tracer = Tracer(enabled=True, queue_size=1, client_factory=lambda: exporter)
        service, _ = self.make_service(tracer)
        try:
            service.chat('one', 'single_nq')
            self.assertTrue(entered.wait(1))
            # These return while export remains blocked.
            service.chat('two', 'single_nq')
            service.chat('three', 'single_nq')
            self.assertFalse(release.is_set())
            self.assertEqual(tracer.dropped, 1)
            release.set()
            wait_for(lambda: tracer.exported == 2)
        finally:
            release.set()
            service.close()

    def test_enrichment_failure_does_not_break_chat(self):
        exporter = Mock()
        tracer = Tracer(enabled=True, client_factory=lambda: exporter)
        service, _ = self.make_service(tracer)
        with patch('app.service.file_version', side_effect=OSError('private path')):
            try:
                self.assertEqual(service.chat('question', 'single_nq')['answer'], 'PRIVATE answer')
                wait_for(lambda: tracer.exported == 1)
            finally:
                service.close()

    def test_missing_key_or_bad_endpoint_disables_tracing(self):
        for settings in ({'LANGSMITH_TRACING': 'true'},
                         {'LANGSMITH_TRACING': 'true', 'LANGSMITH_API_KEY': 'test-only',
                          'LANGSMITH_ENDPOINT': 'http://untrusted.example'}):
            with patch.dict(os.environ, settings, clear=True):
                tracer = Tracer.from_env()
                self.assertFalse(tracer.enabled)
                tracer.close()

    def test_system_certificate_adapter_survives_sdk_initialization(self):
        import ssl
        import requests
        import truststore
        with patch.dict(os.environ, {'LANGSMITH_TRACING': 'true', 'LANGSMITH_API_KEY': 'test-only',
                                     'LANGSMITH_ENDPOINT': 'https://api.smith.langchain.com'}, clear=True):
            tracer = Tracer.from_env()
            client = tracer.client_factory()
            try:
                request = requests.Request('POST', 'https://api.smith.langchain.com/runs').prepare()
                adapter = client.session.get_adapter(request.url)
                _, options = adapter.build_connection_pool_key_attributes(request, True)
                self.assertIsInstance(options['ssl_context'], truststore.SSLContext)
                self.assertEqual(options['ssl_context'].verify_mode, ssl.CERT_REQUIRED)
            finally:
                client.close()
                tracer.close()

    def test_client_constructor_failure_is_fail_open(self):
        factory = Mock(side_effect=ValueError('invalid SDK settings'))
        tracer = Tracer(enabled=True, client_factory=factory)
        service, _ = self.make_service(tracer)
        try:
            self.assertEqual(service.chat('question', 'single_nq')['answer'], 'PRIVATE answer')
            wait_for(lambda: tracer.failed == 1)
        finally:
            service.close()

    def test_real_sdk_accepts_completed_hierarchical_runs_without_network(self):
        from langsmith import Client
        from urllib3.util.retry import Retry
        sdk = Client(api_key='test-only', api_url='https://api.smith.langchain.com',
                     auto_batch_tracing=False, retry_config=Retry(total=0),
                     omit_traced_runtime_info=True, tracing_sampling_rate=1.0)
        tracer = Tracer(enabled=True, client_factory=lambda: sdk)
        service, _ = self.make_service(tracer)
        with patch.object(Client, 'request_with_retries', return_value=Mock()) as transport:
            try:
                service.chat('PRIVATE question', 'single_nq')
                wait_for(lambda: tracer.exported == 1 or tracer.failed > 0)
                self.assertEqual(tracer.failed, 0)
                self.assertEqual(transport.call_count, 5)
                wire = str(transport.call_args_list)
                self.assertNotIn('PRIVATE', wire)
                self.assertIn('usage_metadata', wire)
                self.assertIn('parent_run_id', wire)
            finally:
                service.close()


class UsageTests(unittest.TestCase):
    def test_usage_collected_without_extra_generation_call(self):
        from app.generator import AnswerGenerator
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='answer'))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=5)))
        client = Mock()
        client.chat.completions.create.return_value = response
        with patch('app.generator.OPENAI_API_KEY', 'test-only'), patch('app.generator.OpenAI', return_value=client):
            generator = AnswerGenerator()
            self.assertEqual(generator.generate('question', 'context'), 'answer')
            self.assertEqual(generator.last_usage['total_tokens'], 30)
            self.assertEqual(generator.last_usage['input_token_details']['cache_read'], 5)
            client.chat.completions.create.assert_called_once()
            response.usage = None
            generator.generate('question', 'context')
            self.assertEqual(generator.last_usage, {})


if __name__ == '__main__':
    unittest.main()
