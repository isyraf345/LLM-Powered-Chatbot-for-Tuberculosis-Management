import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from openai import APITimeoutError
import httpx

from app.api import create_app
from app.context import build_context
from app.service import ChatService
from app.observability import Tracer

PASSAGES = [{'text': 'Evidence about TB.', 'source': 'guideline.pdf', 'page': 7, 'score': .8}]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.retriever.search.return_value = PASSAGES
        self.generator = Mock()
        self.generator.generate.return_value = 'An evidence-based answer.'
        self.rf = Mock(return_value=self.retriever)
        self.gf = Mock(return_value=self.generator)
        self.service = ChatService(self.rf, self.gf, tracer=Tracer())
        self.client = TestClient(create_app(lambda: self.service, database_factory=lambda: None))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_retrievers_and_docs_do_not_load_models(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        data = self.client.get('/retrievers').json()
        self.assertEqual(data['default'], 'single_nq')
        self.assertEqual(len(data['retrievers']), 3)
        self.assertIn('/chat', self.client.get('/openapi.json').json()['paths'])
        self.rf.assert_not_called()
        self.gf.assert_not_called()

    def test_request_uses_selected_retriever_and_exact_context(self):
        r = self.client.post('/chat', json={'question': 'What is TB?', 'retriever': 'lfqa'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['passages'], PASSAGES)
        self.assertEqual(r.json()['generation_model'], 'gpt-4o-mini')
        self.rf.assert_called_once_with('lfqa')
        self.retriever.search.assert_called_once_with('What is TB?', k=5)
        self.generator.generate.assert_called_once_with('What is TB?', build_context(PASSAGES))

    def test_invalid_inputs_do_not_run_inference(self):
        for body in ({}, {'question': ''}, {'question': '   '}, {'question': 3},
                     {'question': 'x' * 4001}, {'question': 'TB', 'retriever': 'unknown'},
                     {'question': 'TB', 'history': []}):
            with self.subTest(body=str(body)[:80]):
                self.assertEqual(self.client.post('/chat', json=body).status_code, 422)
        self.rf.assert_not_called()

    def test_reuses_model_and_evicts_on_switch(self):
        for name in ('single_nq', 'single_nq', 'lfqa'):
            self.assertEqual(self.client.post('/chat', json={'question': 'TB', 'retriever': name}).status_code, 200)
        self.assertEqual(self.rf.call_count, 2)
        self.assertEqual(self.gf.call_count, 1)

    def test_missing_configuration_fails_before_loading_weights(self):
        self.gf.side_effect = ValueError('secret configuration detail')
        r = self.client.post('/chat', json={'question': 'TB'})
        self.assertEqual(r.status_code, 503)
        self.assertNotIn('secret', r.text)
        self.rf.assert_not_called()

    def test_model_failure_can_recover(self):
        self.rf.side_effect = [RuntimeError('private path'), self.retriever]
        first = self.client.post('/chat', json={'question': 'TB'})
        self.assertEqual(first.status_code, 503)
        self.assertNotIn('private', first.text)
        self.assertEqual(self.client.post('/chat', json={'question': 'TB'}).status_code, 200)

    def test_timeout_does_not_leak_provider_details(self):
        self.generator.generate.side_effect = APITimeoutError(request=httpx.Request('POST', 'https://example.test'))
        self.assertEqual(self.client.post('/chat', json={'question': 'TB'}).status_code, 504)

    def test_busy_returns_retry_header(self):
        self.service._lock.acquire()
        try:
            r = self.client.post('/chat', json={'question': 'TB'})
            self.assertEqual(r.status_code, 503)
            self.assertIn('retry-after', r.headers)
        finally:
            self.service._lock.release()

    def test_empty_answer_is_not_success(self):
        self.generator.generate.return_value = ' '
        self.assertEqual(self.client.post('/chat', json={'question': 'TB'}).status_code, 503)


class GenerationAndContextTests(unittest.TestCase):
    def test_generation_prompt_and_settings_match_baseline(self):
        from app.generator import AnswerGenerator
        baseline = json.loads((Path(__file__).resolve().parents[1] / 'docs/baseline/manifest.json').read_text())['settings']
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=' answer '))])
        with patch('app.generator.OPENAI_API_KEY', 'test-only'), patch('app.generator.OpenAI', return_value=client):
            generator = AnswerGenerator()
            self.assertEqual(generator.generate('question', 'context'), 'answer')
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs['messages'][0]['content'], baseline['system_prompt'])
            self.assertEqual(len(kwargs['messages']), 2)
            for name in ('temperature', 'top_p'):
                self.assertEqual(kwargs[name], baseline[name])
            self.assertEqual(kwargs['max_tokens'], baseline['max_output_tokens'])
            self.assertEqual(kwargs['model'], baseline['OPENAI_MODEL'])
            generator.close()
            client.close.assert_called_once()

    def test_context_stops_at_first_overflow_instead_of_skipping(self):
        passages = PASSAGES + [{**PASSAGES[0], 'text': 'x' * 4000}] + PASSAGES
        self.assertEqual(build_context(passages), build_context(PASSAGES))
        self.assertEqual(build_context([]), '')

    def test_shutdown_closes_generator(self):
        generator = Mock()
        generator.generate.return_value = 'answer'
        retriever = Mock()
        retriever.search.return_value = PASSAGES
        service = ChatService(lambda _: retriever, lambda: generator, tracer=Tracer())
        with TestClient(create_app(lambda: service, database_factory=lambda: None)) as client:
            client.post('/chat', json={'question': 'TB'})
        generator.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
