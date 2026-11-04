"""PostgreSQL integration tests in a disposable, randomly named schema.

Set TEST_DATABASE_URL explicitly. No application tables are modified.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import os
from threading import Event
import unittest
from unittest.mock import patch
from uuid import uuid4

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, select, func
from sqlalchemy.engine import make_url

from app.api import create_app
from app.database import Database, BrowserSession, Conversation, Message, Feedback, now
from app.history import cleanup, token_hash
from app.observability import Tracer
from app.service import GenerationUnavailable


class FakeService:
    def __init__(self):
        self.tracer = Tracer()
        self.calls = 0
        self.fail = False
        self.entered = Event()
        self.release = None

    def chat(self, question, retriever, request_id):
        self.calls += 1
        self.entered.set()
        if self.release:
            assert self.release.wait(10)
        if self.fail:
            raise GenerationUnavailable()
        return dict(answer='Saved answer', retriever=retriever, generation_model='fake',
                    passages=[dict(text='Evidence', source='test.pdf', page=1, score=.8)],
                    request_id=request_id, trace_id=request_id)

    def close(self):
        pass


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Set TEST_DATABASE_URL for PostgreSQL integration checks')
class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = 'tb_test_' + uuid4().hex
        cls.admin = create_engine(os.environ['TEST_DATABASE_URL'], hide_parameters=True)
        with cls.admin.begin() as db:
            db.execute(text(f'CREATE SCHEMA {cls.schema}'))
        url = make_url(os.environ['TEST_DATABASE_URL']).update_query_dict({'options': f'-csearch_path={cls.schema}'})
        cls.url = url.render_as_string(hide_password=False)
        with patch.dict(os.environ, {'DATABASE_URL': cls.url}):
            command.upgrade(Config('alembic.ini'), 'head')

    @classmethod
    def tearDownClass(cls):
        assert cls.schema.startswith('tb_test_') and len(cls.schema) == 40
        with cls.admin.begin() as db:
            db.execute(text(f'DROP SCHEMA {cls.schema} CASCADE'))
        cls.admin.dispose()

    def setUp(self):
        self.database = Database(self.url)
        self.service = FakeService()
        self.app = create_app(lambda: self.service, lambda: Database(self.url))
        self.client = TestClient(self.app, headers={'X-TB-Client': 'web'})
        self.client.__enter__()
        self.client.post('/session').raise_for_status()
        self.cid, self.mid = str(uuid4()), str(uuid4())
        self.path = f'/conversations/{self.cid}'
        self.message_path = f'{self.path}/messages/{self.mid}'
        self.body = dict(question='What is TB?', retriever='single_nq', attempt=0)
        self.client.put(self.path, json={'title': 'What is TB?'}).raise_for_status()

    def tearDown(self):
        self.client.delete('/conversations')
        self.client.__exit__(None, None, None)
        self.database.close()

    def test_cookie_hashed_and_fixed_expiry(self):
        response = self.client.post('/session')
        cookie = response.headers['set-cookie']
        self.assertIn('HttpOnly', cookie)
        self.assertIn('SameSite=strict', cookie)
        token = self.client.cookies['tb_session']
        with self.database.session() as db:
            record = db.scalar(select(BrowserSession).where(BrowserSession.token_hash == token_hash(token)))
            self.assertNotEqual(record.token_hash, token)
            expiry = record.expires_at
        self.assertEqual(response.json()['expires_at'], self.client.post('/session').json()['expires_at'])
        self.assertLess(expiry - now(), timedelta(days=30))

    def test_save_reopen_restart_and_rename(self):
        response = self.client.put(self.message_path, json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'completed')
        self.assertEqual(self.client.patch(self.path, json={'title': 'My notes'}).status_code, 200)
        with TestClient(create_app(lambda: FakeService(), lambda: Database(self.url)), headers={'X-TB-Client': 'web'}) as other:
            other.cookies.update(self.client.cookies)
            saved = other.get(self.path).json()
            self.assertEqual(saved['title'], 'My notes')
            self.assertEqual(saved['messages'][0]['answer'], response.json()['answer'])
            self.assertEqual(len(other.get('/conversations').json()), 1)

    def test_other_browser_cannot_read_write_delete_or_rate(self):
        self.client.put(self.message_path, json=self.body)
        with TestClient(self.app, headers={'X-TB-Client': 'web'}) as other:
            other.post('/session')
            self.assertEqual(other.get('/conversations').json(), [])
            for response in (other.get(self.path), other.delete(self.path),
                             other.patch(self.path, json={'title': 'stolen'}),
                             other.put(self.path, json={'title': 'stolen'}),
                             other.put(self.message_path, json=self.body),
                             other.put(self.message_path + '/feedback', json={'rating': -1})):
                self.assertEqual(response.status_code, 404)
        self.assertEqual(self.service.calls, 1)

    def test_csrf_and_missing_session(self):
        self.assertEqual(self.client.delete(self.path, headers={'X-TB-Client': ''}).status_code, 403)
        self.assertEqual(self.client.get(self.path, headers={'Sec-Fetch-Site': 'cross-site'}).status_code, 403)
        self.client.cookies.clear()
        self.assertEqual(self.client.get(self.path).status_code, 401)
        self.assertEqual(self.client.get(self.path).headers['cache-control'], 'no-store')

    def test_idempotent_completion_and_payload_conflict(self):
        first = self.client.put(self.message_path, json=self.body).json()
        again = self.client.put(self.message_path, json=self.body).json()
        self.assertEqual(first, again)
        self.assertEqual(self.service.calls, 1)
        self.assertEqual(self.client.put(self.message_path, json={**self.body, 'question': 'Different'}).status_code, 409)

    def test_failure_and_explicit_retry_once(self):
        self.service.fail = True
        first = self.client.put(self.message_path, json=self.body).json()
        self.assertEqual(first['status'], 'failed')
        self.client.put(self.message_path, json=self.body)
        self.assertEqual(self.service.calls, 1)
        self.service.fail = False
        body = {**self.body, 'attempt': 1}
        self.assertEqual(self.client.put(self.message_path, json=body).json()['status'], 'completed')
        self.client.put(self.message_path, json=body)
        self.assertEqual(self.service.calls, 2)
        self.assertEqual(len(self.client.get(self.path).json()['messages']), 1)

    def test_concurrent_duplicate_does_not_generate_twice(self):
        self.service.release = Event()
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.client.put, self.message_path, json=self.body)
            self.assertTrue(self.service.entered.wait(5))
            try:
                duplicate = self.client.put(self.message_path, json=self.body)
                self.assertEqual(duplicate.json()['status'], 'pending')
                self.assertEqual(self.service.calls, 1)
            finally:
                self.service.release.set()
            self.assertEqual(future.result().json()['status'], 'completed')

    def test_delete_during_generation_cannot_resurrect_history(self):
        self.service.release = Event()
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.client.put, self.message_path, json=self.body)
            self.assertTrue(self.service.entered.wait(5))
            try:
                self.assertEqual(self.client.delete(self.path).status_code, 200)
            finally:
                self.service.release.set()
            self.assertEqual(future.result().status_code, 404)
        self.assertEqual(self.client.get('/conversations').json(), [])

    def test_feedback_upsert_and_cascading_delete(self):
        self.client.put(self.message_path, json=self.body)
        path = self.message_path + '/feedback'
        self.assertEqual(self.client.put(path, json={'rating': 1, 'comment': 'Useful'}).status_code, 200)
        self.client.put(path, json={'rating': -1, 'comment': 'Missing evidence'})
        saved = self.client.get(self.path).json()['messages'][0]
        self.assertEqual(saved['feedback'], {'rating': -1, 'comment': 'Missing evidence'})
        self.client.delete(self.path)
        with self.database.session() as db:
            self.assertIsNone(db.get(Message, self.mid))
            self.assertIsNone(db.get(Feedback, self.mid))

    def test_expiry_denies_access_and_cleanup_cascades(self):
        self.client.put(self.message_path, json=self.body)
        with self.database.session.begin() as db:
            session = db.scalar(select(BrowserSession).where(BrowserSession.token_hash == token_hash(self.client.cookies['tb_session'])))
            sid = session.id
            session.expires_at = now() - timedelta(seconds=1)
        self.assertEqual(self.client.get(self.path).status_code, 401)
        cleanup(self.database)
        with self.database.session() as db:
            self.assertIsNone(db.get(BrowserSession, sid))
            self.assertIsNone(db.get(Message, self.mid))
        self.client.post('/session')
        self.assertEqual(self.client.get('/conversations').json(), [])

    def test_stale_pending_is_retryable(self):
        with self.database.session.begin() as db:
            db.add(Message(id=self.mid, conversation_id=self.cid, question=self.body['question'], retriever='single_nq',
                           status='pending', attempt=0, request_id=str(uuid4()), updated_at=now() - timedelta(minutes=11)))
        self.assertEqual(self.client.get(self.path).json()['messages'][0]['status'], 'failed')
        result = self.client.put(self.message_path, json={**self.body, 'attempt': 1})
        self.assertEqual(result.json()['status'], 'completed')

    def test_clear_all_does_not_clear_other_session(self):
        with TestClient(self.app, headers={'X-TB-Client': 'web'}) as other:
            other.post('/session')
            cid = str(uuid4())
            other.put(f'/conversations/{cid}', json={'title': 'Other browser'})
            self.client.delete('/conversations')
            self.assertEqual(len(other.get('/conversations').json()), 1)
            other.delete('/conversations')
