"""Opt-in, privacy-first LangSmith traces with bounded background export."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def file_version(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def passage_summary(passages, corpus):
    lookup = {}
    for p in corpus:
        lookup.setdefault((p['source'], p.get('page'), p['text']), []).append(p['id'])
    return [{'ids': lookup.get((p['source'], p.get('page'), p['text']), []),
             'rank': i, 'score': float(p['score']), 'page': p.get('page')}
            for i, p in enumerate(passages, 1)]


class Trace:
    def __init__(self, exporter, request_id, metadata):
        self.exporter = exporter
        self.id = request_id
        self.records = []
        self.metadata = metadata
        self.root_order = None

    def content(self, **values):
        return values if self.exporter.content_enabled else {}

    def annotate(self, record, producer):
        """Observability enrichment must never affect the business operation."""
        if record is not None:
            try:
                record.update(producer())
            except Exception:
                logger.warning('Trace enrichment unavailable; content omitted.')

    @contextmanager
    def span(self, name, run_type='chain', inputs=None, metadata=None, root=False):
        if not self.exporter.enabled:
            yield None
            return
        now = datetime.now(timezone.utc)
        run_id = self.id if root else str(uuid4())
        order = now.strftime('%Y%m%dT%H%M%S%fZ') + run_id
        if root:
            self.root_order = order
        record = dict(id=run_id, name=name, run_type=run_type, trace_id=self.id,
                      dotted_order=order if root else self.root_order + '.' + order,
                      start_time=now, inputs=inputs or {}, outputs={},
                      extra={'metadata': {**self.metadata, **(metadata or {})}})
        if not root:
            record['parent_run_id'] = self.id
        self.records.append(record)
        start = perf_counter()
        try:
            yield record
        except Exception as exc:
            # No message, traceback, provider body, prompt, or credential export.
            record['error'] = type(exc).__name__
            raise
        finally:
            record['end_time'] = datetime.now(timezone.utc)
            record['extra']['metadata']['latency_ms'] = round((perf_counter() - start) * 1000, 3)
            if root:
                self.exporter.submit(self.records)


class Tracer:
    def __init__(self, *, enabled=False, content_enabled=False, project='tb-chatbot',
                 client_factory=None, queue_size=32):
        self.enabled = enabled
        self.content_enabled = enabled and content_enabled
        self.project = project
        self.client_factory = client_factory
        self.queue = Queue(maxsize=queue_size)
        self.stop = Event()
        self.exported = 0
        self.failed = 0
        self.dropped = 0
        self.worker = None
        if enabled:
            self.worker = Thread(target=self._work, name='langsmith-export', daemon=True)
            self.worker.start()

    @classmethod
    def from_env(cls):
        if os.getenv('LANGSMITH_TRACING', 'false').lower() != 'true':
            return cls()
        key = os.getenv('LANGSMITH_API_KEY', '').strip()
        if not key:
            logger.warning('LangSmith tracing disabled: API key is missing.')
            return cls()
        endpoint = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com').strip()
        # Restrict outbound credentials to the supported hosted regions.
        if endpoint not in ('https://api.smith.langchain.com', 'https://eu.api.smith.langchain.com'):
            logger.warning('LangSmith tracing disabled: unsupported endpoint.')
            return cls()

        def client_factory():
            from langsmith import Client
            from urllib3.util.retry import Retry
            import ssl
            import requests
            import truststore
            from requests.adapters import HTTPAdapter

            class SystemTrustAdapter(HTTPAdapter):
                def build_connection_pool_key_attributes(self, request, verify, cert=None):
                    host, options = super().build_connection_pool_key_attributes(request, verify, cert)
                    options['ssl_context'] = tls_context
                    return host, options

            # Use OS certificate roots without disabling verification or changing
            # SSL behavior globally (e.g. for the separate OpenAI client).
            tls_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            session = requests.Session()
            client = Client(api_url=endpoint, api_key=key, auto_batch_tracing=False,
                          session=session,
                          timeout_ms=(1000, 2000), retry_config=Retry(total=0),
                          omit_traced_runtime_info=True, tracing_sampling_rate=1.0,
                          hide_inputs=False, hide_outputs=False)
            # The SDK mounts its own adapters during construction; install ours last.
            session.mount('https://', SystemTrustAdapter(max_retries=Retry(total=0)))
            return client
        try:
            return cls(enabled=True,
                       content_enabled=os.getenv('TB_TRACE_CONTENT', 'false').lower() == 'true',
                       project=os.getenv('LANGSMITH_PROJECT', 'tb-chatbot').strip() or 'tb-chatbot',
                       client_factory=client_factory)
        except Exception:
            logger.warning('LangSmith tracing initialization failed; continuing without tracing.')
            return cls()

    def trace(self, request_id, metadata):
        return Trace(self, request_id, metadata)

    def submit(self, records):
        if not self.enabled:
            return
        try:
            if self.stop.is_set():
                self.dropped += 1
                return
            self.queue.put_nowait(records)
        except Full:
            self.dropped += 1
            logger.warning('Trace queue full; trace dropped.')
        except Exception:
            self.failed += 1
            logger.warning('Trace submission failed; chat unaffected.')

    def _work(self):
        client = None
        try:
            while not self.stop.is_set():
                try:
                    records = self.queue.get(timeout=.1)
                except Empty:
                    continue
                try:
                    if client is None:
                        client = self.client_factory()
                    for record in records:
                        if self.stop.is_set():
                            self.dropped += 1
                            break
                        client.create_run(project_name=self.project, **record)
                    else:
                        self.exported += 1
                except Exception:
                    self.failed += 1
                    logger.warning('LangSmith export failed; chat unaffected. Check credentials/connectivity.')
                finally:
                    self.queue.task_done()
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def status(self):
        return {'enabled': self.enabled, 'content_logging': self.content_enabled,
                'pending': self.queue.qsize(), 'exported': self.exported,
                'failed': self.failed, 'dropped': self.dropped}

    def close(self):
        # Bounded shutdown, not a durable delivery guarantee. Worker is a daemon.
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=2.0)
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.dropped += 1
            except Empty:
                break
