"""Synchronous orchestration, independent of either web interface."""
from threading import Lock
from uuid import uuid4
from pathlib import Path

from .config import DPR_MODELS, OPENAI_MODEL
from .context import build_context
from .observability import Tracer, file_version, passage_summary


class ServiceBusy(Exception):
    pass


class ModelUnavailable(Exception):
    pass


class GenerationUnavailable(Exception):
    pass


def load_retriever(name):
    from .dpr_retriever import DenseRetriever
    return DenseRetriever(name)


def load_generator():
    from .generator import AnswerGenerator
    return AnswerGenerator()


class ChatService:
    """One active request and one cached retriever per server process.

    Lazy loading keeps health/docs available without weights or API credentials.
    Switching variants evicts the previous retriever to bound model memory usage.
    """

    def __init__(self, retriever_factory=load_retriever, generator_factory=load_generator, tracer=None):
        self.tracer = tracer if tracer is not None else Tracer.from_env()
        self.retriever_factory = retriever_factory
        self.generator_factory = generator_factory
        self._retriever = None
        self._retriever_name = None
        self._generator = None
        self._lock = Lock()

    def chat(self, question, retriever_name, request_id=None):
        request_id = request_id or str(uuid4())
        trace = self.tracer.trace(request_id, {'request_id': request_id,
                                              'retriever': retriever_name,
                                              'generation_model': OPENAI_MODEL,
                                              'prompt_version': 'tb-rag-v1'})
        with trace.span('chat', root=True, inputs=trace.content(question=question)) as root:
            result = self._chat(question, retriever_name, trace)
            trace.annotate(root, lambda: {'outputs': trace.content(answer=result['answer'])})
        return {**result, 'request_id': request_id,
                'trace_id': request_id if self.tracer.enabled else None}

    def _chat(self, question, retriever_name, trace):
        if retriever_name not in DPR_MODELS:
            raise ValueError('Unknown retriever')
        if not self._lock.acquire(blocking=False):
            raise ServiceBusy()
        try:
            with trace.span('load_resources') as resources:
                if self._generator is None:
                    try:
                        self._generator = self.generator_factory()
                    except Exception as exc:
                        raise GenerationUnavailable() from exc
                if self._retriever_name != retriever_name:
                    self._retriever = None
                    self._retriever_name = None
                    try:
                        self._retriever = self.retriever_factory(retriever_name)
                        self._retriever_name = retriever_name
                    except Exception as exc:
                        raise ModelUnavailable() from exc
                trace.annotate(resources, lambda: {'outputs': {
                    'index_sha256': file_version(DPR_MODELS[retriever_name]['index']),
                    'corpus_sha256': file_version(DPR_MODELS[retriever_name]['passages']),
                    'generator_sha256': file_version(str(Path(__file__).with_name('generator.py'))),
                    'question_encoder': DPR_MODELS[retriever_name]['query']}})
            try:
                with trace.span('retrieval', 'retriever', inputs=trace.content(query=question)) as retrieval:
                    passages = self._retriever.search(question, k=5)
                    trace.annotate(retrieval, lambda: {'outputs': {
                        'passages': passage_summary(passages, self._retriever.passages),
                        **trace.content(documents=passages)}})
                with trace.span('context_construction') as context_run:
                    context = build_context(passages)
                    # Same prefix rule as build_context: no skip/reordering on overflow.
                    trace.annotate(context_run, lambda: {'outputs': {
                        'characters': len(context), 'limit': 3500,
                        'included_ranks': self._included_ranks(passages),
                        **trace.content(context=context)}})
            except Exception as exc:
                raise ModelUnavailable() from exc
            with trace.span('generation', 'llm', inputs=trace.content(question=question, context=context),
                            metadata={'ls_provider': 'openai', 'ls_model_name': OPENAI_MODEL,
                                      'ls_model_type': 'chat', 'ls_temperature': 0.3,
                                      'ls_max_tokens': 600}) as generation:
                answer = self._generator.generate(question, context)
                if not isinstance(answer, str) or not answer.strip():
                    raise GenerationUnavailable()
                trace.annotate(generation, lambda: self._generation_trace(answer, trace))
            return {'answer': answer, 'retriever': retriever_name,
                    'generation_model': OPENAI_MODEL, 'passages': passages}
        finally:
            self._lock.release()

    @staticmethod
    def _included_ranks(passages):
        total, ranks = 0, []
        for i, p in enumerate(passages, 1):
            total += len(f"[{p['source']} - page {p['page']}]\n{p['text']}\n\n")
            if total > 3500:
                break
            ranks.append(i)
        return ranks

    def _generation_trace(self, answer, trace):
        usage = getattr(self._generator, 'last_usage', {})
        outputs = trace.content(messages=[{'role': 'assistant', 'content': answer}])
        if isinstance(usage, dict) and usage:
            outputs['usage_metadata'] = usage
        result = {'outputs': outputs}
        messages = getattr(self._generator, 'last_messages', None)
        if self.tracer.content_enabled and isinstance(messages, list):
            result['inputs'] = {'messages': messages}
        return result

    def close(self):
        self.tracer.close()
        with self._lock:
            if self._generator is not None:
                self._generator.close()
            self._generator = None
            self._retriever = None
            self._retriever_name = None
