"""Synchronous orchestration, independent of either web interface."""
from threading import Lock

from .config import DPR_MODELS, OPENAI_MODEL
from .context import build_context


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

    def __init__(self, retriever_factory=load_retriever, generator_factory=load_generator):
        self.retriever_factory = retriever_factory
        self.generator_factory = generator_factory
        self._retriever = None
        self._retriever_name = None
        self._generator = None
        self._lock = Lock()

    def chat(self, question, retriever_name):
        if retriever_name not in DPR_MODELS:
            raise ValueError('Unknown retriever')
        if not self._lock.acquire(blocking=False):
            raise ServiceBusy()
        try:
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
            try:
                passages = self._retriever.search(question, k=5)
                context = build_context(passages)
            except Exception as exc:
                raise ModelUnavailable() from exc
            answer = self._generator.generate(question, context)
            if not isinstance(answer, str) or not answer.strip():
                raise GenerationUnavailable()
            return {'answer': answer, 'retriever': retriever_name,
                    'generation_model': OPENAI_MODEL, 'passages': passages}
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if self._generator is not None:
                self._generator.close()
            self._generator = None
            self._retriever = None
            self._retriever_name = None
