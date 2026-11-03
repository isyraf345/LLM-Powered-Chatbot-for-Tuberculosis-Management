"""Run from the repository root: python -m uvicorn app.api:app."""
from contextlib import asynccontextmanager
import logging
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import OPENAI_API_KEY
from .service import ChatService, GenerationUnavailable, ModelUnavailable, ServiceBusy

logger = logging.getLogger(__name__)
RetrieverName = Literal['single_nq', 'multiset', 'lfqa']


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=4000, strict=True)
    retriever: RetrieverName = 'single_nq'

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Question must contain non-whitespace characters')
        return value


class Passage(BaseModel):
    text: str
    source: str
    page: int | None
    score: float


class ChatResponse(BaseModel):
    answer: str
    retriever: RetrieverName
    generation_model: str
    passages: list[Passage]
    request_id: str
    trace_id: str | None


def create_app(service_factory=ChatService):
    @asynccontextmanager
    async def lifespan(application):
        application.state.chat_service = service_factory()
        try:
            yield
        finally:
            application.state.chat_service.close()

    application = FastAPI(title='TB Chatbot API', version='0.1.0', lifespan=lifespan)

    @application.middleware('http')
    async def request_reference(request: Request, call_next):
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.request_id
        return response

    @application.get('/health')
    def health():
        return {'status': 'ok', 'generation_configured': bool(OPENAI_API_KEY.strip()),
                'model_loading': 'lazy',
                'tracing': application.state.chat_service.tracer.status()}

    @application.get('/retrievers')
    def retrievers():
        return {'default': 'single_nq', 'retrievers': [
            {'id': 'single_nq', 'name': 'Single NQ'},
            {'id': 'multiset', 'name': 'Multiset'},
            {'id': 'lfqa', 'name': 'LFQA'}]}

    @application.post('/chat', response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request):
        try:
            return application.state.chat_service.chat(body.question, body.retriever, request.state.request_id)
        except ServiceBusy:
            raise HTTPException(503, 'Chat service is busy; retry shortly.', headers={'Retry-After': '2'}) from None
        except ModelUnavailable:
            raise HTTPException(503, 'Retrieval model or index is unavailable.') from None
        except (GenerationUnavailable, AuthenticationError):
            raise HTTPException(503, 'Answer generation is unavailable. Check server configuration.') from None
        except APITimeoutError:
            raise HTTPException(504, 'Answer generation timed out.') from None
        except RateLimitError:
            raise HTTPException(503, 'Answer provider is temporarily rate limited.', headers={'Retry-After': '5'}) from None
        except (APIConnectionError, APIStatusError):
            raise HTTPException(502, 'Could not obtain an answer from the provider.') from None
        except Exception as exc:
            # Avoid logging question/context, credentials, or provider error bodies.
            logger.error('Chat failed (%s)', type(exc).__name__)
            raise HTTPException(500, 'The chat request could not be completed.') from None

    return application


app = create_app()
