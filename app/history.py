"""Anonymous history API. Cookie credentials never enter traces or responses."""
import hashlib
import os
import secrets
from datetime import timedelta
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, select

from .database import BrowserSession, Conversation, Feedback, Message, now

COOKIE = 'tb_session'
LIFETIME = timedelta(days=30)
STALE_AFTER = timedelta(minutes=10)


class Title(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(min_length=1, max_length=120)

    @field_validator('title')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('Title cannot be blank')
        return value.strip()


class Submission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=4000, strict=True)
    retriever: Literal['single_nq', 'multiset', 'lfqa'] = 'single_nq'
    attempt: int = Field(default=0, ge=0, strict=True)

    @field_validator('question')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('Question cannot be blank')
        return value


class Rating(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rating: Literal[-1, 1]
    comment: str = Field(default='', max_length=2000)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def cleanup(database):
    if database:
        with database.session.begin() as db:
            db.execute(delete(BrowserSession).where(BrowserSession.expires_at <= now()))


def storage(request: Request):
    database = request.app.state.database
    if database is None:
        raise HTTPException(503, 'History database is not configured.')
    return database


def protect(request: Request):
    # Cross-origin scripts cannot add this header without an allowed CORS preflight.
    if request.method not in ('GET', 'HEAD') and request.headers.get('X-TB-Client') != 'web':
        raise HTTPException(403, 'Missing request protection header.')
    if request.headers.get('Sec-Fetch-Site') == 'cross-site':
        raise HTTPException(403, 'Cross-site history access is not allowed.')


def owner(db, request):
    token = request.cookies.get(COOKIE, '')
    if not token or len(token) > 128:
        raise HTTPException(401, 'Browser session expired. Reconnect to start a new session.')
    session = db.scalar(select(BrowserSession).where(
        BrowserSession.token_hash == token_hash(token), BrowserSession.expires_at > now()
    ).with_for_update())
    if session is None:
        raise HTTPException(401, 'Browser session expired. Reconnect to start a new session.')
    return session


def conversation(db, session, cid):
    item = db.scalar(select(Conversation).where(
        Conversation.id == str(cid), Conversation.session_id == session.id).with_for_update())
    if item is None:
        raise HTTPException(404, 'Conversation not found.')
    return item


def summary(item):
    return dict(id=item.id, title=item.title, created_at=item.created_at, updated_at=item.updated_at)


def expire_pending(message):
    if message.status == 'pending' and message.updated_at < now() - STALE_AFTER:
        message.status = 'failed'
        message.error = 'Processing was interrupted. Retrying may make another provider call.'


def serialize(db, message):
    feedback = db.get(Feedback, message.id)
    return dict(id=message.id, question=message.question, retriever=message.retriever,
                status=message.status, attempt=message.attempt, answer=message.result,
                error=message.error, request_id=message.request_id,
                feedback=dict(rating=feedback.rating, comment=feedback.comment) if feedback else None)


def history_router(generate):
    router = APIRouter(dependencies=[Depends(protect)], tags=['history'])

    @router.post('/session')
    def start_session(request: Request, response: Response, database=Depends(storage)):
        token = request.cookies.get(COOKIE, '')
        with database.session.begin() as db:
            session = db.scalar(select(BrowserSession).where(
                BrowserSession.token_hash == token_hash(token), BrowserSession.expires_at > now())) if token else None
            if session is None:
                token = secrets.token_urlsafe(32)
                session = BrowserSession(id=str(uuid4()), token_hash=token_hash(token), expires_at=now() + LIFETIME)
                db.add(session)
            response.set_cookie(COOKIE, token, httponly=True, samesite='strict', path='/',
                                secure=os.getenv('COOKIE_SECURE', 'false').lower() == 'true',
                                max_age=max(0, int((session.expires_at - now()).total_seconds())))
            return {'expires_at': session.expires_at}

    @router.get('/conversations')
    def list_conversations(request: Request, database=Depends(storage)):
        with database.session.begin() as db:
            session = owner(db, request)
            return [summary(item) for item in db.scalars(select(Conversation).where(
                Conversation.session_id == session.id).order_by(Conversation.updated_at.desc()))]

    @router.put('/conversations/{cid}')
    def create_conversation(cid: UUID, body: Title, request: Request, database=Depends(storage)):
        with database.session.begin() as db:
            session = owner(db, request)
            item = db.get(Conversation, str(cid))
            if item:
                if item.session_id != session.id:
                    raise HTTPException(404, 'Conversation not found.')
            else:
                item = Conversation(id=str(cid), session_id=session.id, title=body.title)
                db.add(item)
                db.flush()
            return summary(item)

    @router.patch('/conversations/{cid}')
    def rename(cid: UUID, body: Title, request: Request, database=Depends(storage)):
        with database.session.begin() as db:
            item = conversation(db, owner(db, request), cid)
            item.title, item.updated_at = body.title, now()
            return summary(item)

    @router.get('/conversations/{cid}')
    def read(cid: UUID, request: Request, database=Depends(storage)):
        with database.session.begin() as db:
            item = conversation(db, owner(db, request), cid)
            messages = list(db.scalars(select(Message).where(Message.conversation_id == item.id).order_by(Message.created_at, Message.id)))
            for message in messages:
                expire_pending(message)
            return {**summary(item), 'messages': [serialize(db, message) for message in messages]}

    @router.delete('/conversations/{cid}')
    def remove(cid: UUID, request: Request, database=Depends(storage)):
        with database.session.begin() as db:
            db.delete(conversation(db, owner(db, request), cid))
        return {'deleted': True}

    @router.delete('/conversations')
    def clear(request: Request, database=Depends(storage)):
        with database.session.begin() as db:
            session = owner(db, request)
            db.execute(delete(Conversation).where(Conversation.session_id == session.id))
        return {'deleted': True}

    @router.put('/conversations/{cid}/messages/{mid}')
    def submit(cid: UUID, mid: UUID, body: Submission, request: Request, database=Depends(storage)):
        # Reserve and commit before generation. No DB locks/connections held during inference.
        with database.session.begin() as db:
            item = conversation(db, owner(db, request), cid)
            message = db.get(Message, str(mid))
            if message:
                if message.conversation_id != item.id:
                    raise HTTPException(404, 'Message not found.')
                if (message.question, message.retriever) != (body.question, body.retriever):
                    raise HTTPException(409, 'Message ID already used for a different question.')
                expire_pending(message)
                if body.attempt <= message.attempt or message.status != 'failed':
                    return serialize(db, message)
                if body.attempt != message.attempt + 1:
                    raise HTTPException(409, 'Reload the conversation before retrying.')
            else:
                if body.attempt != 0:
                    raise HTTPException(409, 'Reload the conversation before retrying.')
                message = Message(id=str(mid), conversation_id=item.id, question=body.question, retriever=body.retriever)
                db.add(message)
            message.status, message.error, message.result = 'pending', None, None
            message.attempt = body.attempt
            message.request_id = request.state.request_id
            message.updated_at = item.updated_at = now()

        try:
            result = generate(body, request)
            status, error = 'completed', None
        except HTTPException as exc:
            result, status, error = None, 'failed', str(exc.detail)

        with database.session.begin() as db:
            # Deletion/expiry during generation must not resurrect records.
            item = conversation(db, owner(db, request), cid)
            message = db.get(Message, str(mid))
            if message is None:
                raise HTTPException(404, 'Message no longer exists.')
            if message.attempt == body.attempt and message.status == 'pending':
                message.result, message.status, message.error = result, status, error
                message.updated_at = item.updated_at = now()
            return serialize(db, message)

    @router.put('/conversations/{cid}/messages/{mid}/feedback')
    def feedback(cid: UUID, mid: UUID, body: Rating, request: Request, database=Depends(storage)):
        with database.session.begin() as db:
            item = conversation(db, owner(db, request), cid)
            message = db.get(Message, str(mid))
            if message is None or message.conversation_id != item.id:
                raise HTTPException(404, 'Message not found.')
            if message.status != 'completed':
                raise HTTPException(409, 'Only completed answers can receive feedback.')
            item = db.get(Feedback, message.id)
            if item is None:
                item = Feedback(message_id=message.id)
                db.add(item)
            item.rating, item.comment, item.updated_at = body.rating, body.comment, now()
            return {'rating': item.rating, 'comment': item.comment}

    return router
