"""Database ownership and lifecycle; migrations are an explicit setup step."""
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Text, DateTime, ForeignKey, JSON, Integer, CheckConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class BrowserSession(Base):
    __tablename__ = 'browser_sessions'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class Conversation(Base):
    __tablename__ = 'conversations'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey('browser_sessions.id', ondelete='CASCADE'), index=True)
    title: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class Message(Base):
    """One question/answer exchange; UUID is the client idempotency key."""
    __tablename__ = 'messages'
    __table_args__ = (CheckConstraint("status IN ('pending', 'completed', 'failed')", name='message_status'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey('conversations.id', ondelete='CASCADE'), index=True)
    question: Mapped[str] = mapped_column(Text)
    retriever: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(12))
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[str] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Feedback(Base):
    __tablename__ = 'feedback'
    __table_args__ = (CheckConstraint('rating IN (-1, 1)', name='feedback_rating'),)
    message_id: Mapped[str] = mapped_column(ForeignKey('messages.id', ondelete='CASCADE'), primary_key=True)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str] = mapped_column(String(2000), default='')
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Database:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True, hide_parameters=True,
                                   connect_args={'connect_timeout': 5})
        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def close(self):
        self.engine.dispose()


def from_env():
    url = os.getenv('DATABASE_URL', '').strip()
    return Database(url) if url else None
