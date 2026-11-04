"""Anonymous browser history with cascading deletion."""
from alembic import op
import sqlalchemy as sa

revision = '0001_history'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('browser_sessions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('token_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('ix_browser_sessions_expires_at', 'browser_sessions', ['expires_at'])
    op.create_table('conversations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('session_id', sa.String(36), sa.ForeignKey('browser_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(120), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('ix_conversations_session_id', 'conversations', ['session_id'])
    op.create_index('ix_conversations_updated_at', 'conversations', ['updated_at'])
    op.create_table('messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question', sa.Text, nullable=False),
        sa.Column('retriever', sa.String(20), nullable=False),
        sa.Column('status', sa.String(12), nullable=False),
        sa.Column('attempt', sa.Integer, nullable=False),
        sa.Column('result', sa.JSON, nullable=True),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('request_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('pending', 'completed', 'failed')", name='message_status'))
    op.create_index('ix_messages_conversation_id', 'messages', ['conversation_id'])
    op.create_table('feedback',
        sa.Column('message_id', sa.String(36), sa.ForeignKey('messages.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('rating', sa.Integer, nullable=False),
        sa.Column('comment', sa.String(2000), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('rating IN (-1, 1)', name='feedback_rating'))


def downgrade():
    for table in ('feedback', 'messages', 'conversations', 'browser_sessions'):
        op.drop_table(table)
