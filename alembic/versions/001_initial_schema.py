"""Initial schema migration.

Revision ID: 001
Revises: 
Create Date: 2025-01-28 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create workspace table
    op.create_table('workspace',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slack_id', sa.String(length=50), nullable=False),
        sa.Column('tokens', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workspace_id'), 'workspace', ['id'], unique=False)
    op.create_index(op.f('ix_workspace_slack_id'), 'workspace', ['slack_id'], unique=True)
    
    # Create user table
    op.create_table('user',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('slack_id', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_id'), 'user', ['id'], unique=False)
    op.create_index(op.f('ix_user_workspace_id'), 'user', ['workspace_id'], unique=False)
    op.create_index(op.f('ix_user_slack_id'), 'user', ['slack_id'], unique=False)
    
    # Create message table
    op.create_table('message',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('channel_id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('raw_payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_message_id'), 'message', ['id'], unique=False)
    op.create_index(op.f('ix_message_workspace_id'), 'message', ['workspace_id'], unique=False)
    op.create_index(op.f('ix_message_channel_id'), 'message', ['channel_id'], unique=False)
    op.create_index(op.f('ix_message_user_id'), 'message', ['user_id'], unique=False)
    
    # Create knowledgeitem table
    op.create_table('knowledgeitem',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('embedding', sa.Text(), nullable=True),  # Will be converted to vector later
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledgeitem_id'), 'knowledgeitem', ['id'], unique=False)
    op.create_index(op.f('ix_knowledgeitem_workspace_id'), 'knowledgeitem', ['workspace_id'], unique=False)
    
    # Create query table
    op.create_table('query',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('response', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_query_id'), 'query', ['id'], unique=False)
    op.create_index(op.f('ix_query_workspace_id'), 'query', ['workspace_id'], unique=False)
    op.create_index(op.f('ix_query_user_id'), 'query', ['user_id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_query_user_id'), table_name='query')
    op.drop_index(op.f('ix_query_workspace_id'), table_name='query')
    op.drop_index(op.f('ix_query_id'), table_name='query')
    op.drop_table('query')
    
    op.drop_index(op.f('ix_knowledgeitem_workspace_id'), table_name='knowledgeitem')
    op.drop_index(op.f('ix_knowledgeitem_id'), table_name='knowledgeitem')
    op.drop_table('knowledgeitem')
    
    op.drop_index(op.f('ix_message_user_id'), table_name='message')
    op.drop_index(op.f('ix_message_channel_id'), table_name='message')
    op.drop_index(op.f('ix_message_workspace_id'), table_name='message')
    op.drop_index(op.f('ix_message_id'), table_name='message')
    op.drop_table('message')
    
    op.drop_index(op.f('ix_user_slack_id'), table_name='user')
    op.drop_index(op.f('ix_user_workspace_id'), table_name='user')
    op.drop_index(op.f('ix_user_id'), table_name='user')
    op.drop_table('user')
    
    op.drop_index(op.f('ix_workspace_slack_id'), table_name='workspace')
    op.drop_index(op.f('ix_workspace_id'), table_name='workspace')
    op.drop_table('workspace')
