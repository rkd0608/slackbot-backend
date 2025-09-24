"""Enhanced conversation state management

Revision ID: 002_enhanced_conversation_state
Revises: 001_initial_schema
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_enhanced_conversation_state'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the old unique constraint
    op.drop_constraint('uq_conversation_workspace_channel', 'conversation', schema='public')
    
    # Add new columns for enhanced conversation state management
    op.add_column('conversation', sa.Column('thread_timestamp', sa.String(50), nullable=True), schema='public')
    op.add_column('conversation', sa.Column('topic', sa.String(255), nullable=True), schema='public')
    op.add_column('conversation', sa.Column('state', sa.String(50), nullable=False, server_default='developing'), schema='public')
    op.add_column('conversation', sa.Column('state_confidence', sa.Float(), nullable=False, server_default='0.5'), schema='public')
    op.add_column('conversation', sa.Column('state_updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()), schema='public')
    op.add_column('conversation', sa.Column('participant_count', sa.Integer(), nullable=False, server_default='0'), schema='public')
    op.add_column('conversation', sa.Column('message_count', sa.Integer(), nullable=False, server_default='0'), schema='public')
    op.add_column('conversation', sa.Column('resolution_indicators', postgresql.JSONB(), nullable=True), schema='public')
    op.add_column('conversation', sa.Column('is_ready_for_extraction', sa.Boolean(), nullable=False, server_default='false'), schema='public')
    op.add_column('conversation', sa.Column('extraction_attempted_at', sa.DateTime(timezone=True), nullable=True), schema='public')
    op.add_column('conversation', sa.Column('extraction_completed_at', sa.DateTime(timezone=True), nullable=True), schema='public')
    
    # Make some existing columns nullable for more flexible conversation grouping
    op.alter_column('conversation', 'slack_channel_name', nullable=True, schema='public')
    op.alter_column('conversation', 'title', nullable=True, schema='public')
    
    # Create new unique constraint that includes thread_timestamp
    op.create_unique_constraint(
        'uq_conversation_workspace_channel_thread', 
        'conversation', 
        ['workspace_id', 'slack_channel_id', 'thread_timestamp'],
        schema='public'
    )
    
    # Create indexes for better query performance
    op.create_index('ix_conversation_thread_timestamp', 'conversation', ['thread_timestamp'], schema='public')
    op.create_index('ix_conversation_state', 'conversation', ['state'], schema='public')
    op.create_index('ix_conversation_ready_extraction', 'conversation', ['is_ready_for_extraction'], schema='public')
    op.create_index('ix_conversation_state_updated', 'conversation', ['state_updated_at'], schema='public')


def downgrade() -> None:
    # Drop new indexes
    op.drop_index('ix_conversation_state_updated', 'conversation', schema='public')
    op.drop_index('ix_conversation_ready_extraction', 'conversation', schema='public')
    op.drop_index('ix_conversation_state', 'conversation', schema='public')
    op.drop_index('ix_conversation_thread_timestamp', 'conversation', schema='public')
    
    # Drop new unique constraint
    op.drop_constraint('uq_conversation_workspace_channel_thread', 'conversation', schema='public')
    
    # Remove new columns
    op.drop_column('conversation', 'extraction_completed_at', schema='public')
    op.drop_column('conversation', 'extraction_attempted_at', schema='public')
    op.drop_column('conversation', 'is_ready_for_extraction', schema='public')
    op.drop_column('conversation', 'resolution_indicators', schema='public')
    op.drop_column('conversation', 'message_count', schema='public')
    op.drop_column('conversation', 'participant_count', schema='public')
    op.drop_column('conversation', 'state_updated_at', schema='public')
    op.drop_column('conversation', 'state_confidence', schema='public')
    op.drop_column('conversation', 'state', schema='public')
    op.drop_column('conversation', 'topic', schema='public')
    op.drop_column('conversation', 'thread_timestamp', schema='public')
    
    # Restore original column constraints
    op.alter_column('conversation', 'title', nullable=False, schema='public')
    op.alter_column('conversation', 'slack_channel_name', nullable=False, schema='public')
    
    # Restore original unique constraint
    op.create_unique_constraint(
        'uq_conversation_workspace_channel',
        'conversation',
        ['workspace_id', 'slack_channel_id'],
        schema='public'
    )
