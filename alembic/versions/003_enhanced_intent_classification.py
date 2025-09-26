"""Enhanced intent classification models

Revision ID: 003_enhanced_intent_classification
Revises: 002_enhanced_conversation_state
Create Date: 2025-01-25 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_intent_classification'
down_revision = '002_enhanced_conversation_state'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create user_communication_profile table
    op.create_table('user_communication_profile',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('formality_level', sa.Float(), nullable=False, default=0.5),
        sa.Column('verbosity_preference', sa.Float(), nullable=False, default=0.5),
        sa.Column('emoji_usage_frequency', sa.Float(), nullable=False, default=0.0),
        sa.Column('question_asking_frequency', sa.Float(), nullable=False, default=0.0),
        sa.Column('common_greetings', postgresql.ARRAY(sa.String()), nullable=True, default=list),
        sa.Column('common_phrases', postgresql.ARRAY(sa.String()), nullable=True, default=list),
        sa.Column('preferred_response_style', sa.String(length=50), nullable=True, default='balanced'),
        sa.Column('communication_culture', sa.String(length=100), nullable=True, default='general'),
        sa.Column('timezone_preference', sa.String(length=50), nullable=True),
        sa.Column('language_preference', sa.String(length=10), nullable=True, default='en'),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('learning_confidence', sa.Float(), nullable=False, default=0.0),
        sa.Column('interaction_count', sa.Integer(), nullable=False, default=0),
        sa.ForeignKeyConstraint(['user_id'], ['public.user.id'], ),
        sa.ForeignKeyConstraint(['workspace_id'], ['public.workspace.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'workspace_id', name='uq_user_workspace_profile'),
        schema='public'
    )
    op.create_index(op.f('ix_public_user_communication_profile_user_id'), 'user_communication_profile', ['user_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_user_communication_profile_workspace_id'), 'user_communication_profile', ['workspace_id'], unique=False, schema='public')

    # Create channel_culture table
    op.create_table('channel_culture',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('channel_id', sa.String(length=50), nullable=False),
        sa.Column('channel_name', sa.String(length=255), nullable=False),
        sa.Column('formality_level', sa.Float(), nullable=False, default=0.5),
        sa.Column('topic_focus', sa.String(length=100), nullable=True),
        sa.Column('response_expectations', sa.String(length=50), nullable=True, default='balanced'),
        sa.Column('common_topics', postgresql.ARRAY(sa.String()), nullable=True, default=list),
        sa.Column('common_phrases', postgresql.ARRAY(sa.String()), nullable=True, default=list),
        sa.Column('active_participants', postgresql.ARRAY(sa.String()), nullable=True, default=list),
        sa.Column('channel_purpose', sa.String(length=255), nullable=True),
        sa.Column('communication_norms', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('last_analyzed', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('analysis_confidence', sa.Float(), nullable=False, default=0.0),
        sa.Column('message_count_analyzed', sa.Integer(), nullable=False, default=0),
        sa.ForeignKeyConstraint(['workspace_id'], ['public.workspace.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('workspace_id', 'channel_id', name='uq_workspace_channel_culture'),
        schema='public'
    )
    op.create_index(op.f('ix_public_channel_culture_workspace_id'), 'channel_culture', ['workspace_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_channel_culture_channel_id'), 'channel_culture', ['channel_id'], unique=False, schema='public')

    # Create conversation_context table
    op.create_table('conversation_context',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('channel_id', sa.String(length=50), nullable=False),
        sa.Column('thread_ts', sa.String(length=50), nullable=True),
        sa.Column('active_topic', sa.String(length=255), nullable=True),
        sa.Column('conversation_stage', sa.String(length=50), nullable=False, default='starting'),
        sa.Column('participant_count', sa.Integer(), nullable=False, default=1),
        sa.Column('message_count', sa.Integer(), nullable=False, default=0),
        sa.Column('last_bot_response', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_user_message', sa.DateTime(timezone=True), nullable=True),
        sa.Column('conversation_sentiment', sa.Float(), nullable=True),
        sa.Column('recent_messages', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=list),
        sa.Column('key_participants', postgresql.ARRAY(sa.String()), nullable=True, default=list),
        sa.Column('conversation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.ForeignKeyConstraint(['workspace_id'], ['public.workspace.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='public'
    )
    op.create_index(op.f('ix_public_conversation_context_workspace_id'), 'conversation_context', ['workspace_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_conversation_context_channel_id'), 'conversation_context', ['channel_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_conversation_context_thread_ts'), 'conversation_context', ['thread_ts'], unique=False, schema='public')

    # Create intent_classification_history table
    op.create_table('intent_classification_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('channel_id', sa.String(length=50), nullable=False),
        sa.Column('query_id', sa.Integer(), nullable=True),
        sa.Column('original_message', sa.Text(), nullable=False),
        sa.Column('classified_intent', sa.String(length=100), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('classification_method', sa.String(length=50), nullable=False),
        sa.Column('conversation_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('user_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('channel_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('response_generated', sa.Boolean(), nullable=False, default=False),
        sa.Column('user_satisfaction', sa.Float(), nullable=True),
        sa.Column('response_effectiveness', sa.Float(), nullable=True),
        sa.Column('follow_up_required', sa.Boolean(), nullable=False, default=False),
        sa.Column('was_correct', sa.Boolean(), nullable=True),
        sa.Column('correction_applied', sa.Boolean(), nullable=False, default=False),
        sa.Column('learning_notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['query_id'], ['public.query.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['public.user.id'], ),
        sa.ForeignKeyConstraint(['workspace_id'], ['public.workspace.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='public'
    )
    op.create_index(op.f('ix_public_intent_classification_history_workspace_id'), 'intent_classification_history', ['workspace_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_intent_classification_history_user_id'), 'intent_classification_history', ['user_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_intent_classification_history_channel_id'), 'intent_classification_history', ['channel_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_intent_classification_history_query_id'), 'intent_classification_history', ['query_id'], unique=False, schema='public')

    # Create response_effectiveness table
    op.create_table('response_effectiveness',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('query_id', sa.Integer(), nullable=False),
        sa.Column('response_type', sa.String(length=50), nullable=False),
        sa.Column('response_style', sa.String(length=50), nullable=False),
        sa.Column('response_length', sa.Integer(), nullable=False),
        sa.Column('user_rating', sa.Float(), nullable=True),
        sa.Column('response_time', sa.Float(), nullable=True),
        sa.Column('follow_up_questions', sa.Integer(), nullable=False, default=0),
        sa.Column('resolution_achieved', sa.Boolean(), nullable=False, default=False),
        sa.Column('intent_confidence', sa.Float(), nullable=False),
        sa.Column('conversation_stage', sa.String(length=50), nullable=False),
        sa.Column('time_of_day', sa.String(length=20), nullable=True),
        sa.Column('was_helpful', sa.Boolean(), nullable=True),
        sa.Column('improvement_suggestions', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['query_id'], ['public.query.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['public.user.id'], ),
        sa.ForeignKeyConstraint(['workspace_id'], ['public.workspace.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='public'
    )
    op.create_index(op.f('ix_public_response_effectiveness_workspace_id'), 'response_effectiveness', ['workspace_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_response_effectiveness_user_id'), 'response_effectiveness', ['user_id'], unique=False, schema='public')
    op.create_index(op.f('ix_public_response_effectiveness_query_id'), 'response_effectiveness', ['query_id'], unique=False, schema='public')


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_public_response_effectiveness_query_id'), table_name='response_effectiveness', schema='public')
    op.drop_index(op.f('ix_public_response_effectiveness_user_id'), table_name='response_effectiveness', schema='public')
    op.drop_index(op.f('ix_public_response_effectiveness_workspace_id'), table_name='response_effectiveness', schema='public')
    op.drop_table('response_effectiveness', schema='public')
    
    op.drop_index(op.f('ix_public_intent_classification_history_query_id'), table_name='intent_classification_history', schema='public')
    op.drop_index(op.f('ix_public_intent_classification_history_channel_id'), table_name='intent_classification_history', schema='public')
    op.drop_index(op.f('ix_public_intent_classification_history_user_id'), table_name='intent_classification_history', schema='public')
    op.drop_index(op.f('ix_public_intent_classification_history_workspace_id'), table_name='intent_classification_history', schema='public')
    op.drop_table('intent_classification_history', schema='public')
    
    op.drop_index(op.f('ix_public_conversation_context_thread_ts'), table_name='conversation_context', schema='public')
    op.drop_index(op.f('ix_public_conversation_context_channel_id'), table_name='conversation_context', schema='public')
    op.drop_index(op.f('ix_public_conversation_context_workspace_id'), table_name='conversation_context', schema='public')
    op.drop_table('conversation_context', schema='public')
    
    op.drop_index(op.f('ix_public_channel_culture_channel_id'), table_name='channel_culture', schema='public')
    op.drop_index(op.f('ix_public_channel_culture_workspace_id'), table_name='channel_culture', schema='public')
    op.drop_table('channel_culture', schema='public')
    
    op.drop_index(op.f('ix_public_user_communication_profile_workspace_id'), table_name='user_communication_profile', schema='public')
    op.drop_index(op.f('ix_public_user_communication_profile_user_id'), table_name='user_communication_profile', schema='public')
    op.drop_table('user_communication_profile', schema='public')
