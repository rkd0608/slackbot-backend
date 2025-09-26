"""Unit tests for enhanced thread detection logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from app.api.v1.slack import should_bot_respond_in_thread


class TestThreadDetection:
    """Test cases for enhanced thread detection logic."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def mock_conversation(self):
        """Create mock conversation."""
        conversation = MagicMock()
        conversation.id = 1
        conversation.workspace_id = 1
        conversation.slack_channel_id = "C123456"
        return conversation

    @pytest.fixture
    def mock_bot_message(self):
        """Create mock bot message."""
        message = MagicMock()
        message.slack_user_id = "B123456"  # Bot user ID
        message.content = "I can help with that!"
        message.slack_thread_ts = "1234567890.123456"
        return message

    @pytest.fixture
    def mock_user_message_with_reno(self):
        """Create mock user message mentioning reno."""
        message = MagicMock()
        message.slack_user_id = "U123456"  # User ID
        message.content = "Hey @reno, can you help me?"
        message.slack_thread_ts = "1234567890.123456"
        return message

    @pytest.mark.asyncio
    async def test_should_not_respond_outside_thread(self, mock_db_session):
        """Test that bot doesn't respond outside of threads."""
        result = await should_bot_respond_in_thread(
            text="How do I deploy this?",
            thread_ts=None,  # No thread
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_not_respond_to_short_messages(self, mock_db_session):
        """Test that bot doesn't respond to very short messages."""
        result = await should_bot_respond_in_thread(
            text="ok",  # Too short
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_not_respond_to_emoji_only(self, mock_db_session):
        """Test that bot doesn't respond to emoji-only messages."""
        result = await should_bot_respond_in_thread(
            text="👍",  # Emoji only
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_not_respond_when_other_users_mentioned(self, mock_db_session, mock_conversation):
        """Test that bot doesn't respond when other users are mentioned."""
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = []
        
        result = await should_bot_respond_in_thread(
            text="Hey @john, can you help me with this?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_not_respond_when_bot_not_participated(self, mock_db_session, mock_conversation):
        """Test that bot doesn't respond when it hasn't participated in the thread."""
        # Mock database responses - no bot participation
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = []
        
        result = await should_bot_respond_in_thread(
            text="How do I deploy this?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_respond_to_questions_in_bot_thread(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to questions in threads where it has participated."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="How do I deploy this application?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_to_conversational_responses(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to conversational responses in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="Thanks for the help!",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_to_direct_address(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to direct address in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="You mentioned something about deployment earlier",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_to_work_related_content(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to work-related content in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="The database migration is complete and the deployment went smoothly",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_when_reno_mentioned_in_thread(self, mock_db_session, mock_conversation, mock_user_message_with_reno):
        """Test that bot responds when reno was mentioned in the thread previously."""
        # Mock database responses - reno was mentioned in thread
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_user_message_with_reno]
        
        result = await should_bot_respond_in_thread(
            text="What about the next steps?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_not_respond_to_irrelevant_content(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot doesn't respond to irrelevant content even in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="The weather is nice today",  # Not work-related
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_respond_to_follow_up_questions(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to follow-up questions in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="What about the database configuration?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_to_status_inquiries(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to status inquiries in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="What's the status of the deployment?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_to_positive_feedback(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to positive feedback in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="Perfect! That worked great",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_to_seeking_more_info(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds when user is seeking more information."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="What else should I know about this?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_not_respond_when_no_conversation(self, mock_db_session):
        """Test that bot doesn't respond when no conversation exists."""
        # Mock database responses - no conversation
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        result = await should_bot_respond_in_thread(
            text="How do I deploy this?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_handle_database_errors_gracefully(self, mock_db_session):
        """Test that bot handles database errors gracefully."""
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database error")
        
        result = await should_bot_respond_in_thread(
            text="How do I deploy this?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_respond_to_technical_questions(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to technical questions in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="How do I fix the SSL certificate issue?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_to_decision_questions(self, mock_db_session, mock_conversation, mock_bot_message):
        """Test that bot responds to decision-related questions in its threads."""
        # Mock database responses - bot has participated
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalars.return_value.__iter__.return_value = [mock_bot_message]
        
        result = await should_bot_respond_in_thread(
            text="What was the decision about the database migration?",
            thread_ts="1234567890.123456",
            channel_id="C123456",
            workspace_id=1,
            db=mock_db_session
        )
        assert result is True
