"""Unit tests for Slack service."""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.slack_service import SlackService


class TestSlackService:
    """Test cases for SlackService."""
    
    @pytest.fixture
    def slack_service(self):
        """Create SlackService instance for testing."""
        return SlackService()
    
    @pytest.mark.asyncio
    async def test_send_message_basic(self, slack_service):
        """Test basic message sending."""
        with patch('app.services.slack_service.AsyncWebClient') as mock_client:
            # Mock the client and its methods
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock successful response
            mock_client_instance.chat_postMessage.return_value = {
                "ok": True,
                "ts": "1234567890.123456"
            }
            
            # Test sending message
            result = await slack_service.send_message(
                channel="C1234567890",
                text="Test message",
                token="test_token"
            )
            
            # Verify result
            assert result == "1234567890.123456"
            mock_client_instance.chat_postMessage.assert_called_once_with(
                channel="C1234567890",
                text="Test message"
            )
    
    @pytest.mark.asyncio
    async def test_send_message_with_blocks(self, slack_service):
        """Test message sending with blocks."""
        with patch('app.services.slack_service.AsyncWebClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            mock_client_instance.chat_postMessage.return_value = {
                "ok": True,
                "ts": "1234567890.123456"
            }
            
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Test message with blocks"
                    }
                }
            ]
            
            result = await slack_service.send_message(
                channel="C1234567890",
                text="Test message",
                blocks=blocks,
                token="test_token"
            )
            
            assert result == "1234567890.123456"
            mock_client_instance.chat_postMessage.assert_called_once_with(
                channel="C1234567890",
                text="Test message",
                blocks=blocks
            )
    
    @pytest.mark.asyncio
    async def test_send_message_with_thread(self, slack_service):
        """Test message sending in thread."""
        with patch('app.services.slack_service.AsyncWebClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            mock_client_instance.chat_postMessage.return_value = {
                "ok": True,
                "ts": "1234567890.123456"
            }
            
            result = await slack_service.send_message(
                channel="C1234567890",
                text="Test message",
                thread_ts="1234567890.000000",
                token="test_token"
            )
            
            assert result == "1234567890.123456"
            mock_client_instance.chat_postMessage.assert_called_once_with(
                channel="C1234567890",
                text="Test message",
                thread_ts="1234567890.000000"
            )
    
    @pytest.mark.asyncio
    async def test_send_message_failure(self, slack_service):
        """Test message sending failure."""
        with patch('app.services.slack_service.AsyncWebClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock failed response
            mock_client_instance.chat_postMessage.return_value = {
                "ok": False,
                "error": "channel_not_found"
            }
            
            result = await slack_service.send_message(
                channel="C1234567890",
                text="Test message",
                token="test_token"
            )
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_user_info(self, slack_service):
        """Test getting user info."""
        with patch('app.services.slack_service.AsyncWebClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            mock_client_instance.users_info.return_value = {
                "ok": True,
                "user": {
                    "id": "U1234567890",
                    "name": "testuser",
                    "real_name": "Test User"
                }
            }
            
            result = await slack_service.get_user_info(
                user_id="U1234567890",
                token="test_token"
            )
            
            assert result["id"] == "U1234567890"
            assert result["name"] == "testuser"
            assert result["real_name"] == "Test User"
    
    @pytest.mark.asyncio
    async def test_get_channel_info(self, slack_service):
        """Test getting channel info."""
        with patch('app.services.slack_service.AsyncWebClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            mock_client_instance.conversations_info.return_value = {
                "ok": True,
                "channel": {
                    "id": "C1234567890",
                    "name": "test-channel",
                    "is_channel": True
                }
            }
            
            result = await slack_service.get_channel_info(
                channel_id="C1234567890",
                token="test_token"
            )
            
            assert result["id"] == "C1234567890"
            assert result["name"] == "test-channel"
            assert result["is_channel"] is True
