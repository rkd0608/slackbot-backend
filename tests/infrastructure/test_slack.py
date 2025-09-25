"""Mock Slack service for testing."""

from typing import Dict, Any, List, Optional
from loguru import logger


class MockSlackService:
    """Mock Slack service for testing."""
    
    def __init__(self):
        self.sent_messages = []
        self.channel_info = {}
        self.user_info = {}
        self.conversation_history = {}
        self.message_counter = 0
    
    async def send_message(
        self, 
        channel: str, 
        text: str, 
        thread_ts: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        token: Optional[str] = None
    ) -> Optional[str]:
        """Mock sending a message to Slack."""
        try:
            self.message_counter += 1
            message_ts = f"1234567890.{self.message_counter:06d}"
            
            message_data = {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
                "blocks": blocks,
                "token": token,
                "timestamp": message_ts,
                "type": "message"
            }
            
            self.sent_messages.append(message_data)
            
            # Store in conversation history
            if channel not in self.conversation_history:
                self.conversation_history[channel] = []
            
            self.conversation_history[channel].append({
                "ts": message_ts,
                "text": text,
                "blocks": blocks,
                "thread_ts": thread_ts
            })
            
            logger.info(f"Mock message sent to channel {channel}: {text[:50]}...")
            return message_ts
            
        except Exception as e:
            logger.error(f"Mock Slack send_message failed: {e}")
            return None
    
    async def get_user_info(self, user_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Mock getting user information."""
        if user_id in self.user_info:
            return self.user_info[user_id]
        
        # Return default user info
        default_user = {
            "id": user_id,
            "name": f"user_{user_id}",
            "real_name": f"User {user_id}",
            "profile": {
                "display_name": f"User {user_id}",
                "real_name": f"User {user_id}",
                "email": f"user_{user_id}@example.com"
            }
        }
        
        self.user_info[user_id] = default_user
        return default_user
    
    async def get_channel_info(self, channel_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Mock getting channel information."""
        if channel_id in self.channel_info:
            return self.channel_info[channel_id]
        
        # Return default channel info
        default_channel = {
            "id": channel_id,
            "name": f"channel_{channel_id}",
            "is_channel": True,
            "is_group": False,
            "is_im": False,
            "is_private": False,
            "is_mpim": False,
            "is_archived": False,
            "is_general": channel_id == "C1234567890"
        }
        
        self.channel_info[channel_id] = default_channel
        return default_channel
    
    async def get_conversation_history(
        self, 
        channel: str, 
        limit: int = 100,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
        token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Mock getting conversation history."""
        messages = self.conversation_history.get(channel, [])
        
        # Apply filters
        if oldest:
            messages = [msg for msg in messages if msg["ts"] >= oldest]
        if latest:
            messages = [msg for msg in messages if msg["ts"] <= latest]
        
        # Apply limit
        messages = messages[:limit]
        
        return {
            "ok": True,
            "messages": messages,
            "has_more": len(self.conversation_history.get(channel, [])) > limit
        }
    
    def add_user_info(self, user_id: str, user_data: Dict[str, Any]):
        """Add custom user info for testing."""
        self.user_info[user_id] = user_data
    
    def add_channel_info(self, channel_id: str, channel_data: Dict[str, Any]):
        """Add custom channel info for testing."""
        self.channel_info[channel_id] = channel_data
    
    def get_sent_messages(self) -> List[Dict[str, Any]]:
        """Get all sent messages for verification."""
        return self.sent_messages
    
    def get_messages_for_channel(self, channel: str) -> List[Dict[str, Any]]:
        """Get messages sent to specific channel."""
        return [msg for msg in self.sent_messages if msg["channel"] == channel]
    
    def clear_messages(self):
        """Clear all sent messages."""
        self.sent_messages = []
        self.conversation_history = {}
        self.message_counter = 0
    
    def verify_message_sent(self, channel: str, text_contains: str) -> bool:
        """Verify if a message with specific text was sent to channel."""
        for msg in self.sent_messages:
            if msg["channel"] == channel and text_contains in msg["text"]:
                return True
        return False
    
    def verify_blocks_sent(self, channel: str, block_type: str) -> bool:
        """Verify if blocks of specific type were sent to channel."""
        for msg in self.sent_messages:
            if msg["channel"] == channel and msg.get("blocks"):
                for block in msg["blocks"]:
                    if block.get("type") == block_type:
                        return True
        return False


# Global mock Slack service
mock_slack_service = MockSlackService()
