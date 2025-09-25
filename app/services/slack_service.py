"""Slack service for handling Slack API interactions."""

from typing import Optional, Dict, Any
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError
from loguru import logger

from ..core.config import settings

class SlackService:
    """Service for interacting with Slack APIs."""
    
    def __init__(self):
        self.default_token = settings.slack_bot_token
    
    def get_client(self, token: Optional[str] = None) -> AsyncWebClient:
        """Get a Slack client instance."""
        token_to_use = token or self.default_token
        
        if not token_to_use:
            raise ValueError("No Slack token provided")
        
        return AsyncWebClient(token=token_to_use)
    
    async def get_user_info(self, user_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get user information from Slack."""
        try:
            client = self.get_client(token)
            response = await client.users_info(user=user_id)
            
            if response["ok"]:
                return response["user"]
            else:
                logger.error(f"Failed to get user info: {response.get('error')}")
                return None
                
        except SlackApiError as e:
            logger.error(f"Slack API error getting user info: {e.response['error']}")
            return None
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
    
    async def get_channel_info(self, channel_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get channel information from Slack."""
        try:
            client = self.get_client(token)
            response = await client.conversations_info(channel=channel_id)
            
            if response["ok"]:
                return response["channel"]
            else:
                logger.error(f"Failed to get channel info: {response.get('error')}")
                return None
                
        except SlackApiError as e:
            logger.error(f"Slack API error getting channel info: {e.response['error']}")
            return None
        except Exception as e:
            logger.error(f"Error getting channel info: {e}")
            return None
    
    async def send_message(
        self, 
        channel: str, 
        text: str, 
        thread_ts: Optional[str] = None,
        blocks: Optional[list] = None,
        token: Optional[str] = None
    ) -> Optional[str]:
        """Send a message to a Slack channel."""
        try:
            client = self.get_client(token)
            
            message_kwargs = {
                "channel": channel,
                "text": text
            }
            
            if thread_ts:
                message_kwargs["thread_ts"] = thread_ts
            
            if blocks:
                message_kwargs["blocks"] = blocks
            
            response = await client.chat_postMessage(**message_kwargs)
            
            if response["ok"]:
                logger.info(f"Message sent to channel {channel}")
                return response["ts"]
            else:
                logger.error(f"Failed to send message: {response.get('error')}")
                return None
                
        except SlackApiError as e:
            logger.error(f"Slack API error sending message: {e.response['error']}")
            return None
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None
    
    async def get_conversation_history(
        self, 
        channel: str, 
        limit: int = 100,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
        token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get conversation history from a channel."""
        try:
            client = self.get_client(token)
            
            response = await client.conversations_history(
                channel=channel,
                limit=limit,
                oldest=oldest,
                latest=latest
            )
            
            if response["ok"]:
                return response
            else:
                logger.error(f"Failed to get conversation history: {response.get('error')}")
                return None
                
        except SlackApiError as e:
            logger.error(f"Slack API error getting conversation history: {e.response['error']}")
            return None
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return None
    
    async def get_channels_list(
        self, 
        types: str = "public_channel,private_channel",
        limit: int = 1000,
        token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get list of channels."""
        try:
            client = self.get_client(token)
            
            response = await client.conversations_list(
                types=types,
                limit=limit
            )
            
            if response["ok"]:
                return response
            else:
                logger.error(f"Failed to get channels list: {response.get('error')}")
                return None
                
        except SlackApiError as e:
            logger.error(f"Slack API error getting channels list: {e.response['error']}")
            return None
        except Exception as e:
            logger.error(f"Error getting channels list: {e}")
            return None
    
    async def add_reaction(
        self, 
        channel: str, 
        timestamp: str, 
        name: str,
        token: Optional[str] = None
    ) -> bool:
        """Add a reaction to a message."""
        try:
            client = self.get_client(token)
            
            response = await client.reactions_add(
                channel=channel,
                timestamp=timestamp,
                name=name
            )
            
            if response["ok"]:
                logger.info(f"Reaction {name} added to message {timestamp}")
                return True
            else:
                logger.error(f"Failed to add reaction: {response.get('error')}")
                return False
                
        except SlackApiError as e:
            logger.error(f"Slack API error adding reaction: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Error adding reaction: {e}")
            return False
    
    async def update_message(
        self, 
        channel: str, 
        timestamp: str, 
        text: str,
        token: Optional[str] = None
    ) -> bool:
        """Update a message in Slack."""
        try:
            client = self.get_client(token)
            
            response = await client.chat_update(
                channel=channel,
                ts=timestamp,
                text=text
            )
            
            if response["ok"]:
                logger.info(f"Message {timestamp} updated in channel {channel}")
                return True
            else:
                logger.error(f"Failed to update message: {response.get('error')}")
                return False
                
        except SlackApiError as e:
            logger.error(f"Slack API error updating message: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Error updating message: {e}")
            return False
    
    async def delete_message(
        self, 
        channel: str, 
        timestamp: str,
        token: Optional[str] = None
    ) -> bool:
        """Delete a message from Slack."""
        try:
            client = self.get_client(token)
            
            response = await client.chat_delete(
                channel=channel,
                ts=timestamp
            )
            
            if response["ok"]:
                logger.info(f"Message {timestamp} deleted from channel {channel}")
                return True
            else:
                logger.error(f"Failed to delete message: {response.get('error')}")
                return False
                
        except SlackApiError as e:
            logger.error(f"Slack API error deleting message: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Error deleting message: {e}")
            return False
    
    async def get_workspace_info(self, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get workspace information."""
        try:
            client = self.get_client(token)
            
            response = await client.team_info()
            
            if response["ok"]:
                return response["team"]
            else:
                logger.error(f"Failed to get workspace info: {response.get('error')}")
                return None
                
        except SlackApiError as e:
            logger.error(f"Slack API error getting workspace info: {e.response['error']}")
            return None
        except Exception as e:
            logger.error(f"Error getting workspace info: {e}")
            return None
    
    async def test_connection(self, token: Optional[str] = None) -> bool:
        """Test the Slack API connection."""
        try:
            client = self.get_client(token)
            
            response = await client.auth_test()
            
            if response["ok"]:
                logger.info(f"Slack connection test successful for workspace: {response.get('team')}")
                return True
            else:
                logger.error(f"Slack connection test failed: {response.get('error')}")
                return False
                
        except SlackApiError as e:
            logger.error(f"Slack API error during connection test: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Error during Slack connection test: {e}")
            return False
    
    async def open_modal(self, trigger_id: str, view: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Open a modal dialog in Slack."""
        try:
            client = AsyncWebClient(token=token)
            response = await client.views_open(
                trigger_id=trigger_id,
                view=view
            )
            
            if response["ok"]:
                logger.info(f"Modal opened successfully: {response.get('view', {}).get('id')}")
                return response
            else:
                logger.error(f"Failed to open modal: {response.get('error')}")
                return {"error": response.get("error", "Unknown error")}
                
        except SlackApiError as e:
            logger.error(f"Slack API error opening modal: {e.response['error']}")
            return {"error": e.response["error"]}
        except Exception as e:
            logger.error(f"Error opening modal: {e}")
            return {"error": str(e)}
    
    async def update_modal(self, view_id: str, view: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Update an existing modal dialog in Slack."""
        try:
            client = AsyncWebClient(token=token)
            response = await client.views_update(
                view_id=view_id,
                view=view
            )
            
            if response["ok"]:
                logger.info(f"Modal updated successfully: {view_id}")
                return response
            else:
                logger.error(f"Failed to update modal: {response.get('error')}")
                return {"error": response.get("error", "Unknown error")}
                
        except SlackApiError as e:
            logger.error(f"Slack API error updating modal: {e.response['error']}")
            return {"error": e.response["error"]}
        except Exception as e:
            logger.error(f"Error updating modal: {e}")
            return {"error": str(e)}
    
    async def publish_view(self, user_id: str, view: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Publish a view to a user's app home."""
        try:
            client = AsyncWebClient(token=token)
            response = await client.views_publish(
                user_id=user_id,
                view=view
            )
            
            if response["ok"]:
                logger.info(f"View published successfully for user: {user_id}")
                return response
            else:
                logger.error(f"Failed to publish view: {response.get('error')}")
                return {"error": response.get("error", "Unknown error")}
                
        except SlackApiError as e:
            logger.error(f"Slack API error publishing view: {e.response['error']}")
            return {"error": e.response["error"]}
        except Exception as e:
            logger.error(f"Error publishing view: {e}")
            return {"error": str(e)}
