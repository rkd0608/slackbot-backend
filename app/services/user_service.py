"""
User Service for resolving Slack user IDs to human-readable names.

This service provides functionality to:
- Resolve Slack user IDs to real names in conversations
- Cache user information for performance
- Replace user mentions with readable names in knowledge content
"""

import re
from typing import Dict, Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from ..models.base import User, Workspace
from .slack_service import SlackService


class UserService:
    """Service for user-related operations and user ID resolution."""
    
    def __init__(self):
        self.slack_service = SlackService()
        self._user_cache: Dict[str, str] = {}  # Cache: slack_id -> name
        
    async def get_user_name(self, slack_user_id: str, workspace_id: int, db: AsyncSession) -> str:
        """
        Get the display name for a Slack user ID.
        
        Args:
            slack_user_id: The Slack user ID (e.g., "U09H7A34YNS")
            workspace_id: The workspace ID
            db: Database session
            
        Returns:
            The user's display name or the original ID if not found
        """
        # Check cache first
        cache_key = f"{workspace_id}:{slack_user_id}"
        if cache_key in self._user_cache:
            return self._user_cache[cache_key]
        
        try:
            # Try to get user from database first
            user_query = select(User).where(
                User.slack_id == slack_user_id,
                User.workspace_id == workspace_id
            )
            result = await db.execute(user_query)
            user = result.scalars().first()  # Get first user if multiple exist
            
            if user and user.name:
                display_name = user.name
                self._user_cache[cache_key] = display_name
                return display_name
                
            # If not in database, try to get from Slack API and create/update user
            slack_user_data = await self._fetch_user_from_slack(slack_user_id, workspace_id, db)
            if slack_user_data:
                display_name = slack_user_data.get('display_name', slack_user_data.get('real_name', f"{slack_user_id[-4:]}"))
                self._user_cache[cache_key] = display_name
                return display_name
            
            # Final fallback: return a more readable version
            display_name = f"{slack_user_id[-4:]}"  # Use last 4 chars of user ID
            self._user_cache[cache_key] = display_name
            return display_name
            
        except Exception as e:
            logger.warning(f"Could not resolve user {slack_user_id}: {e}")
            return slack_user_id
    
    async def resolve_user_mentions_in_text(
        self, 
        text: str, 
        workspace_id: int, 
        db: AsyncSession,
        format_style: str = "name_only"
    ) -> str:
        """
        Replace Slack user ID mentions in text with readable names.
        
        Args:
            text: Text containing user mentions like @U09H7A34YNS
            workspace_id: The workspace ID
            db: Database session
            format_style: How to format mentions ("name_only", "at_name", "name_id")
            
        Returns:
            Text with user IDs replaced by names
        """
        # Pattern to match Slack user IDs (both @U... and bare U...)
        user_id_pattern = r'@?U[0-9A-Z]{8,11}'
        
        async def replace_user_id(match):
            user_id = match.group(0).lstrip('@')  # Remove @ if present
            name = await self.get_user_name(user_id, workspace_id, db)
            
            # Format according to style
            if format_style == "name_only":
                return name
            elif format_style == "at_name":
                return f"@{name}"
            elif format_style == "name_id":
                return f"{name} ({user_id})"
            else:
                return name
        
        # Find all user ID matches
        matches = list(re.finditer(user_id_pattern, text))
        
        # Replace from right to left to avoid index shifting
        result_text = text
        for match in reversed(matches):
            user_id = match.group(0).lstrip('@')
            name = await self.get_user_name(user_id, workspace_id, db)
            
            # Format according to style
            if format_style == "name_only":
                replacement = name
            elif format_style == "at_name":
                replacement = f"@{name}"
            elif format_style == "name_id":
                replacement = f"{name} ({user_id})"
            else:
                replacement = name
            
            result_text = result_text[:match.start()] + replacement + result_text[match.end():]
        
        return result_text
    
    async def get_conversation_participants(
        self, 
        slack_user_ids: List[str], 
        workspace_id: int, 
        db: AsyncSession
    ) -> Dict[str, str]:
        """
        Get a mapping of user IDs to names for conversation participants.
        
        Args:
            slack_user_ids: List of Slack user IDs
            workspace_id: The workspace ID
            db: Database session
            
        Returns:
            Dictionary mapping user_id -> display_name
        """
        participants = {}
        for user_id in slack_user_ids:
            name = await self.get_user_name(user_id, workspace_id, db)
            participants[user_id] = name
        
        return participants
    
    def clear_cache(self):
        """Clear the user name cache."""
        self._user_cache.clear()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "cached_users": len(self._user_cache),
            "cache_keys": list(self._user_cache.keys())
        }
    
    async def _fetch_user_from_slack(
        self, 
        slack_user_id: str, 
        workspace_id: int, 
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch user information from Slack API and update/create user in database.
        
        Args:
            slack_user_id: The Slack user ID
            workspace_id: The workspace ID
            db: Database session
            
        Returns:
            User data dict if successful, None otherwise
        """
        try:
            # Get workspace to get the bot token
            workspace_query = select(Workspace).where(Workspace.id == workspace_id)
            workspace_result = await db.execute(workspace_query)
            workspace = workspace_result.scalar_one_or_none()
            
            # Get bot token from workspace tokens (try both possible keys)
            bot_token = None
            if workspace and workspace.tokens:
                bot_token = workspace.tokens.get('bot_token') or workspace.tokens.get('access_token')
            if not bot_token:
                logger.warning(f"No bot token found for workspace {workspace_id}")
                return None
            
            # Fetch user info from Slack API
            user_info = await self.slack_service.get_user_info(
                user_id=slack_user_id,
                token=bot_token
            )
            
            if not user_info:
                logger.warning(f"Could not fetch user info from Slack for {slack_user_id}")
                return None
            
            # Extract user data
            profile = user_info.get('profile', {})
            real_name = user_info.get('real_name', '')
            display_name = profile.get('display_name', '')
            first_name = profile.get('first_name', '')
            last_name = profile.get('last_name', '')
            
            # Determine best name to use
            best_name = (
                display_name or 
                real_name or 
                f"{first_name} {last_name}".strip() or
                f"{slack_user_id[-4:]}"
            )
            
            # Update or create user in database
            await self._update_or_create_user(
                slack_user_id=slack_user_id,
                workspace_id=workspace_id,
                name=best_name,
                user_info=user_info,
                db=db
            )
            
            logger.info(f"Successfully fetched and updated user {slack_user_id}: {best_name}")
            
            return {
                'real_name': real_name,
                'display_name': display_name,
                'first_name': first_name,
                'last_name': last_name,
                'best_name': best_name
            }
            
        except Exception as e:
            logger.error(f"Error fetching user {slack_user_id} from Slack: {e}")
            return None
    
    async def _update_or_create_user(
        self,
        slack_user_id: str,
        workspace_id: int,
        name: str,
        user_info: Dict[str, Any],
        db: AsyncSession
    ):
        """
        Update existing user or create new user with Slack data.
        
        Args:
            slack_user_id: The Slack user ID
            workspace_id: The workspace ID  
            name: The resolved name for the user
            user_info: Full user info from Slack API
            db: Database session
        """
        try:
            # Check if user already exists
            user_query = select(User).where(
                User.slack_id == slack_user_id,
                User.workspace_id == workspace_id
            )
            result = await db.execute(user_query)
            user = result.scalars().first()
            
            if user:
                # Update existing user
                user.name = name
                logger.info(f"Updated existing user {slack_user_id} with name: {name}")
            else:
                # Create new user
                user = User(
                    workspace_id=workspace_id,
                    slack_id=slack_user_id,
                    name=name,
                    role="user"
                )
                db.add(user)
                logger.info(f"Created new user {slack_user_id} with name: {name}")
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"Error updating/creating user {slack_user_id}: {e}")
            await db.rollback()
    
    async def sync_all_users_from_slack(self, workspace_id: int, db: AsyncSession) -> Dict[str, Any]:
        """
        Sync all users in the database with Slack API to get real names.
        
        Args:
            workspace_id: The workspace ID to sync users for
            db: Database session
            
        Returns:
            Summary of sync operation
        """
        try:
            # Get all users with fallback names (containing "User_")
            users_query = select(User).where(
                User.workspace_id == workspace_id,
                User.name.like('User_%')
            )
            result = await db.execute(users_query)
            users_to_sync = result.scalars().all()
            
            logger.info(f"Found {len(users_to_sync)} users to sync from Slack API")
            
            synced_count = 0
            failed_count = 0
            
            for user in users_to_sync:
                logger.info(f"Syncing user {user.slack_id} (current name: {user.name})")
                
                slack_data = await self._fetch_user_from_slack(
                    user.slack_id, 
                    workspace_id, 
                    db
                )
                
                if slack_data:
                    synced_count += 1
                    # Clear cache for this user so it gets the new name
                    cache_key = f"{workspace_id}:{user.slack_id}"
                    if cache_key in self._user_cache:
                        del self._user_cache[cache_key]
                else:
                    failed_count += 1
            
            return {
                "total_users": len(users_to_sync),
                "synced_successfully": synced_count,
                "failed": failed_count,
                "success_rate": synced_count / len(users_to_sync) if users_to_sync else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error syncing users from Slack: {e}")
            return {
                "error": str(e),
                "total_users": 0,
                "synced_successfully": 0,
                "failed": 0,
                "success_rate": 0.0
            }
