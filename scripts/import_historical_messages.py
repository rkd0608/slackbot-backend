#!/usr/bin/env python3
"""Import historical messages from Slack for testing and development."""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
from loguru import logger

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import get_db, AsyncSessionLocal
from app.core.config import settings
from app.models.base import Workspace, Message, User
from app.services.slack_service import SlackService
from app.workers.message_processor import process_message_async
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class HistoricalMessageImporter:
    """Import historical messages from Slack workspaces."""
    
    def __init__(self):
        self.slack_service = None
        self.db = None
    
    async def initialize(self):
        """Initialize the importer with database and Slack service."""
        try:
            # Initialize database session
            self.db = AsyncSessionLocal()
            
            # Initialize Slack service
            self.slack_service = SlackService()
            
            logger.info("Historical message importer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize importer: {e}")
            raise
    
    async def close(self):
        """Close database connection."""
        if self.db:
            await self.db.close()
    
    async def get_workspaces(self) -> List[Workspace]:
        """Get all available workspaces."""
        try:
            result = await self.db.execute(select(Workspace))
            workspaces = result.scalars().all()
            
            if not workspaces:
                logger.warning("No workspaces found in database")
                return []
            
            logger.info(f"Found {len(workspaces)} workspaces")
            return workspaces
            
        except Exception as e:
            logger.error(f"Error getting workspaces: {e}")
            return []
    
    async def import_workspace_messages(
        self, 
        workspace: Workspace, 
        days_back: int = 30,
        max_messages: int = 1000
    ) -> Dict[str, Any]:
        """Import messages from a specific workspace."""
        try:
            logger.info(f"Starting import for workspace: {workspace.name} (ID: {workspace.id})")
            
            # Get workspace tokens
            tokens = workspace.tokens
            bot_token = tokens.get("access_token")
            
            if not bot_token:
                logger.warning(f"No bot token found for workspace {workspace.name}")
                return {"status": "error", "message": "No bot token available"}
            
            # Initialize Slack client for this workspace
            slack_client = self.slack_service.get_client(bot_token)
            
            # Get channels where bot is present
            channels = await self.get_bot_channels(slack_client)
            
            if not channels:
                logger.warning(f"No channels found for bot in workspace {workspace.name}")
                return {"status": "error", "message": "No channels accessible"}
            
            total_imported = 0
            total_processed = 0
            
            # Import messages from each channel
            for channel in channels:
                try:
                    channel_id = channel["id"]
                    channel_name = channel["name"]
                    
                    logger.info(f"Importing messages from channel: {channel_name} ({channel_id})")
                    
                    # Import messages from this channel
                    imported_count = await self.import_channel_messages(
                        workspace, channel_id, channel_name, days_back, max_messages
                    )
                    
                    total_imported += imported_count
                    
                    # Process imported messages
                    processed_count = await self.process_imported_messages(workspace.id, channel_id)
                    total_processed += processed_count
                    
                    logger.info(f"Channel {channel_name}: {imported_count} messages imported, {processed_count} processed")
                    
                except Exception as e:
                    logger.error(f"Error importing from channel {channel.get('name', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Workspace {workspace.name}: {total_imported} messages imported, {total_processed} processed")
            
            return {
                "status": "success",
                "workspace_name": workspace.name,
                "total_imported": total_imported,
                "total_processed": total_processed,
                "channels_processed": len(channels)
            }
            
        except Exception as e:
            logger.error(f"Error importing workspace {workspace.name}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_bot_channels(self, slack_client) -> List[Dict[str, Any]]:
        """Get channels where the bot is present."""
        try:
            # Get list of channels
            response = await slack_client.conversations_list(
                types="public_channel,private_channel",
                limit=1000
            )
            
            if not response["ok"]:
                logger.error(f"Failed to get channels: {response.get('error')}")
                return []
            
            channels = response["channels"]
            
            # Filter channels where bot is a member
            bot_channels = []
            for channel in channels:
                if channel.get("is_member", False):
                    bot_channels.append({
                        "id": channel["id"],
                        "name": channel["name"],
                        "is_private": channel.get("is_private", False)
                    })
            
            logger.info(f"Found {len(bot_channels)} channels where bot is present")
            return bot_channels
            
        except Exception as e:
            logger.error(f"Error getting bot channels: {e}")
            return []
    
    async def import_channel_messages(
        self,
        workspace: Workspace,
        channel_id: str,
        channel_name: str,
        days_back: int,
        max_messages: int
    ) -> int:
        """Import messages from a specific channel."""
        try:
            # Calculate timestamp for days back
            cutoff_time = datetime.now() - timedelta(days=days_back)
            cutoff_ts = str(cutoff_time.timestamp())
            
            # Get workspace tokens
            tokens = workspace.tokens
            bot_token = tokens.get("access_token")
            slack_client = self.slack_service.get_client(bot_token)
            
            imported_count = 0
            cursor = None
            
            while imported_count < max_messages:
                try:
                    # Get messages from channel
                    response = await slack_client.conversations_history(
                        channel=channel_id,
                        limit=100,  # Slack API limit
                        cursor=cursor,
                        oldest=cutoff_ts
                    )
                    
                    if not response["ok"]:
                        logger.error(f"Failed to get messages from {channel_name}: {response.get('error')}")
                        break
                    
                    messages = response["messages"]
                    
                    if not messages:
                        logger.info(f"No more messages to import from {channel_name}")
                        break
                    
                    # Import messages
                    for msg in messages:
                        try:
                            # Skip bot messages and messages without text
                            if msg.get("bot_id") or not msg.get("text"):
                                continue
                            
                            # Check if message already exists
                            if await self.message_exists(workspace.id, channel_id, msg.get("ts")):
                                continue
                            
                            # Get or create user
                            user = await self.get_or_create_user(
                                msg.get("user", ""), workspace.id
                            )
                            
                            if not user:
                                continue
                            
                            # Create message record
                            message = Message(
                                workspace_id=workspace.id,
                                channel_id=channel_id,
                                user_id=user.id,
                                raw_payload=msg
                            )
                            
                            self.db.add(message)
                            await self.db.flush()
                            
                            imported_count += 1
                            
                            if imported_count >= max_messages:
                                break
                                
                        except Exception as e:
                            logger.error(f"Error importing message: {e}")
                            continue
                    
                    # Check if there are more messages
                    if not response.get("has_more", False):
                        break
                    
                    cursor = response.get("response_metadata", {}).get("next_cursor")
                    if not cursor:
                        break
                    
                except Exception as e:
                    logger.error(f"Error getting messages from {channel_name}: {e}")
                    break
            
            logger.info(f"Imported {imported_count} messages from {channel_name}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing channel messages: {e}")
            return 0
    
    async def message_exists(self, workspace_id: int, channel_id: str, ts: str) -> bool:
        """Check if a message already exists in the database."""
        try:
            result = await self.db.execute(
                select(Message).where(
                    Message.workspace_id == workspace_id,
                    Message.channel_id == channel_id,
                    Message.raw_payload.contains({"ts": ts})
                )
            )
            return result.scalar_one_or_none() is not None
            
        except Exception as e:
            logger.error(f"Error checking if message exists: {e}")
            return False
    
    async def get_or_create_user(self, slack_user_id: str, workspace_id: int) -> Optional[User]:
        """Get or create a user in the database."""
        try:
            if not slack_user_id:
                return None
            
            # Check if user already exists
            result = await self.db.execute(
                select(User).where(
                    User.slack_id == slack_user_id,
                    User.workspace_id == workspace_id
                )
            )
            user = result.scalar_one_or_none()
            
            if user:
                return user
            
            # Create new user
            new_user = User(
                workspace_id=workspace_id,
                slack_id=slack_user_id,
                name=f"User_{slack_user_id}",  # Placeholder name
                role="user"
            )
            
            self.db.add(new_user)
            await self.db.flush()
            
            return new_user
            
        except Exception as e:
            logger.error(f"Error getting/creating user {slack_user_id}: {e}")
            return None
    
    async def process_imported_messages(self, workspace_id: int, channel_id: str) -> int:
        """Process imported messages for knowledge extraction."""
        try:
            # Find unprocessed messages
            result = await self.db.execute(
                select(Message).where(
                    Message.workspace_id == workspace_id,
                    Message.channel_id == channel_id,
                    ~Message.raw_payload.has_key("processed_data")
                )
            )
            unprocessed_messages = result.scalars().all()
            
            processed_count = 0
            
            for message in unprocessed_messages:
                try:
                    # Extract message data
                    text = message.raw_payload.get("text", "")
                    user_id = message.raw_payload.get("user", "")
                    thread_ts = message.raw_payload.get("thread_ts")
                    ts = message.raw_payload.get("ts")
                    
                    if text and user_id:
                        # Add to processing queue
                        process_message_async.delay(
                            message_id=message.id,
                            workspace_id=workspace_id,
                            channel_id=channel_id,
                            user_id=user_id,
                            text=text,
                            thread_ts=thread_ts,
                            ts=ts
                        )
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error queuing message {message.id} for processing: {e}")
                    continue
            
            logger.info(f"Queued {processed_count} messages for processing")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing imported messages: {e}")
            return 0

async def main():
    """Main function for historical message import."""
    importer = HistoricalMessageImporter()
    
    try:
        # Initialize importer
        await importer.initialize()
        
        # Get workspaces
        workspaces = await importer.get_workspaces()
        
        if not workspaces:
            logger.error("No workspaces available for import")
            return
        
        # Import messages from each workspace
        for workspace in workspaces:
            try:
                result = await importer.import_workspace_messages(
                    workspace, 
                    days_back=30,  # Import last 30 days
                    max_messages=1000  # Max 1000 messages per workspace
                )
                
                if result["status"] == "success":
                    logger.info(f"✅ Successfully imported from {workspace.name}: {result['total_imported']} messages")
                else:
                    logger.error(f"❌ Failed to import from {workspace.name}: {result['message']}")
                    
            except Exception as e:
                logger.error(f"❌ Error importing from workspace {workspace.name}: {e}")
                continue
        
        logger.info("Historical message import completed")
        
    except Exception as e:
        logger.error(f"Fatal error during import: {e}")
        raise
    
    finally:
        await importer.close()

if __name__ == "__main__":
    # Configure logging
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Run the import
    asyncio.run(main())
