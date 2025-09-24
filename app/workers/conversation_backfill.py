import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from ..models.base import Workspace, Conversation, Message
from ..core.database import AsyncSessionLocal
from ..services.slack_service import SlackService
from ..workers.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task
def backfill_conversation_history_async(workspace_id: int, channel_id: str, days_back: int = 30):
    """Backfill conversation history for a specific channel."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(backfill_conversation_history(workspace_id, channel_id, days_back))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in backfill_conversation_history_async: {e}", exc_info=True)
        raise

async def backfill_conversation_history(workspace_id: int, channel_id: str, days_back: int = 30) -> Dict[str, Any]:
    """Backfill conversation history for a specific channel."""
    async_session = AsyncSessionLocal()
    async with async_session() as db:
        try:
            # Get workspace and check if it has Slack tokens
            workspace = await db.execute(
                select(Workspace).where(Workspace.id == workspace_id)
            )
            workspace = workspace.scalar_one_or_none()
            
            if not workspace or not hasattr(workspace, 'tokens') or not workspace.tokens:
                logger.error(f"Workspace {workspace_id} not found or missing Slack tokens")
                return {"status": "error", "message": "Workspace not found or missing tokens"}
            
            # Get the bot token
            bot_token = workspace.tokens.get('bot_token')
            if not bot_token:
                logger.error(f"Workspace {workspace_id} missing bot token")
                return {"status": "error", "message": "Missing bot token"}
            
            # Initialize Slack service
            slack_service = SlackService()
            
            # Calculate the timestamp for days_back
            oldest_timestamp = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
            
            logger.info(f"Starting conversation history backfill for channel {channel_id}, workspace {workspace_id}")
            logger.info(f"Fetching messages from {days_back} days ago (timestamp: {oldest_timestamp})")
            
            # Fetch conversation history from Slack
            conversation_history = await slack_service.get_conversation_history(
                channel_id=channel_id,
                oldest=oldest_timestamp,
                limit=1000  # Slack's maximum
            )
            
            if not conversation_history.get('ok'):
                logger.error(f"Failed to fetch conversation history: {conversation_history}")
                return {"status": "error", "message": "Failed to fetch from Slack"}
            
            messages = conversation_history.get('messages', [])
            logger.info(f"Retrieved {len(messages)} messages from Slack")
            
            # Process and store messages
            processed_count = 0
            skipped_count = 0
            
            for message in messages:
                try:
                    # Skip bot messages and system messages
                    if message.get('subtype') in ['bot_message', 'channel_join', 'channel_leave']:
                        skipped_count += 1
                        continue
                    
                    # Create or get conversation first
                    conversation = await get_or_create_conversation(
                        db, workspace_id, channel_id, message, workspace
                    )
                    
                    # Check if message already exists
                    existing_message = await db.execute(
                        select(Message).where(
                            Message.slack_message_id == message.get('ts'),
                            Message.conversation_id == conversation.id
                        )
                    )
                    existing_message = existing_message.scalar_one_or_none()
                    
                    if existing_message:
                        logger.debug(f"Message {message.get('ts')} already exists, skipping")
                        skipped_count += 1
                        continue
                    
                    # Create message record
                    message_record = Message(
                        conversation_id=conversation.id,
                        slack_message_id=message.get('ts'),
                        slack_user_id=message.get('user'),
                        content=message.get('text', ''),
                        message_metadata={
                            'slack_ts': message.get('ts'),
                            'slack_user_id': message.get('user'),
                            'slack_thread_ts': message.get('thread_ts'),
                            'slack_reply_count': message.get('reply_count', 0),
                            'slack_reply_users_count': message.get('reply_users_count', 0),
                            'slack_reactions': message.get('reactions', []),
                            'slack_attachments': message.get('attachments', []),
                            'slack_blocks': message.get('blocks', []),
                            'source_channel_id': channel_id,
                            'workspace_id': workspace_id
                        },
                        created_at=datetime.fromtimestamp(float(message.get('ts', 0))),
                        updated_at=datetime.utcnow()
                    )
                    
                    db.add(message_record)
                    processed_count += 1
                    
                    # Commit every 100 messages to avoid long transactions
                    if processed_count % 100 == 0:
                        await db.commit()
                        logger.info(f"Committed {processed_count} messages so far")
                    
                except Exception as e:
                    logger.error(f"Error processing message {message.get('ts')}: {e}")
                    continue
            
            # Final commit
            await db.commit()
            
            logger.info(f"Successfully backfilled conversation history for channel {channel_id}")
            logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
            
            return {
                "status": "success",
                "channel_id": channel_id,
                "workspace_id": workspace_id,
                "processed_count": processed_count,
                "skipped_count": skipped_count,
                "total_messages": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error backfilling conversation history: {e}", exc_info=True)
            await db.rollback()
            raise

async def get_or_create_conversation(
    db: AsyncSession, 
    workspace_id: int, 
    channel_id: str, 
    message: Dict[str, Any],
    workspace: Workspace
) -> Conversation:
    """Get or create a conversation for the channel."""
    try:
        # Check if conversation already exists
        existing_conversation = await db.execute(
            select(Conversation).where(
                Conversation.slack_channel_id == channel_id,
                Conversation.workspace_id == workspace_id
            )
        )
        existing_conversation = existing_conversation.scalar_one_or_none()
        
        if existing_conversation:
            return existing_conversation
        
        # Get channel info from Slack
        slack_service = SlackService()
        channel_info = await slack_service.get_channel_info(channel_id)
        
        channel_name = channel_info.get('channel', {}).get('name', channel_id) if channel_info.get('ok') else channel_id
        
        # Create new conversation
        conversation = Conversation(
            workspace_id=workspace_id,
            slack_channel_id=channel_id,
            slack_channel_name=channel_name,
            title=f"Conversation in #{channel_name}",
            summary=f"Channel conversation history for #{channel_name}",
            conversation_metadata={
                'channel_name': channel_name,
                'channel_purpose': channel_info.get('channel', {}).get('purpose', {}).get('value', ''),
                'channel_topic': channel_info.get('channel', {}).get('topic', {}).get('value', ''),
                'member_count': channel_info.get('channel', {}).get('num_members', 0),
                'is_private': channel_info.get('channel', {}).get('is_private', False),
                'is_archived': channel_info.get('channel', {}).get('is_archived', False),
                'source_channel_id': channel_id,
                'workspace_id': workspace_id
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(conversation)
        await db.flush()  # Get the ID without committing
        
        return conversation
        
    except Exception as e:
        logger.error(f"Error getting/creating conversation: {e}")
        raise

@celery_app.task
def backfill_all_channels_async(workspace_id: int, days_back: int = 30):
    """Backfill conversation history for all channels in a workspace."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(backfill_all_channels(workspace_id, days_back))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in backfill_all_channels_async: {e}", exc_info=True)
        raise

async def backfill_all_channels(workspace_id: int, days_back: int = 30) -> Dict[str, Any]:
    """Backfill conversation history for all channels in a workspace."""
    async_session = AsyncSessionLocal()
    async with async_session() as db:
        try:
            # Get workspace
            workspace = await db.execute(
                select(Workspace).where(Workspace.id == workspace_id)
            )
            workspace = workspace.scalar_one_or_none()
            
            if not workspace or not hasattr(workspace, 'tokens') or not workspace.tokens:
                logger.error(f"Workspace {workspace_id} not found or missing Slack tokens")
                return {"status": "error", "message": "Workspace not found or missing tokens"}
            
            # Initialize Slack service
            slack_service = SlackService()
            
            # Get all channels for the workspace
            channels_response = await slack_service.get_channels_list()
            
            if not channels_response.get('ok'):
                logger.error(f"Failed to fetch channels: {channels_response}")
                return {"status": "error", "message": "Failed to fetch channels"}
            
            channels = channels_response.get('channels', [])
            logger.info(f"Found {len(channels)} channels to backfill")
            
            results = []
            for channel in channels:
                try:
                    channel_id = channel.get('id')
                    logger.info(f"Backfilling channel: {channel.get('name')} ({channel_id})")
                    
                    result = await backfill_conversation_history(workspace_id, channel_id, days_back)
                    results.append({
                        'channel_id': channel_id,
                        'channel_name': channel.get('name'),
                        'result': result
                    })
                    
                except Exception as e:
                    logger.error(f"Error backfilling channel {channel.get('name')}: {e}")
                    results.append({
                        'channel_id': channel.get('id'),
                        'channel_name': channel.get('name'),
                        'result': {'status': 'error', 'message': str(e)}
                    })
            
            return {
                "status": "completed",
                "workspace_id": workspace_id,
                "total_channels": len(channels),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error backfilling all channels: {e}", exc_info=True)
            raise
