"""Simplified message processor focused on basic message ingestion and conversation tracking."""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import select, and_
from sqlalchemy.orm import sessionmaker
from loguru import logger

from .simplified_celery_app import celery_app
from ..core.config import settings
from ..models.base import Message, User, Workspace, Conversation

def get_async_session():
    """Create a new async session for each task."""
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal

@celery_app.task
def process_message_async(
    message_id: int,
    workspace_id: int,
    channel_id: str,
    user_id: str,
    text: str,
    timestamp: str,
    thread_ts: Optional[str] = None,
    raw_payload: Optional[Dict[str, Any]] = None
):
    """Process a single message with simplified approach."""
    try:
        logger.info(f"Processing message {message_id} from user {user_id}")
        return asyncio.run(process_message(
            message_id, workspace_id, channel_id, user_id, text, timestamp, thread_ts, raw_payload
        ))
    except Exception as e:
        logger.error(f"Error in process_message_async: {e}")
        return {"status": "error", "message": str(e)}

async def process_message(
    message_id: int,
    workspace_id: int,
    channel_id: str,
    user_id: str,
    text: str,
    timestamp: str,
    thread_ts: Optional[str] = None,
    raw_payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a message with simplified logic."""
    
    async_session = get_async_session()
    
    async with async_session() as db:
        try:
            # Get or create conversation
            conversation = await get_or_create_conversation(
                workspace_id, channel_id, thread_ts, db
            )
            
            # Check if message already exists by slack_message_id
            existing_query = select(Message).where(Message.slack_message_id == timestamp)
            existing_result = await db.execute(existing_query)
            existing_message = existing_result.scalar_one_or_none()
            
            if existing_message:
                logger.info(f"Message {timestamp} already processed")
                return {"status": "already_processed", "message_id": existing_message.id}
            
            # Create message record with new model structure
            message = Message(
                conversation_id=conversation.id,
                slack_message_id=timestamp,
                slack_user_id=user_id,
                content=text or "",
                message_metadata={
                    'raw_payload': raw_payload or {},
                    'slack_ts': timestamp,
                    'slack_user_id': user_id,
                    'slack_thread_ts': thread_ts,
                    'slack_type': 'message',
                    'slack_subtype': None
                }
            )
            
            db.add(message)
            
            # Update conversation last activity
            conversation.updated_at = datetime.utcnow()
            
            await db.commit()
            
            logger.info(f"Successfully processed message {message_id}")
            
            # Trigger knowledge extraction if conversation has enough messages
            await check_and_trigger_knowledge_extraction(conversation.id, db)
            
            return {
                "status": "success",
                "message_id": message_id,
                "conversation_id": conversation.id
            }
            
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            await db.rollback()
            return {"status": "error", "message": str(e)}

async def get_or_create_conversation(
    workspace_id: int,
    channel_id: str,
    thread_ts: Optional[str],
    db: AsyncSession
) -> Conversation:
    """Get existing conversation or create new one."""
    
    try:
        # For threaded messages, use thread_ts to group
        # For non-threaded, create conversation per channel (simplified approach)
        conversation_key = thread_ts if thread_ts else f"channel_{channel_id}"
        
        # Look for existing conversation
        query = select(Conversation).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.slack_channel_id == channel_id,
                Conversation.thread_timestamp == thread_ts
            )
        )
        
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()
        
        if conversation:
            return conversation
        
        # Create new conversation
        conversation = Conversation(
            workspace_id=workspace_id,
            slack_channel_id=channel_id,
            slack_channel_name=f"channel-{channel_id}",
            thread_timestamp=thread_ts,
            title=f"#{channel_id}",
            conversation_metadata={}
        )
        
        db.add(conversation)
        await db.flush()  # Get the ID
        
        logger.info(f"Created new conversation {conversation.id} for channel {channel_id}")
        return conversation
        
    except Exception as e:
        logger.error(f"Error getting/creating conversation: {e}")
        raise

async def check_and_trigger_knowledge_extraction(conversation_id: int, db: AsyncSession):
    """Check if conversation is ready for knowledge extraction and trigger if needed."""
    
    try:
        # Count messages in this conversation
        from sqlalchemy import func
        
        query = select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
        result = await db.execute(query)
        message_count = result.scalar()
        
        # If conversation has enough messages, consider it for extraction
        if message_count >= 5:  # At least 5 messages
            logger.info(f"Conversation {conversation_id} has {message_count} messages, eligible for extraction")
            
            # Trigger knowledge extraction (will be processed by background worker)
            from .simple_knowledge_extractor import extract_knowledge_for_workspace
            
            # Get workspace_id from conversation
            conv_query = select(Conversation).where(Conversation.id == conversation_id)
            conv_result = await db.execute(conv_query)
            conversation = conv_result.scalar_one_or_none()
            
            if conversation:
                # Queue extraction task for this workspace (will process all ready conversations)
                extract_knowledge_for_workspace.delay(conversation.workspace_id)
                logger.info(f"Queued knowledge extraction for workspace {conversation.workspace_id}")
        
    except Exception as e:
        logger.error(f"Error checking knowledge extraction readiness: {e}")

# Periodic cleanup task
@celery_app.task
def cleanup_old_messages():
    """Clean up very old messages to keep database manageable."""
    try:
        logger.info("Starting message cleanup...")
        return asyncio.run(cleanup_old_messages_async())
    except Exception as e:
        logger.error(f"Error in cleanup_old_messages: {e}")
        return {"status": "error", "message": str(e)}

async def cleanup_old_messages_async():
    """Clean up messages older than 6 months."""
    async_session = get_async_session()
    
    async with async_session() as db:
        try:
            # Delete messages older than 6 months
            cutoff_date = datetime.utcnow() - timedelta(days=180)
            
            # For now, just log what we would clean up
            from sqlalchemy import func
            query = select(func.count(Message.id)).where(Message.created_at < cutoff_date)
            result = await db.execute(query)
            old_message_count = result.scalar()
            
            logger.info(f"Found {old_message_count} messages older than 6 months")
            
            # In production, you'd actually delete them here
            # For now, just return the count
            return {"status": "success", "old_messages_found": old_message_count}
            
        except Exception as e:
            logger.error(f"Error in cleanup_old_messages_async: {e}")
            return {"status": "error", "message": str(e)}
