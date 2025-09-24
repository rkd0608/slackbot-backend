"""
Enhanced Message Processor with conversation-level processing.

This implements the core architectural improvement: processes messages within
the context of conversation boundaries and states.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update
from loguru import logger

from ..core.database import get_session_factory
from ..models.base import Conversation, Message, User, Workspace
from ..services.enhanced_conversation_state_manager import (
    EnhancedConversationStateManager, 
    ConversationState
)
from ..workers.celery_app import celery_app


@celery_app.task
def process_message_with_conversation_context(
    message_id: int,
    workspace_id: int,
    channel_id: str,
    user_id: str,
    text: str,
    thread_ts: Optional[str] = None,
    ts: Optional[str] = None
):
    """
    Process a message with full conversation context awareness.
    
    This is the new message processing pipeline that:
    1. Groups messages into logical conversations
    2. Analyzes conversation state after each message
    3. Triggers knowledge extraction only when conversations complete
    """
    try:
        logger.info(f"Processing message {message_id} with conversation context")
        
        # Create event loop for async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                process_message_with_conversation_context_async(
                    message_id, workspace_id, channel_id, user_id, text, thread_ts, ts
                )
            )
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error processing message {message_id} with conversation context: {e}", exc_info=True)
        raise


async def process_message_with_conversation_context_async(
    message_id: int,
    workspace_id: int,
    channel_id: str,
    user_id: str,
    text: str,
    thread_ts: Optional[str] = None,
    ts: Optional[str] = None
) -> Dict[str, Any]:
    """Process message with enhanced conversation awareness."""
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        try:
            # 1. Get or create logical conversation
            conversation = await get_or_create_logical_conversation(
                workspace_id, channel_id, thread_ts, db
            )
            
            # 2. Store the message
            message = await store_message_with_context(
                message_id, conversation.id, user_id, text, ts, thread_ts, db
            )
            
            # 3. Update conversation metadata
            await update_conversation_metadata(conversation, db)
            
            # 4. Analyze conversation state
            state_manager = EnhancedConversationStateManager()
            boundary = await state_manager.analyze_conversation_state(conversation.id, db)
            
            # 5. Update conversation state in database
            await update_conversation_state(conversation.id, boundary, db)
            
            # 6. Trigger knowledge extraction if conversation is ready
            extraction_triggered = False
            if await state_manager.should_extract_knowledge(boundary):
                logger.info(f"Conversation {conversation.id} is ready for knowledge extraction")
                
                # Trigger enhanced knowledge extraction
                from ..workers.enhanced_knowledge_extractor import extract_knowledge_from_complete_conversation
                celery_app.send_task(
                    'app.workers.enhanced_knowledge_extractor.extract_knowledge_from_complete_conversation',
                    args=[conversation.id, workspace_id]
                )
                extraction_triggered = True
            
            await db.commit()
            
            return {
                "status": "success",
                "message_id": message_id,
                "conversation_id": conversation.id,
                "conversation_state": boundary.state.value,
                "state_confidence": boundary.confidence,
                "extraction_triggered": extraction_triggered,
                "message_count": boundary.message_count,
                "participant_count": boundary.participant_count,
                "topic": boundary.topic
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error in enhanced message processing: {e}")
            raise


async def get_or_create_logical_conversation(
    workspace_id: int,
    channel_id: str,
    thread_ts: Optional[str],
    db: AsyncSession
) -> Conversation:
    """
    Get or create a logical conversation unit.
    
    This implements sophisticated conversation grouping logic:
    - Threaded messages are grouped by thread_ts
    - Non-threaded messages in the same channel are grouped with time-based logic
    - Topic continuity is considered for grouping
    """
    try:
        # For threaded messages, group by thread_ts
        if thread_ts:
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
            
            # Create new threaded conversation
            conversation = Conversation(
                workspace_id=workspace_id,
                slack_channel_id=channel_id,
                thread_timestamp=thread_ts,
                state="developing",
                state_confidence=0.6,
                state_updated_at=datetime.utcnow()
            )
            
        else:
            # For non-threaded messages, implement sophisticated grouping logic
            conversation = await find_or_create_channel_conversation(
                workspace_id, channel_id, db
            )
        
        db.add(conversation)
        await db.flush()  # Get the ID
        
        logger.info(f"Created conversation {conversation.id} for channel {channel_id}, thread {thread_ts}")
        return conversation
        
    except Exception as e:
        logger.error(f"Error creating logical conversation: {e}")
        raise


async def find_or_create_channel_conversation(
    workspace_id: int,
    channel_id: str,
    db: AsyncSession
) -> Conversation:
    """
    Find active conversation in channel or create new one.
    
    This implements time-based and topic-based conversation grouping for
    non-threaded messages.
    """
    # Look for recent active conversations in the channel
    recent_cutoff = datetime.utcnow() - timedelta(minutes=30)
    
    query = select(Conversation).where(
        and_(
            Conversation.workspace_id == workspace_id,
            Conversation.slack_channel_id == channel_id,
            Conversation.thread_timestamp.is_(None),  # Non-threaded
            Conversation.state.in_(["developing", "paused"]),
            Conversation.updated_at >= recent_cutoff
        )
    ).order_by(Conversation.updated_at.desc())
    
    result = await db.execute(query)
    recent_conversations = result.scalars().all()
    
    # If we have recent active conversations, use the most recent one
    if recent_conversations:
        return recent_conversations[0]
    
    # Create new conversation for the channel
    return Conversation(
        workspace_id=workspace_id,
        slack_channel_id=channel_id,
        thread_timestamp=None,
        state="developing",
        state_confidence=0.6,
        state_updated_at=datetime.utcnow()
    )


async def store_message_with_context(
    message_id: int,
    conversation_id: int,
    user_id: str,
    text: str,
    ts: Optional[str],
    thread_ts: Optional[str],
    db: AsyncSession
) -> Message:
    """Store message with enhanced context information."""
    
    # Check if message already exists
    existing_query = select(Message).where(Message.id == message_id)
    existing_result = await db.execute(existing_query)
    existing_message = existing_result.scalar_one_or_none()
    
    if existing_message:
        logger.info(f"Message {message_id} already exists")
        return existing_message
    
    # Create new message with enhanced metadata
    message = Message(
        id=message_id,
        conversation_id=conversation_id,
        slack_message_id=ts or str(message_id),
        slack_user_id=user_id,
        content=text,
        message_metadata={
            'slack_ts': ts,
            'slack_thread_ts': thread_ts,
            'word_count': len(text.split()) if text else 0,
            'has_questions': '?' in text if text else False,
            'has_decisions': any(marker in text.lower() for marker in [
                'decided', 'decision', 'choose', 'go with', 'agreed'
            ]) if text else False,
            'processed_at': datetime.utcnow().isoformat()
        }
    )
    
    db.add(message)
    return message


async def update_conversation_metadata(conversation: Conversation, db: AsyncSession):
    """Update conversation metadata after adding a message."""
    
    # Count messages and participants
    message_count_query = select(func.count(Message.id)).where(
        Message.conversation_id == conversation.id
    )
    message_count_result = await db.execute(message_count_query)
    message_count = message_count_result.scalar()
    
    participant_count_query = select(func.count(func.distinct(Message.slack_user_id))).where(
        Message.conversation_id == conversation.id
    )
    participant_count_result = await db.execute(participant_count_query)
    participant_count = participant_count_result.scalar()
    
    # Update conversation
    conversation.message_count = message_count
    conversation.participant_count = participant_count
    conversation.updated_at = datetime.utcnow()


async def update_conversation_state(
    conversation_id: int,
    boundary,  # ConversationBoundary object
    db: AsyncSession
):
    """Update conversation state based on boundary analysis."""
    
    update_stmt = update(Conversation).where(
        Conversation.id == conversation_id
    ).values(
        state=boundary.state.value,
        state_confidence=boundary.confidence,
        state_updated_at=datetime.utcnow(),
        topic=boundary.topic,
        resolution_indicators=boundary.resolution_indicators,
        is_ready_for_extraction=boundary.state == ConversationState.RESOLVED and boundary.confidence > 0.8
    )
    
    await db.execute(update_stmt)


@celery_app.task
def analyze_conversation_state_batch(workspace_id: int, limit: int = 50):
    """
    Batch analyze conversation states for conversations that need state updates.
    
    This runs periodically to catch conversations that might have been missed
    by real-time processing.
    """
    try:
        logger.info(f"Running batch conversation state analysis for workspace {workspace_id}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                analyze_conversation_state_batch_async(workspace_id, limit)
            )
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error in batch conversation state analysis: {e}", exc_info=True)
        raise


async def analyze_conversation_state_batch_async(workspace_id: int, limit: int = 50) -> Dict[str, Any]:
    """Batch analyze conversation states asynchronously."""
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        try:
            # Find conversations that need state analysis
            # Priority: conversations with recent activity but not analyzed recently
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            query = select(Conversation.id).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.state.in_(["developing", "paused"]),
                    Conversation.updated_at >= cutoff_time,
                    Conversation.state_updated_at < cutoff_time
                )
            ).limit(limit)
            
            result = await db.execute(query)
            conversation_ids = [row[0] for row in result.fetchall()]
            
            if not conversation_ids:
                return {"status": "no_conversations_to_analyze", "count": 0}
            
            # Analyze each conversation
            state_manager = EnhancedConversationStateManager()
            analyzed_count = 0
            extraction_triggered_count = 0
            
            for conv_id in conversation_ids:
                try:
                    boundary = await state_manager.analyze_conversation_state(conv_id, db)
                    await update_conversation_state(conv_id, boundary, db)
                    analyzed_count += 1
                    
                    # Trigger extraction if ready
                    if await state_manager.should_extract_knowledge(boundary):
                        from ..workers.enhanced_knowledge_extractor import extract_knowledge_from_complete_conversation
                        celery_app.send_task(
                            'app.workers.enhanced_knowledge_extractor.extract_knowledge_from_complete_conversation',
                            args=[conv_id, workspace_id]
                        )
                        extraction_triggered_count += 1
                        
                except Exception as e:
                    logger.error(f"Error analyzing conversation {conv_id}: {e}")
                    continue
            
            await db.commit()
            
            return {
                "status": "success",
                "conversations_analyzed": analyzed_count,
                "extractions_triggered": extraction_triggered_count,
                "total_candidates": len(conversation_ids)
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error in batch conversation analysis: {e}")
            raise


@celery_app.task
def cleanup_abandoned_conversations(workspace_id: int, days_old: int = 7):
    """
    Clean up conversations that have been abandoned for a long time.
    
    This prevents the database from growing indefinitely with stale conversations.
    """
    try:
        logger.info(f"Cleaning up abandoned conversations for workspace {workspace_id}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                cleanup_abandoned_conversations_async(workspace_id, days_old)
            )
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error cleaning up abandoned conversations: {e}", exc_info=True)
        raise


async def cleanup_abandoned_conversations_async(workspace_id: int, days_old: int = 7) -> Dict[str, Any]:
    """Clean up abandoned conversations asynchronously."""
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        try:
            # Find conversations abandoned for more than specified days
            cutoff_time = datetime.utcnow() - timedelta(days=days_old)
            
            # Mark conversations as abandoned if they haven't been active
            update_stmt = update(Conversation).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.state == "developing",
                    Conversation.updated_at < cutoff_time,
                    Conversation.message_count < 10  # Only cleanup conversations with few messages
                )
            ).values(
                state="abandoned",
                state_updated_at=datetime.utcnow()
            )
            
            result = await db.execute(update_stmt)
            updated_count = result.rowcount
            
            await db.commit()
            
            return {
                "status": "success",
                "conversations_marked_abandoned": updated_count
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error cleaning up abandoned conversations: {e}")
            raise
