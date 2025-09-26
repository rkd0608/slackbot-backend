"""Message processor worker for handling Slack message ingestion and processing."""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

from loguru import logger
from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .celery_app import celery_app
from ..core.config import settings
from ..models.base import Message, User, Workspace
from ..services.conversation_knowledge_extractor import ConversationKnowledgeExtractor
from ..services.enhanced_conversation_state_manager import EnhancedConversationStateManager as ConversationStateManager
from ..utils.text_processor import clean_text, extract_mentions, extract_urls


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
    timestamp: Optional[str] = None,
    thread_ts: Optional[str] = None,
    raw_payload: Optional[dict] = None
):
    """Process a message asynchronously using Celery."""
    try:
        logger.info(f"Processing message {message_id} from user {user_id} in channel {channel_id}")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                process_message(
                    message_id=message_id,
                    workspace_id=workspace_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    text=text,
                    thread_ts=thread_ts,
                    ts=timestamp
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing message {message_id}: {e}", exc_info=True)
        raise

async def process_message(
    message_id: int,
    workspace_id: int,
    channel_id: str,
    user_id: str,
    text: str,
    thread_ts: Optional[str] = None,
    ts: Optional[str] = None
) -> Dict[str, Any]:
    """Process a single message for knowledge extraction and analysis."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Get the message from database
            result = await db.execute(
                select(Message).where(Message.id == message_id)
            )
            message = result.scalar_one_or_none()
            
            if not message:
                logger.warning(f"Message {message_id} not found in database")
                return {"status": "error", "message": "Message not found"}
            
            # Get workspace and user info from the new model structure
            # First get the conversation to get the workspace
            from ..models.base import Conversation
            conversation_result = await db.execute(
                select(Conversation).where(Conversation.id == message.conversation_id)
            )
            conversation = conversation_result.scalar_one_or_none()
            
            if not conversation:
                logger.warning(f"Conversation not found for message {message_id}")
                return {"status": "error", "message": "Conversation not found"}
            
            workspace_result = await db.execute(
                select(Workspace).where(Workspace.id == conversation.workspace_id)
            )
            workspace = workspace_result.scalar_one_or_none()
            
            # Get user by slack_user_id
            user_result = await db.execute(
                select(User).where(
                    and_(
                        User.slack_id == message.slack_user_id,
                        User.workspace_id == conversation.workspace_id
                    )
                )
            )
            user = user_result.scalar_one_or_none()
            
            if not workspace or not user:
                logger.warning(f"Workspace or user not found for message {message_id}")
                return {"status": "error", "message": "Workspace or user not found"}
            
            # Process the message text
            processed_data = await process_message_text(text, message, workspace, user)
            
            # Update message with processing metadata
            await update_message_metadata(message_id, processed_data, db)
            
            # NOTE: Disabled old message-level extraction in favor of conversation-level extraction
            # The new system waits for complete conversations before extracting knowledge
            # if should_extract_knowledge(processed_data):
            #     await trigger_knowledge_extraction(message_id, processed_data, db)
            
            logger.info(f"Successfully processed message {message_id}")
            return {
                "status": "success",
                "message_id": message_id,
                "processed_data": processed_data
            }
            
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}", exc_info=True)
            await db.rollback()
            raise

async def process_message_text(
    text: str,
    message: Message,
    workspace: Workspace,
    user: User
) -> Dict[str, Any]:
    """Process and analyze message text content."""
    try:
        # Clean and normalize text
        cleaned_text = clean_text(text)
        
        # Extract mentions
        mentions = extract_mentions(text)
        
        # Extract URLs
        urls = extract_urls(text)
        
        # Basic text analysis
        word_count = len(cleaned_text.split())
        char_count = len(cleaned_text)
        
        # Determine message type
        message_type = classify_message_type(text, cleaned_text)
        
        # Check if message is in a thread
        is_thread = bool(message.message_metadata.get("slack_thread_ts"))
        is_thread_reply = bool(message.message_metadata.get("slack_thread_ts") and 
                              message.message_metadata.get("slack_thread_ts") != message.message_metadata.get("slack_ts"))
        
        # Determine if message is significant for knowledge extraction
        significance_score = calculate_significance_score(
            text=cleaned_text,
            word_count=word_count,
            message_type=message_type,
            is_thread_reply=is_thread_reply,
            mentions=mentions,
            urls=urls
        )
        
        return {
            "cleaned_text": cleaned_text,
            "mentions": mentions,
            "urls": urls,
            "word_count": word_count,
            "char_count": char_count,
            "message_type": message_type,
            "is_thread": is_thread,
            "is_thread_reply": is_thread_reply,
            "significance_score": significance_score,
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing message text: {e}", exc_info=True)
        return {
            "error": str(e),
            "cleaned_text": text,
            "significance_score": 0
        }

def classify_message_type(text: str, cleaned_text: str) -> str:
    """Classify the type of message."""
    text_lower = cleaned_text.lower()
    
    # Check for questions
    if any(q in text_lower for q in ["?", "what", "how", "why", "when", "where", "who"]):
        return "question"
    
    # Check for commands
    if text.startswith("/") or text.startswith("!"):
        return "command"
    
    # Check for links
    if "http" in text_lower:
        return "link_share"
    
    # Check for code blocks
    if "```" in text:
        return "code"
    
    # Check for file uploads
    if "uploaded a file" in text_lower:
        return "file_upload"
    
    # Check for reactions
    if text.startswith("+") or text.startswith("-"):
        return "reaction"
    
    return "conversation"

def calculate_significance_score(
    text: str,
    word_count: int,
    message_type: str,
    is_thread_reply: bool,
    mentions: List[str],
    urls: List[str]
) -> float:
    """Calculate how significant a message is for knowledge extraction."""
    score = 0.0
    
    # Base score from word count
    if word_count > 20:
        score += 0.3
    elif word_count > 10:
        score += 0.2
    elif word_count > 5:
        score += 0.1
    
    # Message type scoring
    type_scores = {
        "question": 0.4,
        "conversation": 0.2,
        "code": 0.3,
        "link_share": 0.25,
        "command": 0.1,
        "file_upload": 0.15,
        "reaction": 0.05
    }
    score += type_scores.get(message_type, 0.1)
    
    # Thread reply penalty (usually less significant)
    if is_thread_reply:
        score *= 0.7
    
    # Mentions bonus
    if mentions:
        score += 0.1 * len(mentions)
    
    # URLs bonus
    if urls:
        score += 0.15 * len(urls)
    
    # Cap the score
    return min(score, 1.0)

def should_extract_knowledge(processed_data: Dict[str, Any]) -> bool:
    """Determine if a message should trigger knowledge extraction."""
    significance_score = processed_data.get("significance_score", 0)
    message_type = processed_data.get("message_type", "conversation")
    
    # High significance messages
    if significance_score > 0.4:  # Temporarily lowered for testing
        return True
    
    # Questions are always worth extracting
    if message_type == "question":
        return True
    
    # Code and link shares with decent significance
    if message_type in ["code", "link_share"] and significance_score > 0.4:
        return True
    
    return False

async def update_message_metadata(
    message_id: int,
    processed_data: Dict[str, Any],
    db: AsyncSession
):
    """Update message with processing metadata."""
    try:
        # Update the message with processing results
        # Use a simpler approach: get current metadata and update it
        result = await db.execute(
            select(Message.message_metadata).where(Message.id == message_id)
        )
        current_metadata = result.scalar_one_or_none()
        
        if current_metadata:
            # Update the metadata with processed data
            current_metadata["processed_data"] = processed_data
            
            await db.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(message_metadata=current_metadata)
            )
        await db.commit()
        
    except Exception as e:
        logger.error(f"Error updating message metadata: {e}", exc_info=True)
        await db.rollback()
        raise

async def trigger_knowledge_extraction(
    message_id: int,
    processed_data: Dict[str, Any],
    db: AsyncSession
):
    """Trigger knowledge extraction for significant messages."""
    try:
        # Import here to avoid circular imports
        from .knowledge_extractor import extract_knowledge
        
        # Add to knowledge extraction queue
        extract_knowledge.delay(
            message_id=message_id,
            text=processed_data.get("cleaned_text", ""),
            message_type=processed_data.get("message_type", "conversation"),
            significance_score=processed_data.get("significance_score", 0)
        )
        
        logger.info(f"Triggered knowledge extraction for message {message_id}")
        
    except Exception as e:
        logger.error(f"Error triggering knowledge extraction: {e}", exc_info=True)

@celery_app.task
def process_pending_messages():
    """Process any pending messages that haven't been processed yet."""
    try:
        logger.info("Processing pending messages...")
        
        # Create a new event loop for this task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function
            result = loop.run_until_complete(process_pending_messages_async())
            return result
        finally:
            # Clean up the loop properly
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error processing pending messages: {e}", exc_info=True)
        raise

async def process_pending_messages_async() -> Dict[str, Any]:
    """Process messages that don't have processed_data yet."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Find messages without processed_data
            result = await db.execute(
                select(Message).where(
                    ~Message.message_metadata.has_key("processed_data")
                ).limit(100)  # Process in batches
            )
            pending_messages = result.scalars().all()
            
            processed_count = 0
            for message in pending_messages:
                try:
                    # Extract basic info from message metadata
                    text = message.content
                    user_id = message.slack_user_id
                    # Get channel_id and workspace_id from conversation
                    from ..models.base import Conversation
                    conv_result = await db.execute(
                        select(Conversation.slack_channel_id, Conversation.workspace_id).where(Conversation.id == message.conversation_id)
                    )
                    conv_data = conv_result.first()
                    if conv_data:
                        channel_id = conv_data.slack_channel_id
                        workspace_id = conv_data.workspace_id
                    else:
                        continue
                    
                    thread_ts = message.message_metadata.get("slack_thread_ts")
                    ts = message.message_metadata.get("slack_ts")
                    
                    if text and user_id and channel_id:
                        await process_message(
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
                    logger.error(f"Error processing pending message {message.id}: {e}")
                    continue
            
            logger.info(f"Processed {processed_count} pending messages")
            return {"status": "success", "processed_count": processed_count}
            
        except Exception as e:
            logger.error(f"Error processing pending messages: {e}", exc_info=True)
            await db.rollback()
            raise


@celery_app.task
def process_completed_conversations(workspace_id: int, batch_size: int = 10):
    """
    Process completed conversations for knowledge extraction.
    
    This is the NEW conversation-level processing approach that waits for
    conversations to complete before extracting knowledge.
    """
    try:
        logger.info(f"Processing completed conversations for workspace {workspace_id}")
        
        # Create a new event loop for this task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function
            result = loop.run_until_complete(
                process_completed_conversations_async(workspace_id, batch_size)
            )
            return result
        finally:
            # Clean up the loop properly
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error processing completed conversations: {e}", exc_info=True)
        raise


async def process_completed_conversations_async(workspace_id: int, batch_size: int = 10) -> Dict[str, Any]:
    """Process conversations that have reached completion for knowledge extraction."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Initialize the conversation-level knowledge extractor
            extractor = ConversationKnowledgeExtractor()
            
            # Extract knowledge from completed conversations
            knowledge_items = await extractor.extract_from_completed_conversations(
                workspace_id=workspace_id,
                db=db,
                batch_size=batch_size
            )
            
            logger.info(f"Extracted {len(knowledge_items)} knowledge items from completed conversations")
            
            return {
                "status": "success",
                "workspace_id": workspace_id,
                "knowledge_items_extracted": len(knowledge_items),
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in conversation-level processing: {e}", exc_info=True)
            await db.rollback()
            raise


@celery_app.task
def analyze_conversation_state(conversation_id: int):
    """
    Analyze the current state of a specific conversation.
    
    This can be triggered when new messages arrive to check if a conversation
    has reached completion and is ready for knowledge extraction.
    """
    try:
        logger.info(f"Analyzing state of conversation {conversation_id}")
        
        # Create a new event loop for this task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function
            result = loop.run_until_complete(
                analyze_conversation_state_async(conversation_id)
            )
            return result
        finally:
            # Clean up the loop properly
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error analyzing conversation state: {e}", exc_info=True)
        raise


async def analyze_conversation_state_async(conversation_id: int) -> Dict[str, Any]:
    """Analyze conversation state and trigger extraction if ready."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Initialize services
            state_manager = ConversationStateManager()
            extractor = ConversationKnowledgeExtractor()
            
            # Analyze conversation state
            boundary = await state_manager.analyze_conversation_state(conversation_id, db)
            
            result = {
                "conversation_id": conversation_id,
                "state": boundary.state.value,
                "confidence": boundary.confidence,
                "is_ready_for_extraction": await state_manager.should_extract_knowledge(boundary),
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
            # If conversation is completed and ready, trigger extraction
            if result["is_ready_for_extraction"]:
                logger.info(f"Conversation {conversation_id} is ready for knowledge extraction")
                
                # Get workspace_id from conversation
                from ..models.base import Conversation
                conv_result = await db.execute(
                    select(Conversation.workspace_id).where(Conversation.id == conversation_id)
                )
                workspace_id = conv_result.scalar()
                
                if workspace_id:
                    # Extract knowledge from this specific conversation
                    knowledge_items = await extractor._extract_from_single_conversation(
                        conversation_id, workspace_id, db
                    )
                    result["knowledge_items_extracted"] = len(knowledge_items)
                    
                    # Mark as processed
                    await extractor._mark_conversation_processed(conversation_id, db)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing conversation {conversation_id}: {e}", exc_info=True)
            await db.rollback()
            raise


if __name__ == "__main__":
    celery_app.start()
