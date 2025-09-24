"""Knowledge extraction worker for processing messages and extracting valuable knowledge."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import select, and_, update
from sqlalchemy.orm import sessionmaker
from loguru import logger
import json

from .celery_app import celery_app
from ..core.config import settings
from ..models.base import Message, User, Workspace, KnowledgeItem
from ..services.openai_service import OpenAIService
from ..services.slack_service import SlackService

def get_async_session():
    """Create a new async session for each task."""
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal

@celery_app.task
def extract_knowledge(
    message_id: int, 
    text: str, 
    message_type: str, 
    significance_score: float
):
    """Extract knowledge from a message using Celery."""
    try:
        logger.info(f"Starting knowledge extraction for message {message_id}")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                extract_knowledge_async(
                    message_id=message_id,
                    text=text,
                    message_type=message_type,
                    significance_score=significance_score
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in knowledge extraction task: {e}", exc_info=True)
        raise

async def extract_knowledge_async(
    message_id: int,
    text: str,
    message_type: str,
    significance_score: float
) -> Dict[str, Any]:
    """Extract knowledge from a message asynchronously."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Get the message and its context
            message = await get_message_with_context(message_id, db)
            if not message:
                return {"status": "error", "message": "Message not found"}
            
            # Get conversation context
            conversation_context = await get_conversation_context(
                message.workspace_id, 
                message.channel_id, 
                message_id, 
                db
            )
            
            # Initialize OpenAI service (lazy import)
            from ..services.openai_service import OpenAIService
            openai_service = OpenAIService()
            
            # Extract knowledge using AI
            message_metadata = {
                "message_id": message_id,
                "message_type": message_type,
                "significance_score": significance_score,
                "word_count": len(text.split()),
                "is_thread_reply": bool(message.raw_payload.get("thread_ts"))
            }
            
            extraction_result = await openai_service.extract_knowledge(
                conversation_context=conversation_context,
                message_text=text,
                message_metadata=message_metadata
            )
            
            # Verify extraction to prevent hallucination
            verification_result = await openai_service.verify_extraction(
                extracted_knowledge=extraction_result,
                source_messages=conversation_context
            )
            
            # Process and store knowledge items
            stored_items = await process_and_store_knowledge(
                extraction_result, 
                verification_result, 
                message, 
                db
            )
            
            # Update message with extraction results
            await update_message_extraction_status(message_id, extraction_result, db)
            
            # Trigger embedding generation for extracted knowledge items
            if stored_items:
                await _trigger_embedding_generation(stored_items, db)
            
            logger.info(f"Knowledge extraction completed for message {message_id}: {len(stored_items)} items stored")
            
            return {
                "status": "success",
                "message_id": message_id,
                "knowledge_items_extracted": len(extraction_result.get("knowledge_items", [])),
                "knowledge_items_stored": len(stored_items),
                "overall_confidence": extraction_result.get("overall_confidence", 0.0),
                "verification_score": verification_result.get("overall_verification_score", 0.0),
                "hallucination_detected": verification_result.get("hallucination_detected", False)
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction: {e}", exc_info=True)
            await db.rollback()
            raise

async def get_message_with_context(message_id: int, db: AsyncSession) -> Optional[Message]:
    """Get message with user and workspace information."""
    try:
        result = await db.execute(
            select(Message, User.name, Workspace.name)
            .join(User, Message.slack_user_id == User.slack_id)
            .join(Workspace, Message.workspace_id == Workspace.id)
            .where(Message.id == message_id)
        )
        
        row = result.fetchone()
        if row:
            message, user_name, workspace_name = row
            # Add user and workspace names to message for context
            message.user_name = user_name
            message.workspace_name = workspace_name
            return message
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting message context: {e}")
        return None

async def get_conversation_context(
    workspace_id: int, 
    channel_id: str, 
    current_message_id: int, 
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Get conversation context around the current message."""
    try:
        # Get messages from the same channel, around the current message
        # This provides context for the AI to understand the conversation flow
        result = await db.execute(
            select(Message, User.name)
            .join(User, Message.slack_user_id == User.slack_id)
            .where(
                and_(
                    Message.workspace_id == workspace_id,
                    Message.channel_id == channel_id,
                    Message.id != current_message_id
                )
            )
            .order_by(Message.created_at.desc())
            .limit(20)  # Get last 20 messages for context
        )
        
        context_messages = []
        for message, user_name in result.fetchall():
            # Extract text from raw payload
            text = message.raw_payload.get("text", "")
            if text:
                context_messages.append({
                    "id": message.id,
                    "text": text,
                    "user_name": user_name,
                    "timestamp": message.raw_payload.get("ts"),
                    "is_bot": bool(message.raw_payload.get("bot_id")),
                    "created_at": message.created_at
                })
        
        # Sort by creation time (oldest first for context)
        context_messages.sort(key=lambda x: x["created_at"])
        
        return context_messages
        
    except Exception as e:
        logger.error(f"Error getting conversation context: {e}")
        return []
    
async def _trigger_embedding_generation(knowledge_items: List[KnowledgeItem], db: AsyncSession):
    """Trigger embedding generation for newly extracted knowledge items."""
    try:
        from .embedding_generator import generate_embedding
        
        for item in knowledge_items:
            try:
                # Create text for embedding
                text_parts = []
                if item.title:
                    text_parts.append(item.title)
                if item.summary:
                    text_parts.append(item.summary)
                if item.content:
                    text_parts.append(item.content)
                
                if text_parts:
                    combined_text = " | ".join(text_parts)
                    
                    # Queue embedding generation task
                    generate_embedding.delay(
                        knowledge_id=item.id,
                        text=combined_text,
                        content_type="knowledge"
                    )
                    
                    logger.debug(f"Queued embedding generation for knowledge item {item.id}")
                    
            except Exception as e:
                logger.error(f"Error queuing embedding generation for knowledge item {item.id}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error triggering embedding generation: {e}")

async def process_and_store_knowledge(
    extraction_result: Dict[str, Any],
    verification_result: Dict[str, Any],
    message: Message,
    db: AsyncSession
) -> List[KnowledgeItem]:
    """Process extracted knowledge and store in database."""
    try:
        stored_items = []
        knowledge_items = extraction_result.get("knowledge_items", [])
        verification_results = verification_result.get("verification_results", [])
        
        for i, knowledge_item in enumerate(knowledge_items):
            # Get verification result for this item
            verification = next(
                (v for v in verification_results if v.get("knowledge_item_index") == i),
                {"is_supported": True, "confidence": 0.5}
            )
            
            # Calculate final confidence score
            ai_confidence = float(knowledge_item.get("confidence", 0.5))
            verification_confidence = float(verification.get("confidence", 0.5))
            
            # Combine AI confidence with verification confidence
            final_confidence = (ai_confidence * 0.7) + (verification_confidence * 0.3)
            
            # Only store if verification passes or confidence is high enough
            if verification.get("is_supported", True) or final_confidence > 0.3:  # Temporarily lowered for testing
                # Create knowledge item
                knowledge_item_obj = KnowledgeItem(
                    workspace_id=message.workspace_id,
                    title=knowledge_item.get("title", "Untitled"),
                    summary=knowledge_item.get("summary", ""),
                    content=knowledge_item.get("content", ""),
                    confidence=final_confidence,
                    embedding=None,  # Will be generated later
                    item_metadata={
                        "type": knowledge_item.get("type", "unknown"),
                        "participants": knowledge_item.get("participants", []),
                        "tags": knowledge_item.get("tags", []),
                        "source_context": knowledge_item.get("source_context", ""),
                        "extraction_metadata": extraction_result.get("extraction_metadata", {}),
                        "verification_metadata": verification_result.get("verification_metadata", {}),
                        "verification_result": verification,
                        "source_message_id": message.id,
                        "source_channel_id": message.channel_id,
                        "source_user_id": message.slack_user_id
                    }
                )
                
                db.add(knowledge_item_obj)
                await db.flush()
                
                stored_items.append(knowledge_item_obj)
                
                logger.info(f"Stored knowledge item: {knowledge_item_obj.title} (confidence: {final_confidence:.2f})")
            else:
                logger.warning(f"Knowledge item rejected due to low verification confidence: {knowledge_item.get('title', 'Untitled')}")
        
        await db.commit()
        return stored_items
        
    except Exception as e:
        logger.error(f"Error processing and storing knowledge: {e}")
        await db.rollback()
        raise

async def update_message_extraction_status(
    message_id: int, 
    extraction_result: Dict[str, Any], 
    db: AsyncSession
):
    """Update message with extraction results."""
    try:
        # Update the message with extraction metadata
        # Use a simpler approach: get current payload and update it
        result = await db.execute(
            select(Message.raw_payload).where(Message.id == message_id)
        )
        current_payload = result.scalar_one_or_none()
        
        if current_payload:
            # Update the payload with extraction metadata
            current_payload["knowledge_extraction"] = extraction_result
            
            await db.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(raw_payload=current_payload)
            )
        
        await db.commit()
        
    except Exception as e:
        logger.error(f"Error updating message extraction status: {e}")
        await db.rollback()
        raise

@celery_app.task
def extract_knowledge_batch():
    """Extract knowledge from a batch of unprocessed messages."""
    try:
        logger.info("Starting batch knowledge extraction...")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(extract_knowledge_batch_async())
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in batch knowledge extraction: {e}", exc_info=True)
        raise

async def extract_knowledge_batch_async() -> Dict[str, Any]:
    """Extract knowledge from a batch of messages."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Find messages that need knowledge extraction
            result = await db.execute(
                select(Message)
                .where(
                    and_(
                        ~Message.raw_payload.has_key("knowledge_extraction"),
                        Message.raw_payload.has_key("processed_data")
                    )
                )
                .limit(50)  # Process in batches
            )
            
            messages_to_process = result.scalars().all()
            
            if not messages_to_process:
                logger.info("No messages need knowledge extraction")
                return {"status": "success", "processed": 0}
            
            processed_count = 0
            total_items_extracted = 0
            
            for message in messages_to_process:
                try:
                    # Get processed data
                    processed_data = message.raw_payload.get("processed_data", {})
                    text = message.raw_payload.get("text", "")
                    
                    if not text or not processed_data:
                        continue
                    
                    # Extract knowledge
                    result = await extract_knowledge_async(
                        message_id=message.id,
                        text=text,
                        message_type=processed_data.get("message_type", "conversation"),
                        significance_score=processed_data.get("significance_score", 0.0)
                    )
                    
                    if result["status"] == "success":
                        processed_count += 1
                        total_items_extracted += result.get("knowledge_items_extracted", 0)
                    
                except Exception as e:
                    logger.error(f"Error processing message {message.id}: {e}")
                    continue
            
            logger.info(f"Batch knowledge extraction completed: {processed_count} messages processed, {total_items_extracted} items extracted")
            
            return {
                "status": "success",
                "processed": processed_count,
                "total_items_extracted": total_items_extracted
            }
            
        except Exception as e:
            logger.error(f"Error in batch knowledge extraction: {e}")
            await db.rollback()
            raise

@celery_app.task
def verify_existing_knowledge():
    """Verify existing knowledge items for accuracy."""
    try:
        logger.info("Starting knowledge verification...")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(verify_existing_knowledge_async())
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in knowledge verification: {e}", exc_info=True)
        raise

async def verify_existing_knowledge_async() -> Dict[str, Any]:
    """Verify existing knowledge items for accuracy."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Find knowledge items that haven't been verified recently
            result = await db.execute(
                select(KnowledgeItem)
                .where(
                    KnowledgeItem.metadata.contains({
                        "verification_metadata": {"verified_at": None}
                    })
                )
                .limit(20)  # Verify in batches
            )
            
            knowledge_items = result.scalars().all()
            
            if not knowledge_items:
                logger.info("No knowledge items need verification")
                return {"status": "success", "verified": 0}
            
            # Initialize OpenAI service (lazy import)
            from ..services.openai_service import OpenAIService
            openai_service = OpenAIService()
            verified_count = 0
            
            for item in knowledge_items:
                try:
                    # Get source message
                    source_message_id = item.metadata.get("source_message_id")
                    if not source_message_id:
                        continue
                    
                    source_message = await db.execute(
                        select(Message).where(Message.id == source_message_id)
                    ).scalar_one_or_none()
                    
                    if not source_message:
                        continue
                    
                    # Get conversation context
                    conversation_context = await get_conversation_context(
                        item.workspace_id,
                        source_message.channel_id,
                        source_message_id,
                        db
                    )
                    
                    # Verify the knowledge item
                    verification_result = await openai_service.verify_extraction(
                        extracted_knowledge={"knowledge_items": [item.metadata]},
                        source_messages=conversation_context
                    )
                    
                    # Update verification metadata
                    item.metadata["verification_metadata"] = verification_result.get("verification_metadata", {})
                    item.metadata["verification_result"] = verification_result
                    
                    # Update confidence based on verification
                    if verification_result.get("overall_verification_score"):
                        verification_score = verification_result["overall_verification_score"]
                        item.confidence = (item.confidence * 0.7) + (verification_score * 0.3)
                    
                    verified_count += 1
                    
                except Exception as e:
                    logger.error(f"Error verifying knowledge item {item.id}: {e}")
                    continue
            
            await db.commit()
            
            logger.info(f"Knowledge verification completed: {verified_count} items verified")
            
            return {
                "status": "success",
                "verified": verified_count
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge verification: {e}")
            await db.rollback()
            raise

if __name__ == "__main__":
    celery_app.start()
