#!/usr/bin/env python3
"""
Manual Knowledge Extraction Script

This script manually triggers knowledge extraction for existing conversations
that have substantial content but haven't been processed yet.
"""

import asyncio
import sys
import os
from datetime import datetime
from loguru import logger

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from app.core.database import get_session_factory
from app.services.enhanced_conversation_state_manager import EnhancedConversationStateManager
from app.services.conversation_knowledge_extractor import ConversationKnowledgeExtractor
from app.models.base import Conversation, Message, KnowledgeItem
from sqlalchemy import select, func, and_

async def extract_knowledge_chunked(conversation_id: int, workspace_id: int, message_count: int, db):
    """Extract knowledge from large conversations by processing them in chunks."""
    try:
        logger.info(f"  📦 Chunking conversation {conversation_id} with {message_count} messages")
        
        # Get conversation details first
        conv_result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = conv_result.scalar_one_or_none()
        
        if not conversation:
            logger.warning(f"  ⚠️ Conversation {conversation_id} not found")
            return
        
        # Get all messages from the conversation
        messages_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        messages = messages_result.scalars().all()
        
        if not messages:
            logger.warning(f"  ⚠️ No messages found for conversation {conversation_id}")
            return
        
        # Process in chunks of 40 messages (gpt-4o has 128K token limit)
        chunk_size = 40
        chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
        
        logger.info(f"  📦 Split into {len(chunks)} chunks of ~{chunk_size} messages each")
        
        total_extracted = 0
        for i, chunk in enumerate(chunks):
            logger.info(f"  🔍 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} messages)")
            
            try:
                # Create a properly formatted conversation context for this chunk
                # Format messages as text for the AI extractor
                formatted_messages = []
                for msg in chunk:
                    timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M")
                    user_id = msg.slack_user_id[-4:]  # Last 4 chars for brevity
                    content = msg.content[:500]  # Truncate very long messages
                    formatted_messages.append(f"[{timestamp}] {user_id}: {content}")
                
                conversation_text = "\n".join(formatted_messages)
                
                chunk_context = {
                    "conversation_id": conversation_id,
                    "workspace_id": workspace_id,
                    "conversation_text": conversation_text,  # This is what the AI extractor expects
                    "messages": [
                        {
                            "content": msg.content[:500],  # Truncate long messages
                            "user_id": msg.slack_user_id,
                            "timestamp": msg.created_at.isoformat(),
                            "metadata": msg.message_metadata or {}
                        }
                        for msg in chunk
                    ],
                    "metadata": {
                        "chunk_number": i + 1,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk),
                        "date_range": f"{chunk[0].created_at.strftime('%Y-%m-%d')} to {chunk[-1].created_at.strftime('%Y-%m-%d')}",
                        "extraction_method": "chunked_manual"
                    }
                }
                
                # Use the knowledge extractor's AI method directly
                knowledge_extractor = ConversationKnowledgeExtractor()
                knowledge_data = await knowledge_extractor._ai_extract_knowledge(chunk_context)
                
                if knowledge_data and knowledge_data.get('knowledge_items'):
                    items = knowledge_data['knowledge_items']
                    logger.info(f"    ✅ Extracted {len(items)} knowledge items from chunk {i+1}")
                    
                    # Store each knowledge item
                    for item_data in items:
                        if item_data.get('title') and item_data.get('content'):
                            knowledge_item = KnowledgeItem(
                                conversation_id=conversation_id,
                                workspace_id=workspace_id,
                                title=item_data['title'][:200],  # Truncate title
                                content=item_data['content'],
                                summary=item_data.get('summary', '')[:500],  # Truncate summary
                                tags=item_data.get('tags', []),
                                confidence_score=item_data.get('confidence', 0.7),
                                extraction_metadata={
                                    'method': 'chunked_manual_extraction',
                                    'chunk_number': i + 1,
                                    'total_chunks': len(chunks),
                                    'message_count': len(chunk),
                                    'source_channel_id': conversation.slack_channel_id,
                                    'source_channel_name': conversation.slack_channel_name
                                }
                            )
                            db.add(knowledge_item)
                            total_extracted += 1
                    
                    await db.commit()
                    logger.info(f"    💾 Saved {len(items)} knowledge items from chunk {i+1}")
                else:
                    logger.info(f"    ⚠️ No knowledge extracted from chunk {i+1}")
                
                # Add delay between chunks to avoid rate limits
                if i < len(chunks) - 1:  # Don't wait after the last chunk
                    logger.info(f"    ⏳ Waiting 3 seconds before next chunk...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                logger.error(f"    ❌ Error processing chunk {i+1}: {e}")
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    logger.info(f"    ⏳ Rate limit hit, waiting 60 seconds...")
                    await asyncio.sleep(60)
                continue
        
        logger.info(f"  🎉 Chunked extraction completed! Total items extracted: {total_extracted}")
        
    except Exception as e:
        logger.error(f"  ❌ Chunked extraction failed: {e}")

async def main():
    """Main function to manually extract knowledge from conversations."""
    logger.info("🚀 Starting manual knowledge extraction process")
    
    # Initialize services
    state_manager = EnhancedConversationStateManager()
    knowledge_extractor = ConversationKnowledgeExtractor()
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        try:
            # Get all conversations with message counts
            logger.info("📊 Analyzing conversations...")
            
            conversations_query = select(
                Conversation.id,
                Conversation.workspace_id,
                Conversation.slack_channel_id,
                Conversation.message_count,
                Conversation.participant_count,
                func.count(Message.id).label('actual_message_count'),
                func.count(func.distinct(Message.slack_user_id)).label('actual_participant_count')
            ).join(
                Message, Conversation.id == Message.conversation_id
            ).group_by(
                Conversation.id,
                Conversation.workspace_id,
                Conversation.slack_channel_id,
                Conversation.message_count,
                Conversation.participant_count
            ).having(
                func.count(Message.id) >= 5  # At least 5 messages
            ).order_by(
                func.count(Message.id).desc()
            )
            
            result = await db.execute(conversations_query)
            conversations = result.fetchall()
            
            logger.info(f"Found {len(conversations)} conversations with substantial content")
            
            for conv in conversations:
                conv_id = conv.id
                workspace_id = conv.workspace_id
                channel_id = conv.slack_channel_id
                stored_msg_count = conv.message_count or 0
                stored_participant_count = conv.participant_count or 0
                actual_msg_count = conv.actual_message_count
                actual_participant_count = conv.actual_participant_count
                
                logger.info(f"\n📝 Processing conversation {conv_id}:")
                logger.info(f"  Channel: {channel_id}")
                logger.info(f"  Stored counts: {stored_msg_count} messages, {stored_participant_count} participants")
                logger.info(f"  Actual counts: {actual_msg_count} messages, {actual_participant_count} participants")
                
                # Update conversation metadata if it's wrong
                if stored_msg_count != actual_msg_count or stored_participant_count != actual_participant_count:
                    logger.info(f"  🔧 Updating conversation metadata...")
                    conv_result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
                    conversation = conv_result.scalar_one_or_none()
                    if conversation:
                        conversation.message_count = actual_msg_count
                        conversation.participant_count = actual_participant_count
                        await db.commit()
                        logger.info(f"  ✅ Updated metadata: {actual_msg_count} messages, {actual_participant_count} participants")
                
                # Analyze conversation state
                logger.info(f"  🧠 Analyzing conversation state...")
                try:
                    boundary = await state_manager.analyze_conversation_state(conv_id, db)
                    logger.info(f"  📊 State: {boundary.state}, Confidence: {boundary.confidence:.2f}")
                    logger.info(f"  🎯 Topic: {boundary.topic or 'None'}")
                    
                    # Check if it should be extracted
                    should_extract = await state_manager.should_extract_knowledge(boundary)
                    logger.info(f"  🤔 Should extract: {should_extract}")
                    
                    if should_extract:
                        logger.info(f"  🔍 Extracting knowledge...")
                        try:
                            # For conversations with >80 messages, use chunked extraction (gpt-4o can handle much more)
                            if actual_msg_count > 80:
                                logger.info(f"  ⚠️ Large conversation ({actual_msg_count} messages) - using chunked extraction")
                                await extract_knowledge_chunked(conv_id, workspace_id, actual_msg_count, db)
                            else:
                                # Small conversations can be processed normally
                                extraction_results = await knowledge_extractor._extract_from_single_conversation(
                                    conversation_id=conv_id,
                                    workspace_id=workspace_id,
                                    db=db
                                )
                                
                                if extraction_results:
                                    logger.info(f"  ✅ Extracted knowledge from conversation!")
                                    logger.info(f"    - Results: {extraction_results}")
                                else:
                                    logger.info(f"  ⚠️ No knowledge extracted")
                                
                        except Exception as e:
                            logger.error(f"  ❌ Knowledge extraction failed: {e}")
                            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                                logger.info(f"  🔄 Rate limit hit - will retry with chunked extraction")
                                await extract_knowledge_chunked(conv_id, workspace_id, actual_msg_count, db)
                            else:
                                import traceback
                                logger.error(f"  Traceback: {traceback.format_exc()}")
                    else:
                        logger.info(f"  ⏭️ Skipping extraction (doesn't meet criteria)")
                        
                except Exception as e:
                    logger.error(f"  ❌ State analysis failed: {e}")
                
                # Add a small delay to avoid overwhelming the system
                await asyncio.sleep(1)
            
            logger.info(f"\n🎉 Manual knowledge extraction completed!")
            logger.info(f"Processed {len(conversations)} conversations")
            
        except Exception as e:
            logger.error(f"❌ Error in manual knowledge extraction: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())
