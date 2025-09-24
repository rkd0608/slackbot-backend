"""Simplified query processor focused on team knowledge extraction and retrieval."""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..models.base import Query, KnowledgeItem, User, Workspace
from ..services.vector_service import VectorService
from ..services.slack_service import SlackService
from ..services.openai_service import OpenAIService
from .simplified_celery_app import celery_app

def simple_sanitize(text: str) -> str:
    """Simple text sanitization for queries."""
    import re
    if not text:
        return ""
    
    # Remove potential script tags and basic XSS
    text = re.sub(r'<[^>]*>', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Limit length
    text = text[:500]
    
    return text

def get_async_session():
    """Create a new async session for each task."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal

@celery_app.task
def process_query_async(
    query_id: Optional[int],
    workspace_id: int,
    user_id: int,
    channel_id: str,
    query_text: str,
    response_url: Optional[str] = None,
    is_slash_command: bool = False
):
    """Process a user query with simplified, focused approach."""
    try:
        logger.info(f"Processing query: {query_text} from user {user_id}")
        
        # Run the async function in a new event loop
        return asyncio.run(process_query(
            query_id, workspace_id, user_id, channel_id, query_text, response_url, is_slash_command
        ))
        
    except Exception as e:
        logger.error(f"Error in process_query_async: {e}")
        return {"status": "error", "message": str(e)}

async def process_query(
    query_id: Optional[int],
    workspace_id: int,
    user_id: int,
    channel_id: str,
    query_text: str,
    response_url: Optional[str] = None,
    is_slash_command: bool = False
) -> Dict[str, Any]:
    """
    Simplified query processing pipeline:
    1. Parse user query
    2. Search existing knowledge base
    3. Generate response with proper attribution
    4. Return formatted answer
    """
    
    async_session = get_async_session()
    
    async with async_session() as db:
        try:
            # Basic security check - just clean the query text
            sanitized_query = simple_sanitize(query_text)
            
            # Create or get query record
            if query_id:
                query_result = await db.execute(select(Query).where(Query.id == query_id))
                query = query_result.scalar_one_or_none()
            else:
                query = Query(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    text=sanitized_query,
                    created_at=datetime.utcnow()
                )
                db.add(query)
                await db.flush()
                query_id = query.id
            
            # Search knowledge base
            vector_service = VectorService()
            search_results = await vector_service.hybrid_search(
                query=sanitized_query,
                channel_id=channel_id,
                workspace_id=workspace_id,
                limit=5
            )
            
            # If we don't have enough results, check recent channel activity
            if len(search_results) < 2:
                from ..services.simple_knowledge_extractor import SimpleKnowledgeExtractor
                extractor = SimpleKnowledgeExtractor()
                recent_insights = await extractor.extract_from_recent_messages(
                    channel_id, workspace_id, db, hours_back=6
                )
                
                # Add recent insights to search results
                for insight in recent_insights:
                    search_results.append({
                        'knowledge_item': {
                            'content': insight.get('content', ''),
                            'metadata': {
                                'source_channel': f"#{channel_id}",
                                'created_date': datetime.now().strftime('%Y-%m-%d'),
                                'participants': ['recent discussion'],
                                'type': insight.get('type', 'recent_insight')
                            }
                        },
                        'score': 0.8  # High relevance for recent content
                    })
            
            # Generate response
            response_data = await generate_focused_response(
                sanitized_query, search_results, workspace_id, db
            )
            
            # Update query with response
            if query:
                query.response = response_data
                # Note: Query model doesn't have status/completed_at fields in current schema
            
            await db.commit()
            
            # Send response to Slack
            if response_url:
                await send_slack_response(response_url, response_data)
            
            # Also send to channel if it's a slash command
            if is_slash_command:
                await send_channel_response(channel_id, workspace_id, response_data)
            
            logger.info(f"Successfully processed query {query_id}")
            
            return {
                "status": "success",
                "query_id": query_id,
                "response": response_data
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            await db.rollback()
            
            # Send error response
            error_response = {
                "response_type": "ephemeral",
                "text": "❌ Sorry, I encountered an error processing your question. Please try again."
            }
            
            if response_url:
                await send_slack_response(response_url, error_response)
                
            return {"status": "error", "message": str(e)}

async def generate_focused_response(
    query_text: str,
    search_results: List[Dict[str, Any]],
    workspace_id: int,
    db: AsyncSession
) -> Dict[str, Any]:
    """Generate a focused response based on team knowledge."""
    
    if not search_results:
        return {
            "response_type": "in_channel",
            "text": "🔍 **No Information Found**\n\nI couldn't find any relevant team discussions about this topic. Try asking in the channel or check if this has been discussed before.",
            "blocks": []
        }
    
    # Extract relevant knowledge
    relevant_knowledge = []
    for result in search_results:
        if result.get('score', 0) > 0.7:  # Only high-confidence results
            relevant_knowledge.append(result)
    
    if not relevant_knowledge:
        return {
            "response_type": "in_channel", 
            "text": f"🤔 **Limited Information Found**\n\nI found some related discussions but nothing directly answering: *{query_text}*\n\nTry rephrasing your question or ask team members directly.",
            "blocks": []
        }
    
    # Generate AI response using found knowledge
    openai_service = OpenAIService()
    
    system_prompt = """You are a team knowledge assistant. Your ONLY job is to help teams find information from their own conversations.

CRITICAL RULES:
1. ONLY use information from the provided search results
2. Be specific about who said what and when
3. Include channel names and dates
4. If information is incomplete, say so clearly
5. Never make up or infer information not in the results
6. Focus on decisions, processes, and specific details teams shared

Response format:
- Start with a clear answer if you have one
- Include specific quotes and details
- Always cite sources (who, when, where)
- Be concise and actionable"""

    context_text = ""
    for item in relevant_knowledge:
        knowledge = item.get('knowledge_item', {})
        metadata = knowledge.get('metadata', {})
        
        context_text += f"""
Source: {metadata.get('source_channel', 'Unknown channel')} on {metadata.get('created_date', 'Unknown date')}
Participants: {', '.join(metadata.get('participants', []))}
Content: {knowledge.get('content', '')}
---
"""

    try:
        response = await openai_service.chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query_text}\n\nTeam Knowledge:\n{context_text}"}
        ], model="gpt-3.5-turbo", max_tokens=500, temperature=0.1)
        
        ai_response = response['choices'][0]['message']['content']
        
        return {
            "response_type": "in_channel",
            "text": ai_response,
            "blocks": []
        }
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        
        # Fallback to simple knowledge listing
        response_text = f"📚 **Found Information About: {query_text}**\n\n"
        
        for item in relevant_knowledge[:3]:  # Limit to top 3
            knowledge = item.get('knowledge_item', {})
            metadata = knowledge.get('metadata', {})
            
            response_text += f"**From {metadata.get('source_channel', 'team discussion')}:**\n"
            response_text += f"{knowledge.get('content', '')[:200]}...\n\n"
        
        return {
            "response_type": "in_channel",
            "text": response_text,
            "blocks": []
        }

async def send_slack_response(response_url: str, response_data: Dict[str, Any]):
    """Send response to Slack using the response URL."""
    try:
        import aiohttp
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                response_url, 
                json=response_data,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    logger.info("Successfully sent response to Slack")
                else:
                    response_text = await resp.text()
                    logger.error(f"Failed to send response to Slack: {resp.status} - {response_text}")
                    
    except Exception as e:
        logger.error(f"Error sending response to Slack: {e}")

async def send_channel_response(channel_id: str, workspace_id: int, response_data: Dict[str, Any]):
    """Send response to the Slack channel."""
    try:
        slack_service = SlackService()
        
        # Get workspace token
        async_session = get_async_session()
        async with async_session() as db:
            workspace_result = await db.execute(
                select(Workspace).where(Workspace.id == workspace_id)
            )
            workspace = workspace_result.scalar_one_or_none()
            
            if workspace and workspace.slack_bot_token:
                await slack_service.send_message(
                    channel_id=channel_id,
                    text=response_data.get("text", "Response generated"),
                    blocks=response_data.get("blocks", []),
                    token=workspace.slack_bot_token
                )
                logger.info(f"Sent response to channel {channel_id}")
            else:
                logger.error(f"No bot token found for workspace {workspace_id}")
                
    except Exception as e:
        logger.error(f"Error sending channel response: {e}")

# Periodic task to process any pending queries
@celery_app.task
def process_pending_queries():
    """Process any queries that might be stuck in pending state."""
    try:
        logger.info("Processing pending queries...")
        return asyncio.run(process_pending_queries_async())
    except Exception as e:
        logger.error(f"Error in process_pending_queries: {e}")
        return {"status": "error", "message": str(e)}

async def process_pending_queries_async():
    """Check for and process any pending queries."""
    async_session = get_async_session()
    
    async with async_session() as db:
        try:
            # For now, just return success since Query model doesn't have status field
            # In a production system, you'd add status tracking to the Query model
            logger.info("Pending query processing - simplified implementation")
            return {"status": "success", "processed_count": 0}
            
        except Exception as e:
            logger.error(f"Error in process_pending_queries_async: {e}")
            return {"status": "error", "message": str(e)}
