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
            
            # Search knowledge base using text search (more reliable)
            vector_service = VectorService()
            search_results = await vector_service._text_search(
                query=sanitized_query,
                workspace_id=workspace_id,
                channel_id=channel_id,
                limit=5,
                db=db
            )
            
            logger.info(f"Found {len(search_results)} knowledge items for query: {sanitized_query}")
            
            # Generate response
            response_data = await generate_focused_response(
                sanitized_query, search_results, workspace_id, db, query_id
            )
            
            # Update query with response
            if query:
                query.response = response_data
                # Note: Query model doesn't have status/completed_at fields in current schema
            
            await db.commit()
            
            # Send response to Slack
            if response_url:
                await send_slack_response(response_url, response_data)
            
            # Send to channel for both slash commands and mentions
            if is_slash_command or not response_url:  # mentions don't have response_url
                await send_channel_response(channel_id, workspace_id, response_data)
            
            logger.info(f"Successfully processed query {query_id}")
            
            return {
                "status": "success",
                "query_id": query_id,
                "response": response_data
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            try:
                await db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
            
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
    db: AsyncSession,
    query_id: Optional[int] = None
) -> Dict[str, Any]:
    """Generate a focused response based on team knowledge."""
    
    if not search_results:
        return {
            "response_type": "in_channel",
            "text": "🔍 **No Information Found**\n\nI couldn't find any relevant team discussions about this topic. Try asking in the channel or check if this has been discussed before.",
            "blocks": []
        }
    
    # Extract relevant knowledge (use confidence score for filtering)
    relevant_knowledge = []
    for result in search_results:
        confidence = result.get('confidence', 0)
        similarity = result.get('similarity', 0)
        # Consider results with high confidence OR high similarity
        if confidence > 0.6 or similarity > 0.7:
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
        # Handle both knowledge items and conversation results
        if 'knowledge_item' in item:
            # Old format from recent insights
            knowledge = item.get('knowledge_item', {})
            metadata = knowledge.get('metadata', {})
            content = knowledge.get('content', '')
        else:
            # New format from search results
            metadata = item.get('metadata', {})
            content = item.get('content', '')
        
        context_text += f"""
Source: {metadata.get('source_channel_id', 'Unknown channel')} on {metadata.get('created_date', 'Unknown date')}
Participants: {', '.join(metadata.get('participants', []))}
Content: {content}
Type: {item.get('type', 'unknown')}
Confidence: {item.get('confidence', 0)}
---
"""

    try:
        user_message = f"Question: {query_text}\n\nTeam Knowledge:\n{context_text}"
        logger.info(f"Sending to AI - Question: {query_text}")
        logger.info(f"Context length: {len(context_text)} characters")
        logger.info(f"Number of knowledge items: {len(relevant_knowledge)}")
        
        response = await openai_service._make_request(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        ai_response = response['choices'][0]['message']['content']
        
        # Add source attribution
        sources_text = "\n\n📚 **Sources:**\n"
        for i, item in enumerate(relevant_knowledge[:3], 1):
            metadata = item.get('metadata', {})
            channel_id = metadata.get('source_channel_id', 'unknown')
            created_date = metadata.get('created_date', 'unknown')
            if created_date != 'unknown':
                # Format the date nicely
                try:
                    from datetime import datetime
                    parsed_date = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                    formatted_date = parsed_date.strftime('%Y-%m-%d')
                except:
                    formatted_date = created_date
            else:
                formatted_date = 'unknown'
            sources_text += f"{i}. #{channel_id} on {formatted_date}\n"
        
        # Add interactive button for viewing full sources
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ai_response + sources_text
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "👍 Helpful"
                        },
                        "style": "primary",
                        "action_id": f"feedback_helpful_{query_id}"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "👎 Not Helpful"
                        },
                        "action_id": f"feedback_not_helpful_{query_id}"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "🔍 View Sources"
                        },
                        "action_id": f"view_sources_{query_id}"
                    }
                ]
            }
        ]
        
        return {
            "response_type": "in_channel",
            "text": ai_response + sources_text,
            "blocks": blocks
        }
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        
        # Fallback to simple knowledge listing
        response_text = f"📚 **Found Information About: {query_text}**\n\n"
        
        for item in relevant_knowledge[:3]:  # Limit to top 3
            # Handle both knowledge items and conversation results
            if 'knowledge_item' in item:
                knowledge = item.get('knowledge_item', {})
                metadata = knowledge.get('metadata', {})
                content = knowledge.get('content', '')
            else:
                metadata = item.get('metadata', {})
                content = item.get('content', '')
            
            response_text += f"**From {metadata.get('source_channel_id', 'team discussion')}:**\n"
            response_text += f"{content[:200]}...\n\n"
        
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
            
            if workspace and workspace.tokens.get('access_token'):
                await slack_service.send_message(
                    channel=channel_id,
                    text=response_data.get("text", "Response generated"),
                    blocks=response_data.get("blocks", []),
                    token=workspace.tokens.get('access_token')
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
