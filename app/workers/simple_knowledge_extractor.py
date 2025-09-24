"""Simplified knowledge extraction worker focused on processing conversations."""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

from .simplified_celery_app import celery_app
from ..core.config import settings
from ..services.simple_knowledge_extractor import SimpleKnowledgeExtractor

def get_async_session():
    """Create a new async session for each task."""
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal

@celery_app.task
def extract_knowledge_from_conversations():
    """
    Main task to extract knowledge from completed conversations.
    Runs periodically to process conversations and build knowledge base.
    """
    try:
        logger.info("Starting knowledge extraction from conversations")
        return asyncio.run(extract_knowledge_from_conversations_async())
    except Exception as e:
        logger.error(f"Error in extract_knowledge_from_conversations: {e}")
        return {"status": "error", "message": str(e)}

async def extract_knowledge_from_conversations_async():
    """Extract knowledge from conversations for all workspaces."""
    async_session = get_async_session()
    
    async with async_session() as db:
        try:
            from ..models.base import Workspace
            from sqlalchemy import select
            
            # Get all active workspaces
            workspace_query = select(Workspace)
            result = await db.execute(workspace_query)
            workspaces = result.scalars().all()
            
            total_extracted = 0
            extractor = SimpleKnowledgeExtractor()
            
            for workspace in workspaces:
                try:
                    logger.info(f"Processing knowledge extraction for workspace {workspace.id}")
                    
                    # Extract knowledge from this workspace's conversations
                    knowledge_items = await extractor.extract_from_conversations(
                        workspace.id, db, batch_size=5
                    )
                    
                    total_extracted += len(knowledge_items)
                    logger.info(f"Extracted {len(knowledge_items)} items from workspace {workspace.id}")
                    
                except Exception as e:
                    logger.error(f"Error processing workspace {workspace.id}: {e}")
                    continue
            
            logger.info(f"Knowledge extraction completed. Total items extracted: {total_extracted}")
            
            return {
                "status": "success",
                "workspaces_processed": len(workspaces),
                "total_extracted": total_extracted
            }
            
        except Exception as e:
            logger.error(f"Error in extract_knowledge_from_conversations_async: {e}")
            return {"status": "error", "message": str(e)}

@celery_app.task
def extract_knowledge_for_workspace(workspace_id: int):
    """Extract knowledge for a specific workspace."""
    try:
        logger.info(f"Starting knowledge extraction for workspace {workspace_id}")
        return asyncio.run(extract_knowledge_for_workspace_async(workspace_id))
    except Exception as e:
        logger.error(f"Error in extract_knowledge_for_workspace: {e}")
        return {"status": "error", "message": str(e)}

async def extract_knowledge_for_workspace_async(workspace_id: int):
    """Extract knowledge for a specific workspace."""
    async_session = get_async_session()
    
    async with async_session() as db:
        try:
            extractor = SimpleKnowledgeExtractor()
            
            knowledge_items = await extractor.extract_from_conversations(
                workspace_id, db, batch_size=10
            )
            
            logger.info(f"Extracted {len(knowledge_items)} items from workspace {workspace_id}")
            
            return {
                "status": "success",
                "workspace_id": workspace_id,
                "items_extracted": len(knowledge_items)
            }
            
        except Exception as e:
            logger.error(f"Error extracting knowledge for workspace {workspace_id}: {e}")
            return {"status": "error", "message": str(e)}

@celery_app.task  
def extract_from_recent_channel_activity(channel_id: str, workspace_id: int):
    """Extract insights from recent activity in a specific channel."""
    try:
        logger.info(f"Extracting from recent activity in channel {channel_id}")
        return asyncio.run(extract_from_recent_channel_activity_async(channel_id, workspace_id))
    except Exception as e:
        logger.error(f"Error in extract_from_recent_channel_activity: {e}")
        return {"status": "error", "message": str(e)}

async def extract_from_recent_channel_activity_async(channel_id: str, workspace_id: int):
    """Extract insights from recent channel activity."""
    async_session = get_async_session()
    
    async with async_session() as db:
        try:
            extractor = SimpleKnowledgeExtractor()
            
            insights = await extractor.extract_from_recent_messages(
                channel_id, workspace_id, db, hours_back=4
            )
            
            logger.info(f"Extracted {len(insights)} insights from recent activity in channel {channel_id}")
            
            return {
                "status": "success",
                "channel_id": channel_id,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error extracting from recent channel activity: {e}")
            return {"status": "error", "message": str(e)}
