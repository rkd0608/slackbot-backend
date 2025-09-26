import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from .database import init_db, check_db_health
from ..services.backfill_service import BackfillService
from ..core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for FastAPI."""
    # Startup
    logger.info("Starting SlackBot Backend...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_db()
        
        # Check database health
        logger.info("Checking database health...")
        db_healthy = await check_db_health()
        if not db_healthy:
            logger.error("Database health check failed!")
            raise Exception("Database connection failed")
        logger.info("Database health check passed")
        
        # Trigger automatic conversation backfill
        logger.info("Triggering automatic conversation backfill...")
        await trigger_automatic_backfill()
        
        logger.info("SlackBot Backend startup completed")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SlackBot Backend...")
    # Add any cleanup logic here

async def trigger_automatic_backfill():
    """Automatically trigger conversation backfill on startup."""
    try:
        logger.info("Starting automatic conversation backfill check...")
        
        # Get database session
        async_session = AsyncSessionLocal()
        async with async_session() as db:
            # Initialize backfill service
            backfill_service = BackfillService()
            
            # Check and trigger backfill if needed
            result = await backfill_service.check_and_trigger_backfill(db)
            
            if result.get('status') == 'completed':
                logger.info(f"Automatic backfill check completed for {result.get('total_workspaces', 0)} workspaces")
                
                # Log details for each workspace
                for workspace_result in result.get('results', []):
                    workspace_name = workspace_result.get('workspace_name', 'Unknown')
                    backfill_needed = workspace_result.get('backfill_needed', False)
                    reason = workspace_result.get('reason', [])
                    
                    if backfill_needed:
                        logger.info(f"Backfill triggered for workspace '{workspace_name}': {', '.join(reason)}")
                    else:
                        logger.info(f"Workspace '{workspace_name}' is up to date")
                        
            elif result.get('status') == 'no_workspaces':
                logger.info("No workspaces found, skipping backfill")
            else:
                logger.warning(f"Unexpected backfill result: {result}")
                
    except Exception as e:
        logger.error(f"Error during automatic backfill: {e}", exc_info=True)
        # Don't fail startup for backfill errors
        logger.warning("Continuing startup despite backfill error")

async def check_backfill_status():
    """Check the current status of conversation backfill."""
    try:
        logger.info("Checking conversation backfill status...")
        
        # Get database session
        async_session = AsyncSessionLocal()
        async with async_session() as db:
            # Initialize backfill service
            backfill_service = BackfillService()
            
            # Get status
            status = await backfill_service.get_backfill_status(db)
            
            if status.get('status') == 'success':
                logger.info(f"Backfill Status:")
                logger.info(f"  Total Workspaces: {status.get('total_workspaces', 0)}")
                logger.info(f"  Total Conversations: {status.get('total_conversations', 0)}")
                logger.info(f"  Total Messages: {status.get('total_messages', 0)}")
                
                # Log workspace details
                for workspace in status.get('workspaces', []):
                    workspace_name = workspace.get('workspace_name', 'Unknown')
                    conversations = workspace.get('conversations_count', 0)
                    messages = workspace.get('messages_count', 0)
                    has_tokens = workspace.get('has_tokens', False)
                    
                    logger.info(f"  {workspace_name}: {conversations} conversations, {messages} messages, tokens: {'Yes' if has_tokens else 'No'}")
            else:
                logger.warning(f"Failed to get backfill status: {status}")
                
    except Exception as e:
        logger.error(f"Error checking backfill status: {e}", exc_info=True)

async def manual_backfill_trigger(workspace_id: int, channel_id: Optional[str] = None, days_back: int = 30):
    """Manually trigger conversation backfill."""
    try:
        logger.info(f"Manually triggering backfill for workspace {workspace_id}")
        
        # Get database session
        async_session = AsyncSessionLocal()
        async with async_session() as db:
            # Initialize backfill service
            backfill_service = BackfillService()
            
            if channel_id:
                # Trigger channel-specific backfill
                result = await backfill_service.trigger_channel_backfill(workspace_id, channel_id, days_back)
                logger.info(f"Channel backfill queued: {result}")
            else:
                # Trigger workspace-wide backfill
                result = await backfill_service.trigger_workspace_backfill(workspace_id, days_back)
                logger.info(f"Workspace backfill queued: {result}")
                
            return result
            
    except Exception as e:
        logger.error(f"Error triggering manual backfill: {e}", exc_info=True)
        raise
