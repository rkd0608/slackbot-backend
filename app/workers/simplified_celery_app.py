"""Simplified Celery application configuration focused on core functionality."""

from celery import Celery
from celery.schedules import crontab
import os
from loguru import logger

# Celery configuration
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Create Celery app with only essential workers
celery_app = Celery(
    "slack-knowledge-bot",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "app.workers.simple_message_processor",    # Process incoming messages
        "app.workers.simple_knowledge_extractor",  # Extract knowledge from conversations
        "app.workers.simple_query_processor",      # Handle user queries (simplified)
        "app.workers.conversation_backfill"        # Backfill historical conversations
    ]
)

# Simplified Celery configuration
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Simplified task routing - only two queues
    task_routes={
        "app.workers.simple_query_processor.*": {"queue": "queries"},  # High priority user queries
        "*": {"queue": "background"},  # Everything else runs in background
    },
    
    # Queue configuration
    task_default_queue="background",
    task_queues={
        "queries": {"exchange": "queries", "routing_key": "queries"},        # User-facing queries
        "background": {"exchange": "background", "routing_key": "background"} # Background processing
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Task timeouts
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
)

# Simplified periodic tasks - only essential ones
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up essential periodic tasks only."""
    logger.info("Setting up periodic tasks...")
    
    # Process conversations for knowledge extraction every 30 minutes
    sender.add_periodic_task(
        1800.0,  # 30 minutes
        "app.workers.simple_knowledge_extractor.extract_knowledge_from_conversations",
        name="extract-knowledge-every-30min"
    )
    
    # Process any stuck queries every 5 minutes
    sender.add_periodic_task(
        300.0,  # 5 minutes
        process_pending_queries.s(),
        name="process-pending-queries-every-5min"
    )
    
    # Clean up old results every 24 hours
    sender.add_periodic_task(
        crontab(hour=2, minute=0),  # 2 AM daily
        cleanup_old_data.s(),
        name="cleanup-daily"
    )

# Import tasks to register them
from .simple_query_processor import process_query_async, process_pending_queries
from .simple_message_processor import process_message_async

@celery_app.task
def cleanup_old_data():
    """Clean up old query results and temporary data."""
    try:
        logger.info("Starting daily cleanup...")
        # This would clean up old query results, expired cache entries, etc.
        # For now, just log that cleanup ran
        logger.info("Daily cleanup completed")
        return {"status": "success", "cleaned_items": 0}
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return {"status": "error", "message": str(e)}

# Health check task
@celery_app.task
def health_check():
    """Basic health check task."""
    return {"status": "healthy", "timestamp": "2025-01-05T12:00:00Z"}

if __name__ == "__main__":
    celery_app.start()
