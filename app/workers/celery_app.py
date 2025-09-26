"""Celery application configuration for background job processing."""

from celery import Celery
from celery.schedules import crontab
import os
from loguru import logger

# Celery configuration
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Create Celery app
celery_app = Celery(
    "slack-knowledge-bot",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "app.workers.message_processor",
        "app.workers.knowledge_extractor",
        "app.workers.embedding_generator",
        "app.workers.query_processor",
        "app.workers.conversation_backfill",
        "app.workers.accuracy_tester"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "app.workers.message_processor.*": {"queue": "messages"},
        "app.workers.knowledge_extractor.*": {"queue": "knowledge"},
        "app.workers.embedding_generator.*": {"queue": "embeddings"},
        "app.workers.query_processor.*": {"queue": "queries"},
        "app.workers.conversation_backfill.*": {"queue": "conversations"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues={
        "default": {"exchange": "default", "routing_key": "default"},
        "messages": {"exchange": "messages", "routing_key": "messages"},
        "knowledge": {"exchange": "knowledge", "routing_key": "knowledge"},
        "embeddings": {"exchange": "embeddings", "routing_key": "embeddings"},
        "queries": {"exchange": "queries", "routing_key": "queries"},
        "conversations": {"exchange": "conversations", "routing_key": "conversations"},
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Task configuration
    task_always_eager=False,  # Set to True for testing
    task_eager_propagates=True,
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    
    # Result backend configuration
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "process-pending-messages": {
            "task": "app.workers.message_processor.process_pending_messages",
            "schedule": crontab(minute="*/5"),  # Every 5 minutes
        },
        "process-completed-conversations": {
            "task": "app.workers.message_processor.process_completed_conversations",
            "schedule": crontab(minute="*/10"),  # Every 10 minutes - NEW conversation-level processing
            "kwargs": {"workspace_id": 1, "batch_size": 5}  # Process 5 conversations at a time
        },
        "extract-knowledge-batch": {
            "task": "app.workers.knowledge_extractor.extract_knowledge_batch",
            "schedule": crontab(minute="0", hour="*/2"),  # Every 2 hours
        },
        "generate-embeddings-batch": {
            "task": "app.workers.embedding_generator.generate_embeddings_batch",
            "schedule": crontab(minute="30", hour="*/4"),  # Every 4 hours
        },
        "process-pending-queries": {
            "task": "app.workers.query_processor.process_pending_queries",
            "schedule": crontab(minute="*/2"),  # Every 2 minutes
        },
        "daily-accuracy-tests": {
            "task": "app.workers.accuracy_tester.run_daily_accuracy_tests",
            "schedule": crontab(hour=2, minute=0),  # Every day at 2 AM
        },
        # Temporarily disabled - was flooding the worker queue
        # "backfill-conversations": {
        #     "task": "app.workers.conversation_backfill.backfill_all_channels_async",
        #     "schedule": crontab(minute=0, hour="*/6"),  # Every 6 hours
        #     "kwargs": {"days_back": 7}  # Keep last 7 days of conversations
        # },
    },
    
    # Logging
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
)

# Task annotations for better monitoring
celery_app.conf.task_annotations = {
    "app.workers.message_processor.process_message_async": {
        "rate_limit": "100/m",  # Max 100 tasks per minute
        "time_limit": 300,      # 5 minutes timeout
        "soft_time_limit": 240, # 4 minutes soft timeout
    },
    "app.workers.knowledge_extractor.extract_knowledge": {
        "rate_limit": "50/m",   # Max 50 tasks per minute
        "time_limit": 600,      # 10 minutes timeout
        "soft_time_limit": 480, # 8 minutes soft timeout
    },
    "app.workers.embedding_generator.generate_embedding": {
        "rate_limit": "20/m",   # Max 20 tasks per minute (API rate limits)
        "time_limit": 120,      # 2 minutes timeout
        "soft_time_limit": 90,  # 1.5 minutes soft timeout
    },
    "app.workers.query_processor.process_query_async": {
        "rate_limit": "30/m",   # Max 30 queries per minute
        "time_limit": 300,      # 5 minutes timeout
        "soft_time_limit": 240, # 4 minutes soft timeout
    },
    "app.workers.conversation_backfill.backfill_conversation_history_async": {
        "rate_limit": "10/m",   # Max 10 backfills per minute (Slack API limits)
        "time_limit": 1800,     # 30 minutes timeout
        "soft_time_limit": 1500, # 25 minutes soft timeout
    },
    "app.workers.conversation_backfill.backfill_all_channels_async": {
        "rate_limit": "1/h",    # Max 1 full backfill per hour
        "time_limit": 3600,     # 1 hour timeout
        "soft_time_limit": 3000, # 50 minutes soft timeout
    },
}

@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup."""
    logger.info(f"Request: {self.request!r}")
    return "Debug task completed successfully"

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks after Celery configuration."""
    logger.info("Setting up periodic tasks...")
    
    # Import realtime extractor tasks
    from . import realtime_extractor
    
    # Real-time knowledge extraction - every 15 minutes
    sender.add_periodic_task(
        crontab(minute="*/15"),  # Every 15 minutes
        realtime_extractor.extract_recent_knowledge.s(workspace_id=1, hours_back=2),
        name="extract-recent-knowledge"
    )
    
    # Today's discussions extraction - every 2 hours
    sender.add_periodic_task(
        crontab(minute="0", hour="*/2"),  # Every 2 hours
        realtime_extractor.extract_todays_discussions.s(workspace_id=1),
        name="extract-todays-discussions"
    )
    
    # Add any additional periodic task setup here
    sender.add_periodic_task(
        crontab(minute="0", hour="0"),  # Daily at midnight
        cleanup_old_tasks.s(),
        name="daily-cleanup"
    )

@celery_app.task
def cleanup_old_tasks():
    """Clean up old task results and logs."""
    logger.info("Running daily cleanup task...")
    # Implementation for cleanup logic
    return "Cleanup completed"

# Explicitly register all tasks to ensure they're available
try:
    from app.workers.query_processor import process_query_async, process_pending_queries
    from app.workers.message_processor import process_message_async, process_completed_conversations, analyze_conversation_state, process_pending_messages
    from app.workers.knowledge_extractor import extract_knowledge, extract_knowledge_async
    from app.workers.embedding_generator import generate_embedding, generate_embedding_async
    # from app.workers.conversation_backfill import backfill_conversation_history_async  # Circular import issue
    from app.workers.accuracy_tester import run_daily_accuracy_tests
    logger.info("Successfully registered all worker tasks")
except Exception as e:
    logger.error(f"Failed to register worker tasks: {e}")

if __name__ == "__main__":
    celery_app.start()
