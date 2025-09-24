#!/usr/bin/env python3
"""Manually trigger knowledge extraction for message 11."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.workers.knowledge_extractor import extract_knowledge

def trigger_extraction():
    """Trigger knowledge extraction for message 11."""
    print("🔄 Manually triggering knowledge extraction for message 11...")
    
    # Trigger the Celery task
    task_result = extract_knowledge.delay(
        message_id=11,
        text="We decided to use PostgreSQL for our database because it has better JSON support than MySQL. The team agreed this was the right choice for our use case.",
        message_type="conversation",
        significance_score=0.5
    )
    
    print(f"✅ Knowledge extraction task queued: {task_result.id}")
    print("⏳ Wait for processing to complete...")
    print("📊 Monitor logs with: docker-compose logs -f celery-worker")

if __name__ == "__main__":
    trigger_extraction()
