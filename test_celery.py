#!/usr/bin/env python3
"""Test script to verify Celery tasks are working after the event loop fix."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.workers.message_processor import process_pending_messages
from app.workers.celery_app import debug_task

def test_celery_tasks():
    """Test if Celery tasks can be executed without event loop errors."""
    print("🧪 Testing Celery tasks...")
    
    try:
        # Test the debug task first
        print("1. Testing debug task...")
        result = debug_task.delay()
        print(f"   Debug task result: {result.get(timeout=10)}")
        
        # Test the pending messages task
        print("2. Testing pending messages task...")
        result = process_pending_messages.delay()
        print(f"   Pending messages task result: {result.get(timeout=30)}")
        
        print("✅ All Celery tasks executed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Celery tasks: {e}")
        return False

if __name__ == "__main__":
    success = test_celery_tasks()
    sys.exit(0 if success else 1)
