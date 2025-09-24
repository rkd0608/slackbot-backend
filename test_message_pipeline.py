#!/usr/bin/env python3
"""Test script to verify the message processing pipeline works."""

import sys
import os
from pathlib import Path
import asyncio
import json
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.database import AsyncSessionLocal
from app.models.base import Workspace, Message, User
from app.workers.message_processor import process_message_async
from sqlalchemy import select

async def test_message_pipeline():
    """Test the message processing pipeline with a manual message."""
    print("🧪 Testing message processing pipeline...")
    
    async_session = AsyncSessionLocal()
    async with async_session() as db:
        try:
            # Get the first workspace
            result = await db.execute(select(Workspace))
            workspace = result.scalar_one_or_none()
            
            if not workspace:
                print("❌ No workspace found in database")
                return False
            
            print(f"✅ Found workspace: {workspace.name}")
            
            # Create a test user
            test_user = User(
                workspace_id=workspace.id,
                slack_id="U1234567890",
                name="Test User",
                role="user"
            )
            db.add(test_user)
            await db.flush()
            print(f"✅ Created test user: {test_user.name}")
            
            # Create a test message
            test_message = Message(
                workspace_id=workspace.id,
                channel_id="C1234567890",
                user_id=test_user.id,
                raw_payload={
                    "text": "This is a test message to verify the knowledge extraction pipeline works correctly.",
                    "ts": "1234567890.123456",
                    "user": test_user.slack_id,
                    "channel": "C1234567890",
                    "team": workspace.slack_id
                }
            )
            db.add(test_message)
            await db.flush()
            print(f"✅ Created test message: {test_message.id}")
            
            await db.commit()
            
            # Now test the message processing
            print("🔄 Testing message processing...")
            
            # Trigger the Celery task
            task_result = process_message_async.delay(
                message_id=test_message.id,
                workspace_id=workspace.id,
                channel_id="C1234567890",
                user_id=test_user.slack_id,
                text="This is a test message to verify the knowledge extraction pipeline works correctly.",
                ts="1234567890.123456"
            )
            
            print(f"✅ Celery task queued: {task_result.id}")
            
            # Wait for the task to complete
            print("⏳ Waiting for task completion...")
            result = task_result.get(timeout=30)
            
            print(f"✅ Task completed successfully: {result}")
            return True
            
        except Exception as e:
            print(f"❌ Error testing message pipeline: {e}")
            await db.rollback()
            return False

if __name__ == "__main__":
    success = asyncio.run(test_message_pipeline())
    sys.exit(0 if success else 1)
