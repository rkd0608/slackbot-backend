#!/usr/bin/env python3
"""Test script with a simple message that should pass verification."""

import sys
import os
from pathlib import Path
import asyncio
import json

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.database import AsyncSessionLocal
from app.models.base import Workspace, Message, User
from sqlalchemy import select

async def test_simple_knowledge():
    """Test with a simple, verifiable message."""
    print("🧪 Testing with a simple, verifiable message...")
    
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
            
            # Use the existing test user
            result = await db.execute(
                select(User).where(User.slack_id == "U9999999999")
            )
            test_user = result.scalar_one_or_none()
            
            if not test_user:
                print("❌ Test user not found")
                return False
            
            print(f"✅ Using test user: {test_user.name}")
            
            # Create a simple, verifiable message
            simple_text = """We decided to use PostgreSQL for our database because it has better JSON support than MySQL. The team agreed this was the right choice for our use case."""
            
            test_message = Message(
                workspace_id=workspace.id,
                channel_id="C041D76JMRA",
                user_id=test_user.id,
                raw_payload={
                    "text": simple_text,
                    "ts": "1234567890.123458",
                    "user": test_user.slack_id,
                    "channel": "C041D76JMRA",
                    "team": workspace.slack_id
                }
            )
            
            db.add(test_message)
            await db.flush()
            print(f"✅ Created simple message: {test_message.id}")
            print(f"📝 Message text: {simple_text}")
            
            await db.commit()
            
            # Trigger message processing
            print("🔄 Triggering message processing...")
            from app.workers.message_processor import process_message_async
            
            task_result = process_message_async.delay(
                message_id=test_message.id,
                workspace_id=workspace.id,
                channel_id="C041D76JMRA",
                user_id=test_user.slack_id,
                text=simple_text,
                ts="1234567890.123458"
            )
            
            print(f"✅ Message processing task queued: {task_result.id}")
            print("⏳ Wait for processing to complete...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing simple knowledge: {e}")
            await db.rollback()
            return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_knowledge())
    sys.exit(0 if success else 1)
