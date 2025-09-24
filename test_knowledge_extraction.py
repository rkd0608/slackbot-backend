#!/usr/bin/env python3
"""Test script to test the fixed knowledge extraction system."""

import sys
import os
from pathlib import Path
import asyncio
import json

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.database import AsyncSessionLocal
from app.models.base import Workspace, Message, Userq
from sqlalchemy import select

async def test_knowledge_extraction():
    """Test the knowledge extraction system with a high-significance message."""
    print("🧪 Testing the fixed knowledge extraction system...")
    
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
            
            # Create a test user if it doesn't exist
            result = await db.execute(
                select(User).where(User.slack_id == "U9999999999")
            )
            test_user = result.scalar_one_or_none()
            
            if not test_user:
                test_user = User(
                    workspace_id=workspace.id,
                    slack_id="U9999999999",
                    name="Test User 2",
                    role="user"
                )
                db.add(test_user)
                await db.flush()
                print(f"✅ Created test user: {test_user.name}")
            else:
                print(f"✅ Using existing test user: {test_user.name}")
            
            # Create a high-significance message (should score > 0.6)
            high_significance_text = """We need to implement a comprehensive disaster recovery plan for our production systems. This should include automated backups, failover procedures, and regular testing. What are the key components we should focus on first? Also, how do we ensure minimal downtime during maintenance windows?"""
            
            test_message = Message(
                workspace_id=workspace.id,
                channel_id="C041D76JMRA",  # Your real channel
                user_id=test_user.id,
                raw_payload={
                    "text": high_significance_text,
                    "ts": "1234567890.123457",
                    "user": test_user.slack_id,
                    "channel": "C041D76JMRA",
                    "team": workspace.slack_id
                }
            )
            
            db.add(test_message)
            await db.flush()
            print(f"✅ Created high-significance message: {test_message.id}")
            print(f"📝 Message text: {high_significance_text[:100]}...")
            
            await db.commit()
            
            # Now trigger message processing
            print("🔄 Triggering message processing...")
            from app.workers.message_processor import process_message_async
            
            task_result = process_message_async.delay(
                message_id=test_message.id,
                workspace_id=workspace.id,
                channel_id="C041D76JMRA",
                user_id=test_user.slack_id,
                text=high_significance_text,
                ts="1234567890.123457"
            )
            
            print(f"✅ Message processing task queued: {task_result.id}")
            print("⏳ Wait for processing to complete, then check if knowledge extraction is triggered...")
            print("📊 Monitor logs with: docker-compose logs -f celery-worker")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing knowledge extraction: {e}")
            await db.rollback()
            return False

if __name__ == "__main__":
    success = asyncio.run(test_knowledge_extraction())
    sys.exit(0 if success else 1)
