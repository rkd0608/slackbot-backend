#!/usr/bin/env python3
"""
Script to manually trigger conversation history backfill.
This is useful for initial setup and testing.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.workers.conversation_backfill import backfill_conversation_history, backfill_all_channels
from app.models.base import Workspace
from app.core.database import AsyncSessionLocal
from sqlalchemy import select

async def main():
    """Main function to run the backfill."""
    print("🚀 Starting conversation history backfill...")
    
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python scripts/backfill_conversations.py <workspace_id> [channel_id] [days_back]")
        print("  workspace_id: ID of the workspace to backfill")
        print("  channel_id: (optional) Specific channel ID to backfill")
        print("  days_back: (optional) Number of days to go back (default: 30)")
        sys.exit(1)
    
    workspace_id = int(sys.argv[1])
    channel_id = sys.argv[2] if len(sys.argv) > 2 else None
    days_back = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    print(f"📋 Backfill Configuration:")
    print(f"  Workspace ID: {workspace_id}")
    print(f"  Channel ID: {channel_id or 'ALL CHANNELS'}")
    print(f"  Days Back: {days_back}")
    print()
    
    async_session = AsyncSessionLocal()
    async with async_session() as db:
        try:
            # Verify workspace exists
            workspace = await db.execute(
                select(Workspace).where(Workspace.id == workspace_id)
            )
            workspace = workspace.scalar_one_or_none()
            
            if not workspace:
                print(f"❌ Workspace {workspace_id} not found!")
                sys.exit(1)
            
            print(f"✅ Found workspace: {workspace.name}")
            
            if not hasattr(workspace, 'tokens') or not workspace.tokens:
                print(f"❌ Workspace {workspace_id} missing Slack tokens!")
                sys.exit(1)
            
            print(f"✅ Workspace has Slack tokens")
            print()
            
            # Run the appropriate backfill
            if channel_id:
                print(f"🔄 Backfilling single channel: {channel_id}")
                result = await backfill_conversation_history(workspace_id, channel_id, days_back)
                
                if result.get('status') == 'success':
                    print(f"✅ Channel backfill completed successfully!")
                    print(f"  Processed: {result.get('processed_count', 0)} messages")
                    print(f"  Skipped: {result.get('skipped_count', 0)} messages")
                    print(f"  Total: {result.get('total_messages', 0)} messages")
                else:
                    print(f"❌ Channel backfill failed: {result.get('message', 'Unknown error')}")
                    sys.exit(1)
            else:
                print(f"🔄 Backfilling all channels...")
                result = await backfill_all_channels(workspace_id, days_back)
                
                if result.get('status') == 'completed':
                    print(f"✅ All channels backfill completed successfully!")
                    print(f"  Total channels: {result.get('total_channels', 0)}")
                    
                    # Show results for each channel
                    for channel_result in result.get('results', []):
                        channel_name = channel_result.get('channel_name', 'Unknown')
                        status = channel_result.get('result', {}).get('status', 'Unknown')
                        processed = channel_result.get('result', {}).get('processed_count', 0)
                        skipped = channel_result.get('result', {}).get('skipped_count', 0)
                        
                        if status == 'success':
                            print(f"  ✅ #{channel_name}: {processed} processed, {skipped} skipped")
                        else:
                            print(f"  ❌ #{channel_name}: {channel_result.get('result', {}).get('message', 'Failed')}")
                else:
                    print(f"❌ All channels backfill failed: {result.get('message', 'Unknown error')}")
                    sys.exit(1)
            
            print()
            print("🎉 Conversation history backfill completed!")
            
        except Exception as e:
            print(f"❌ Error during backfill: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
