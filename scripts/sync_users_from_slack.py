#!/usr/bin/env python3
"""
Sync Users from Slack API Script

This script fetches real usernames from Slack API for all users in the database
that currently have fallback names (User_XXXX format), and updates them with 
their actual Slack display names or real names.

Usage:
    python scripts/sync_users_from_slack.py [workspace_id]

If no workspace_id is provided, it will sync users for workspace ID 1.
"""

import asyncio
import sys
import os
from datetime import datetime
from loguru import logger

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.database import get_session_factory
from app.services.user_service import UserService
from app.models.base import User, Workspace
from sqlalchemy import select


async def main():
    """Main function to sync users from Slack API."""
    logger.info("Starting Slack user synchronization process")
    
    # Get workspace ID from command line or default to 1
    workspace_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    logger.info(f"Target workspace ID: {workspace_id}")
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        user_service = UserService()
        
        # Check workspace exists and has bot token
        workspace_query = select(Workspace).where(Workspace.id == workspace_id)
        workspace_result = await db.execute(workspace_query)
        workspace = workspace_result.scalar_one_or_none()
        
        if not workspace:
            logger.error(f"Workspace {workspace_id} not found")
            return
        
        bot_token = None
        if workspace.tokens:
            bot_token = workspace.tokens.get('bot_token') or workspace.tokens.get('access_token')
        if not bot_token:
            logger.error(f"No Slack bot token found for workspace {workspace_id}")
            logger.info("   Make sure the workspace has a valid Slack bot token configured")
            logger.info(f"   Current tokens: {list(workspace.tokens.keys()) if workspace.tokens else 'None'}")
            return
        
        logger.info(f"Workspace found: {workspace.name}")
        
        # Get current users with fallback names
        users_query = select(User).where(
            User.workspace_id == workspace_id,
            User.name.like('User_%')
        )
        result = await db.execute(users_query)
        users_with_fallback_names = result.scalars().all()
        
        logger.info(f"Found {len(users_with_fallback_names)} users with fallback names")
        
        if not users_with_fallback_names:
            logger.info("No users need syncing - all users already have proper names!")
            return
        
        print("\nUsers to sync:")
        for user in users_with_fallback_names:
            print(f"  - {user.slack_id}: {user.name}")
        
        # Confirm before proceeding
        if len(sys.argv) <= 2 or sys.argv[-1] != '--force':
            confirm = input(f"\nSync {len(users_with_fallback_names)} users from Slack API? (y/N): ")
            if confirm.lower() != 'y':
                logger.info("Sync cancelled by user")
                return
        
        # Perform the sync
        logger.info("Starting user synchronization...")
        sync_result = await user_service.sync_all_users_from_slack(workspace_id, db)
        
        # Display results
        logger.info("User synchronization completed!")
        logger.info(f"Results:")
        logger.info(f"   Total users processed: {sync_result['total_users']}")
        logger.info(f"   Successfully synced: {sync_result['synced_successfully']}")
        logger.info(f"   Failed: {sync_result['failed']}")
        logger.info(f"   Success rate: {sync_result['success_rate']:.1%}")
        
        if sync_result.get('error'):
            logger.error(f"Error occurred: {sync_result['error']}")
        
        # Show updated users
        if sync_result['synced_successfully'] > 0:
            logger.info("\nUpdated users:")
            updated_users_query = select(User).where(
                User.workspace_id == workspace_id,
                ~User.name.like('User_%')  # Users that no longer have fallback names
            )
            updated_result = await db.execute(updated_users_query)
            updated_users = updated_result.scalars().all()
            
            for user in updated_users[-sync_result['synced_successfully']:]:
                logger.info(f"   {user.slack_id} -> {user.name}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
