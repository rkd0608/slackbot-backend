import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, text

from ..models.base import Workspace, Conversation, Message
from ..workers.conversation_backfill import backfill_conversation_history_async, backfill_all_channels_async

logger = logging.getLogger(__name__)

class BackfillService:
    """Service to manage automatic conversation history backfill."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def check_and_trigger_backfill(self, db: AsyncSession) -> Dict[str, Any]:
        """Check if backfill is needed and trigger it automatically."""
        try:
            self.logger.info("Checking if conversation backfill is needed...")
            
            # Get all workspaces with Slack tokens
            workspaces = await db.execute(
                select(Workspace).where(
                    and_(
                        Workspace.tokens.isnot(None),
                        Workspace.tokens != {}
                    )
                )
            )
            workspaces = workspaces.scalars().all()
            
            if not workspaces:
                self.logger.info("No workspaces with Slack tokens found")
                return {"status": "no_workspaces", "message": "No workspaces with Slack tokens"}
            
            self.logger.info(f"Found {len(workspaces)} workspaces with Slack tokens")
            
            results = []
            for workspace in workspaces:
                try:
                    workspace_result = await self._check_workspace_backfill_needed(workspace, db)
                    results.append(workspace_result)
                    
                    # If backfill is needed, trigger it
                    if workspace_result.get('backfill_needed'):
                        self.logger.info(f"Triggering backfill for workspace {workspace.id}")
                        await self._trigger_workspace_backfill(workspace.id)
                        
                except Exception as e:
                    self.logger.error(f"Error checking workspace {workspace.id}: {e}")
                    results.append({
                        'workspace_id': workspace.id,
                        'workspace_name': workspace.name,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                "status": "completed",
                "total_workspaces": len(workspaces),
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error in check_and_trigger_backfill: {e}", exc_info=True)
            raise
    
    async def _check_workspace_backfill_needed(self, workspace: Workspace, db: AsyncSession) -> Dict[str, Any]:
        """Check if a specific workspace needs backfill."""
        try:
            # Check if workspace has any conversations
            conversations_count = await db.execute(
                select(Conversation).where(Conversation.workspace_id == workspace.id)
            )
            conversations_count = len(conversations_count.scalars().all())
            
            # Check if workspace has any messages
            messages_count = await db.execute(
                select(Message).join(Conversation).where(Conversation.workspace_id == workspace.id)
            )
            messages_count = len(messages_count.scalars().all())
            
            # Check last message timestamp
            last_message_result = await db.execute(
                select(Message).join(Conversation).where(
                    Conversation.workspace_id == workspace.id
                ).order_by(Message.created_at.desc()).limit(1)
            )
            last_message = last_message_result.scalar_one_or_none()
            
            last_message_age = None
            if last_message:
                from datetime import timezone
                now = datetime.now(timezone.utc)
                last_message_age = (now - last_message.created_at).days
            
            # Determine if backfill is needed
            backfill_needed = False
            reason = []
            
            if conversations_count == 0:
                backfill_needed = True
                reason.append("No conversations exist")
            
            if messages_count == 0:
                backfill_needed = True
                reason.append("No messages exist")
            
            if last_message_age and last_message_age > 7:
                backfill_needed = True
                reason.append(f"Last message is {last_message_age} days old")
            
            return {
                'workspace_id': workspace.id,
                'workspace_name': workspace.name,
                'conversations_count': conversations_count,
                'messages_count': messages_count,
                'last_message_age_days': last_message_age,
                'backfill_needed': backfill_needed,
                'reason': reason,
                'status': 'checked'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking workspace {workspace.id}: {e}")
            raise
    
    async def _trigger_workspace_backfill(self, workspace_id: int) -> None:
        """Trigger backfill for a specific workspace."""
        try:
            # Queue the backfill task
            task_result = backfill_all_channels_async.delay(workspace_id, days_back=30)
            self.logger.info(f"Queued backfill task for workspace {workspace_id}: {task_result.id}")
            
        except Exception as e:
            self.logger.error(f"Error triggering backfill for workspace {workspace_id}: {e}")
            raise
    
    async def trigger_workspace_backfill(self, workspace_id: int, days_back: int = 30) -> Dict[str, Any]:
        """Manually trigger backfill for a specific workspace."""
        try:
            self.logger.info(f"Manually triggering backfill for workspace {workspace_id}")
            
            # Queue the backfill task
            task_result = backfill_all_channels_async.delay(workspace_id, days_back=days_back)
            
            return {
                "status": "queued",
                "workspace_id": workspace_id,
                "task_id": task_result.id,
                "days_back": days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error triggering workspace backfill: {e}")
            raise
    
    async def trigger_channel_backfill(self, workspace_id: int, channel_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Manually trigger backfill for a specific channel."""
        try:
            self.logger.info(f"Manually triggering backfill for channel {channel_id} in workspace {workspace_id}")
            
            # Queue the channel backfill task
            task_result = backfill_conversation_history_async.delay(workspace_id, channel_id, days_back)
            
            return {
                "status": "queued",
                "workspace_id": workspace_id,
                "channel_id": channel_id,
                "task_id": task_result.id,
                "days_back": days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error triggering channel backfill: {e}")
            raise
    
    async def get_backfill_status(self, db: AsyncSession) -> Dict[str, Any]:
        """Get the current status of conversation backfill across all workspaces."""
        try:
            # Get all workspaces
            workspaces = await db.execute(select(Workspace))
            workspaces = workspaces.scalars().all()
            
            status_summary = []
            total_conversations = 0
            total_messages = 0
            
            for workspace in workspaces:
                # Count conversations and messages for this workspace
                conversations = await db.execute(
                    select(Conversation).where(Conversation.workspace_id == workspace.id)
                )
                conversations = conversations.scalars().all()
                
                messages = await db.execute(
                    select(Message).join(Conversation).where(Conversation.workspace_id == workspace.id)
                )
                messages = messages.scalars().all()
                
                workspace_status = {
                    'workspace_id': workspace.id,
                    'workspace_name': workspace.name,
                    'slack_id': workspace.slack_id,
                    'has_tokens': bool(workspace.tokens),
                    'conversations_count': len(conversations),
                    'messages_count': len(messages),
                    'last_updated': workspace.updated_at.isoformat() if workspace.updated_at else None
                }
                
                status_summary.append(workspace_status)
                total_conversations += len(conversations)
                total_messages += len(messages)
            
            return {
                "status": "success",
                "total_workspaces": len(workspaces),
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "workspaces": status_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error getting backfill status: {e}")
            raise
