"""Service for handling Slack interactive components and user feedback."""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update

from ..models.base import (
    Query, QueryFeedback, InteractionEvent, KnowledgeQuality, 
    User, Workspace, KnowledgeItem
)
from .slack_service import SlackService

class InteractionService:
    """Service for processing Slack interactive components."""
    
    def __init__(self):
        self.slack_service = SlackService()
        
        # Action ID mappings
        self.action_handlers = {
            "feedback_helpful": self._handle_helpful_feedback,
            "feedback_not_helpful": self._handle_not_helpful_feedback,
            "view_sources": self._handle_view_sources,
            "report_issue": self._handle_report_issue,
            "detailed_feedback": self._handle_detailed_feedback,
            "export_knowledge": self._handle_export_knowledge,
            "save_response": self._handle_save_response,
            "share_response": self._handle_share_response,
        }
    
    async def process_interaction(
        self, 
        payload: Dict[str, Any], 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process an interactive component interaction."""
        try:
            # Extract interaction details
            interaction_type = payload.get("type")
            user_data = payload.get("user", {})
            team_data = payload.get("team", {})
            channel_data = payload.get("channel", {})
            message_data = payload.get("message", {})
            
            slack_user_id = user_data.get("id")
            team_id = team_data.get("id")
            channel_id = channel_data.get("id")
            message_ts = message_data.get("ts")
            trigger_id = payload.get("trigger_id")
            
            # Get workspace and user
            workspace = await self._get_workspace_by_slack_id(team_id, db)
            if not workspace:
                return {"error": "Workspace not found"}
            
            user = await self._get_user_by_slack_id(slack_user_id, workspace.id, db)
            if not user:
                return {"error": "User not found"}
            
            # Handle different interaction types
            if interaction_type == "block_actions":
                return await self._handle_block_actions(payload, workspace, user, db)
            elif interaction_type == "view_submission":
                return await self._handle_modal_submission(payload, workspace, user, db)
            elif interaction_type == "view_closed":
                return await self._handle_modal_closed(payload, workspace, user, db)
            else:
                logger.warning(f"Unknown interaction type: {interaction_type}")
                return {"error": f"Unknown interaction type: {interaction_type}"}
                
        except Exception as e:
            logger.error(f"Error processing interaction: {e}", exc_info=True)
            return {"error": "Failed to process interaction"}
    
    async def _handle_block_actions(
        self, 
        payload: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Handle block action interactions (buttons, selects, etc.)."""
        actions = payload.get("actions", [])
        if not actions:
            return {"error": "No actions found"}
        
        action = actions[0]  # Handle first action
        action_id = action.get("action_id")
        value = action.get("value")
        
        # Log the interaction
        await self._log_interaction_event(
            user=user,
            workspace=workspace,
            payload=payload,
            interaction_type="block_action",
            component_type=action.get("type", "unknown"),
            action_id=action_id,
            db=db
        )
        
        # Route to appropriate handler
        handler = self.action_handlers.get(action_id)
        if handler:
            return await handler(payload, workspace, user, value, db)
        else:
            logger.warning(f"No handler found for action: {action_id}")
            return {"error": f"Unknown action: {action_id}"}
    
    async def _handle_helpful_feedback(
        self, 
        payload: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        value: str, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Handle positive feedback (👍) button click."""
        try:
            # Extract query ID from value
            query_id = int(value) if value and value.isdigit() else None
            if not query_id:
                return {"error": "Invalid query ID"}
            
            # Record feedback
            feedback = QueryFeedback(
                query_id=query_id,
                user_id=user.id,
                workspace_id=workspace.id,
                feedback_type="thumbs_up",
                is_helpful=True,
                rating=5,
                interaction_metadata={
                    "button_clicked": "helpful",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            db.add(feedback)
            await db.flush()
            
            # Update knowledge quality scores
            await self._update_knowledge_quality(query_id, True, db)
            
            await db.commit()
            
            # Update the message to show feedback was received
            return {
                "response_type": "in_channel",
                "replace_original": True,
                "text": self._get_updated_message_with_feedback(payload, "👍 Thanks for the feedback!"),
                "attachments": []
            }
            
        except Exception as e:
            logger.error(f"Error handling helpful feedback: {e}")
            return {"error": "Failed to record feedback"}
    
    async def _handle_not_helpful_feedback(
        self, 
        payload: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        value: str, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Handle negative feedback (👎) button click."""
        try:
            # Extract query ID from value
            query_id = int(value) if value and value.isdigit() else None
            if not query_id:
                return {"error": "Invalid query ID"}
            
            # Record feedback
            feedback = QueryFeedback(
                query_id=query_id,
                user_id=user.id,
                workspace_id=workspace.id,
                feedback_type="thumbs_down",
                is_helpful=False,
                rating=1,
                interaction_metadata={
                    "button_clicked": "not_helpful",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            db.add(feedback)
            await db.flush()
            
            # Update knowledge quality scores
            await self._update_knowledge_quality(query_id, False, db)
            
            await db.commit()
            
            # Show modal for detailed feedback
            modal = self._create_detailed_feedback_modal(query_id)
            
            # Open modal
            trigger_id = payload.get("trigger_id")
            if trigger_id:
                await self.slack_service.open_modal(
                    trigger_id=trigger_id,
                    view=modal,
                    token=workspace.tokens.get("access_token")
                )
            
            return {
                "response_type": "in_channel",
                "replace_original": True,
                "text": self._get_updated_message_with_feedback(payload, "👎 Thanks for the feedback! Please provide more details."),
                "attachments": []
            }
            
        except Exception as e:
            logger.error(f"Error handling not helpful feedback: {e}")
            return {"error": "Failed to record feedback"}
    
    async def _handle_view_sources(
        self, 
        payload: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        value: str, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Handle view sources button click."""
        try:
            # Extract query ID from value
            query_id = int(value) if value and value.isdigit() else None
            if not query_id:
                return {"error": "Invalid query ID"}
            
            # Get the query and its knowledge sources
            query_result = await db.execute(
                select(Query).where(Query.id == query_id)
            )
            query = query_result.scalar_one_or_none()
            
            if not query:
                return {"error": "Query not found"}
            
            # Create sources modal
            sources_modal = await self._create_sources_modal(query, db)
            
            # Open modal
            trigger_id = payload.get("trigger_id")
            if trigger_id:
                await self.slack_service.open_modal(
                    trigger_id=trigger_id,
                    view=sources_modal,
                    token=workspace.tokens.get("access_token")
                )
            
            return {"response_type": "ephemeral", "text": "Opening sources..."}
            
        except Exception as e:
            logger.error(f"Error handling view sources: {e}")
            return {"error": "Failed to show sources"}
    
    async def _handle_report_issue(
        self, 
        payload: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        value: str, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Handle report issue button click."""
        try:
            query_id = int(value) if value and value.isdigit() else None
            
            # Create issue report modal
            modal = self._create_issue_report_modal(query_id)
            
            # Open modal
            trigger_id = payload.get("trigger_id")
            if trigger_id:
                await self.slack_service.open_modal(
                    trigger_id=trigger_id,
                    view=modal,
                    token=workspace.tokens.get("access_token")
                )
            
            return {"response_type": "ephemeral", "text": "Opening issue report..."}
            
        except Exception as e:
            logger.error(f"Error handling report issue: {e}")
            return {"error": "Failed to open issue report"}
    
    async def _handle_modal_submission(
        self, 
        payload: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Handle modal form submissions."""
        try:
            view = payload.get("view", {})
            callback_id = view.get("callback_id")
            values = view.get("state", {}).get("values", {})
            
            if callback_id == "detailed_feedback_modal":
                return await self._process_detailed_feedback_submission(values, workspace, user, db)
            elif callback_id == "issue_report_modal":
                return await self._process_issue_report_submission(values, workspace, user, db)
            else:
                return {"error": f"Unknown modal callback: {callback_id}"}
                
        except Exception as e:
            logger.error(f"Error handling modal submission: {e}")
            return {"error": "Failed to process modal submission"}
    
    async def _handle_modal_closed(
        self, 
        payload: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Handle modal closed events."""
        # Log the event but don't need to return anything special
        await self._log_interaction_event(
            user=user,
            workspace=workspace,
            payload=payload,
            interaction_type="modal_closed",
            component_type="modal",
            action_id="modal_closed",
            db=db
        )
        return {}
    
    async def _update_knowledge_quality(
        self, 
        query_id: int, 
        is_positive: bool, 
        db: AsyncSession
    ):
        """Update knowledge quality scores based on feedback."""
        try:
            # Get the query and its response data
            query_result = await db.execute(
                select(Query).where(Query.id == query_id)
            )
            query = query_result.scalar_one_or_none()
            
            if not query or not query.response:
                return
            
            # Extract knowledge item IDs from the response
            # This would depend on how you store the knowledge items used in the response
            # For now, we'll implement a basic version
            
            # Update quality scores for knowledge items used in this query
            # This is a simplified implementation - you'd want to track which specific
            # knowledge items were used in each response
            
            logger.info(f"Updated knowledge quality based on feedback for query {query_id}")
            
        except Exception as e:
            logger.error(f"Error updating knowledge quality: {e}")
    
    def _create_detailed_feedback_modal(self, query_id: int) -> Dict[str, Any]:
        """Create a modal for collecting detailed feedback."""
        return {
            "type": "modal",
            "callback_id": "detailed_feedback_modal",
            "title": {
                "type": "plain_text",
                "text": "Provide Feedback"
            },
            "submit": {
                "type": "plain_text",
                "text": "Submit"
            },
            "close": {
                "type": "plain_text",
                "text": "Cancel"
            },
            "private_metadata": json.dumps({"query_id": query_id}),
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Help us improve by providing more details about the issue:"
                    }
                },
                {
                    "type": "input",
                    "block_id": "issue_category",
                    "element": {
                        "type": "radio_buttons",
                        "action_id": "category_select",
                        "options": [
                            {
                                "text": {"type": "plain_text", "text": "Inaccurate information"},
                                "value": "inaccurate"
                            },
                            {
                                "text": {"type": "plain_text", "text": "Incomplete answer"},
                                "value": "incomplete"
                            },
                            {
                                "text": {"type": "plain_text", "text": "Irrelevant response"},
                                "value": "irrelevant"
                            },
                            {
                                "text": {"type": "plain_text", "text": "Other issue"},
                                "value": "other"
                            }
                        ]
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "What was the main issue?"
                    }
                },
                {
                    "type": "input",
                    "block_id": "feedback_details",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "details_input",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Please provide more details about the issue..."
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Additional details"
                    },
                    "optional": True
                },
                {
                    "type": "input",
                    "block_id": "suggested_correction",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "correction_input",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "If you know the correct answer, please share it here..."
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Suggested correction"
                    },
                    "optional": True
                }
            ]
        }
    
    def _create_issue_report_modal(self, query_id: Optional[int]) -> Dict[str, Any]:
        """Create a modal for reporting issues."""
        return {
            "type": "modal",
            "callback_id": "issue_report_modal",
            "title": {
                "type": "plain_text",
                "text": "Report Issue"
            },
            "submit": {
                "type": "plain_text",
                "text": "Report"
            },
            "close": {
                "type": "plain_text",
                "text": "Cancel"
            },
            "private_metadata": json.dumps({"query_id": query_id}),
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Please describe the issue you encountered:"
                    }
                },
                {
                    "type": "input",
                    "block_id": "issue_description",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "description_input",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Describe the issue in detail..."
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Issue description"
                    }
                },
                {
                    "type": "input",
                    "block_id": "issue_priority",
                    "element": {
                        "type": "radio_buttons",
                        "action_id": "priority_select",
                        "options": [
                            {
                                "text": {"type": "plain_text", "text": "Low - Minor issue"},
                                "value": "low"
                            },
                            {
                                "text": {"type": "plain_text", "text": "Medium - Affects usability"},
                                "value": "medium"
                            },
                            {
                                "text": {"type": "plain_text", "text": "High - Urgent issue"},
                                "value": "high"
                            }
                        ]
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Priority level"
                    }
                }
            ]
        }
    
    async def _create_sources_modal(self, query: Query, db: AsyncSession) -> Dict[str, Any]:
        """Create a modal showing the sources used in the response."""
        # This would extract the actual sources from the query response
        # For now, we'll create a placeholder implementation
        
        return {
            "type": "modal",
            "callback_id": "sources_modal",
            "title": {
                "type": "plain_text",
                "text": "Knowledge Sources"
            },
            "close": {
                "type": "plain_text",
                "text": "Close"
            },
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Sources for:* {query.text}\n\nThe following conversations and documents were used to generate this response:"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "📝 *Conversation in #general*\n<https://slack.com/link|View original message>\n_\"We decided to migrate to PostgreSQL because...\"_"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "📋 *Documentation*\n<https://docs.example.com|Migration Guide>\n_\"Step-by-step process for database migration\"_"
                    }
                }
            ]
        }
    
    def _get_updated_message_with_feedback(self, payload: Dict[str, Any], feedback_text: str) -> str:
        """Update the original message to show feedback was received."""
        original_message = payload.get("message", {}).get("text", "")
        return f"{original_message}\n\n{feedback_text}"
    
    async def _process_detailed_feedback_submission(
        self, 
        values: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process detailed feedback form submission."""
        try:
            # Extract form values
            category_block = values.get("issue_category", {})
            details_block = values.get("feedback_details", {})
            correction_block = values.get("suggested_correction", {})
            
            category = category_block.get("category_select", {}).get("selected_option", {}).get("value")
            details = details_block.get("details_input", {}).get("value", "")
            correction = correction_block.get("correction_input", {}).get("value", "")
            
            # Here you would save the detailed feedback to the database
            logger.info(f"Received detailed feedback: category={category}, details={details}")
            
            return {"response_action": "clear"}
            
        except Exception as e:
            logger.error(f"Error processing detailed feedback: {e}")
            return {"response_action": "errors", "errors": {"general": "Failed to process feedback"}}
    
    async def _process_issue_report_submission(
        self, 
        values: Dict[str, Any], 
        workspace: Workspace, 
        user: User, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process issue report form submission."""
        try:
            # Extract form values
            description_block = values.get("issue_description", {})
            priority_block = values.get("issue_priority", {})
            
            description = description_block.get("description_input", {}).get("value", "")
            priority = priority_block.get("priority_select", {}).get("selected_option", {}).get("value")
            
            # Here you would save the issue report
            logger.info(f"Received issue report: priority={priority}, description={description}")
            
            return {"response_action": "clear"}
            
        except Exception as e:
            logger.error(f"Error processing issue report: {e}")
            return {"response_action": "errors", "errors": {"general": "Failed to process issue report"}}
    
    async def _log_interaction_event(
        self, 
        user: User, 
        workspace: Workspace, 
        payload: Dict[str, Any], 
        interaction_type: str, 
        component_type: str, 
        action_id: str, 
        db: AsyncSession
    ):
        """Log an interaction event for analytics."""
        try:
            event = InteractionEvent(
                user_id=user.id,
                workspace_id=workspace.id,
                interaction_type=interaction_type,
                component_type=component_type,
                action_id=action_id,
                slack_user_id=user.slack_id,
                slack_channel_id=payload.get("channel", {}).get("id", ""),
                slack_message_ts=payload.get("message", {}).get("ts"),
                slack_trigger_id=payload.get("trigger_id"),
                payload=payload
            )
            
            db.add(event)
            await db.flush()
            
        except Exception as e:
            logger.error(f"Error logging interaction event: {e}")
    
    async def _get_workspace_by_slack_id(self, slack_id: str, db: AsyncSession) -> Optional[Workspace]:
        """Get workspace by Slack team ID."""
        result = await db.execute(
            select(Workspace).where(Workspace.slack_id == slack_id)
        )
        return result.scalar_one_or_none()
    
    async def _get_user_by_slack_id(self, slack_user_id: str, workspace_id: int, db: AsyncSession) -> Optional[User]:
        """Get user by Slack user ID and workspace."""
        result = await db.execute(
            select(User).where(
                and_(
                    User.slack_id == slack_user_id,
                    User.workspace_id == workspace_id
                )
            )
        )
        return result.scalar_one_or_none()
