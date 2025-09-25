"""Slack Events API endpoint for handling webhooks and message ingestion."""

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import json
import hmac
import hashlib
import time
from typing import Dict, Any, Optional
from loguru import logger
from sqlalchemy import func
from datetime import datetime, timedelta

from ...core.database import get_db
from ...core.config import settings
from ...models.base import Workspace, Message, User
from ...services.slack_service import SlackService
from ...services.backfill_service import BackfillService
from ...services.interaction_service import InteractionService

# Import removed to avoid circular dependency - will be imported when needed

router = APIRouter()


def verify_slack_signature(request: Request, body: bytes) -> bool:
    """Verify Slack request signature for security."""
    try:
        timestamp = request.headers.get("x-slack-request-timestamp", "")
        signature = request.headers.get("x-slack-signature", "")

        if not timestamp or not signature:
            return False

        # Check if request is too old (replay attack protection)
        if abs(time.time() - int(timestamp)) > 60 * 5:  # 5 minutes
            return False

        # Create expected signature
        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected_signature = f"v0={hmac.new(settings.slack_signing_secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()}"

        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error("Error verifying Slack signature: {}", str(e))
        return False


async def get_workspace_by_slack_id(slack_id: str, db: AsyncSession) -> Optional[Workspace]:
    """Get workspace by Slack ID."""
    result = await db.execute(
        select(Workspace).where(Workspace.slack_id == slack_id)
    )
    return result.scalar_one_or_none()


async def get_or_create_user(slack_user_id: str, workspace_id: int, db: AsyncSession) -> User:
    """Get or create user in the database."""
    # First try to find existing user
    result = await db.execute(
        select(User).where(
            and_(
                User.slack_id == slack_user_id,
                User.workspace_id == workspace_id
            )
        )
    )
    user = result.scalar_one_or_none()

    if user:
        return user

    # Create new user if not found
    # Note: We'll need to fetch user info from Slack API later
    new_user = User(
        workspace_id=workspace_id,
        slack_id=slack_user_id,
        name=f"User_{slack_user_id}",  # Placeholder name
        role="user"
    )
    db.add(new_user)
    await db.flush()  # This ensures the user gets an ID
    await db.refresh(new_user)  # This ensures we have the full user object
    return new_user


@router.post("/events")
async def handle_slack_events(
        request: Request,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
):
    """Handle Slack Events API webhooks."""
    try:
        # Read request body
        body = await request.body()

        # Verify Slack signature
        if not verify_slack_signature(request, body):
            logger.warning("Invalid Slack signature received")
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse event data
        event_data = json.loads(body.decode('utf-8'))
        logger.info(f"Received Slack event: {event_data.get('type', 'unknown')}")

        # Handle URL verification challenge
        if event_data.get("type") == "url_verification":
            return JSONResponse(content={"challenge": event_data.get("challenge")})

        # Handle events
        event = event_data.get("event", {})
        event_type = event.get("type")

        if event_type == "message":
            await handle_message_event(event, event_data, db, background_tasks)
        elif event_type == "app_mention":
            await handle_app_mention_event(event, event_data, db, background_tasks)
        elif event_type == "team_join":
            await handle_team_join_event(event, event_data, db)
        elif event_type == "app_home_opened" or event_type == "app_installed":
            # Trigger automatic backfill for new workspace
            try:
                backfill_service = BackfillService()
                team_id = event_data.get("team_id")
                if team_id:
                    workspace = await get_workspace_by_slack_id(team_id, db)
                    if workspace:
                        logger.info(f"🔄 Triggering automatic backfill for new workspace {workspace.id}")
                        await backfill_service.trigger_workspace_backfill(workspace.id, days_back=30)
            except Exception as e:
                logger.warning(f"Failed to trigger automatic backfill: {e}")
                # Don't fail the event for backfill errors
        elif event_type == "channel_created":
            # Trigger automatic backfill for new channel
            try:
                backfill_service = BackfillService()
                team_id = event_data.get("team_id")
                channel_id = event_data.get("channel", {}).get("id")
                if team_id and channel_id:
                    workspace = await get_workspace_by_slack_id(team_id, db)
                    if workspace:
                        logger.info(
                            f"🔄 Triggering automatic backfill for new channel {channel_id} in workspace {workspace.id}")
                        await backfill_service.trigger_channel_backfill(workspace.id, channel_id, days_back=30)
            except Exception as e:
                logger.warning(f"Failed to trigger channel backfill: {e}")
                # Don't fail the event for backfill errors
        else:
            logger.info(f"Unhandled event type: {event_type}")

        return JSONResponse(content={"status": "ok"})

    except Exception as e:
        logger.error("Error handling Slack event: {}", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def handle_message_event(
        event: Dict[str, Any],
        event_data: Dict[str, Any],
        db: AsyncSession,
        background_tasks: BackgroundTasks
):
    """Handle message events from Slack."""
    try:
        # Extract event data
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        team_id = event_data.get("team_id")

        # Skip bot messages and messages without text
        if event.get("bot_id") or not text or not user_id:
            return

        # Get workspace
        workspace = await get_workspace_by_slack_id(team_id, db)
        if not workspace:
            logger.warning(f"Workspace not found for team: {team_id}")
            return

        # Get or create user
        user = await get_or_create_user(user_id, workspace.id, db)

        # Get or create conversation
        from ...models.base import Conversation
        conversation_result = await db.execute(
            select(Conversation).where(
                and_(
                    Conversation.workspace_id == workspace.id,
                    Conversation.slack_channel_id == channel_id
                )
            )
        )
        conversation = conversation_result.scalar_one_or_none()

        if not conversation:
            # Create new conversation
            conversation = Conversation(
                workspace_id=workspace.id,
                slack_channel_id=channel_id,
                slack_channel_name=f"channel-{channel_id}",  # Use fallback name
                title=f"#{channel_id}",
                conversation_metadata={}
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            logger.info(f"Created new conversation {conversation.id} for channel {channel_id}")

        # Idempotency: avoid duplicate inserts for the same Slack message
        message_ts = event.get('ts', '')
        existing_msg_result = await db.execute(
            select(Message).where(
                and_(
                    Message.conversation_id == conversation.id,
                    Message.slack_message_id == message_ts
                )
            )
        )
        existing_message = existing_msg_result.scalar_one_or_none()

        if existing_message:
            message_id_var = existing_message.id
            logger.info(
                f"Message already stored for ts {message_ts} in conversation {conversation.id}, skipping insert")
        else:
            # Store message with new model structure
            message = Message(
                conversation_id=conversation.id,
                slack_message_id=message_ts,
                slack_user_id=user_id,
                content=event.get('text', ''),
                message_metadata={
                    'raw_payload': event,
                    'slack_ts': event.get('ts'),
                    'slack_user_id': user_id,
                    'slack_thread_ts': event.get('thread_ts'),
                    'slack_reply_count': event.get('reply_count', 0),
                    'slack_reply_users_count': event.get('reply_users_count', 0),
                    'slack_type': event.get('type'),
                    'slack_subtype': event.get('subtype')
                }
            )
            db.add(message)
            await db.commit()
            await db.refresh(message)
            message_id_var = message.id
            logger.info(f"Stored message {message.id} from user {user_id} in channel {channel_id}")

        # Add background task for message processing using simple system
        # Import here to avoid circular dependency
        from ...workers.simple_message_processor import process_message_async

        # Process the individual message (for immediate needs)
        background_tasks.add_task(
            process_message_async,
            message_id=message_id_var,
            workspace_id=workspace.id,
            channel_id=channel_id,
            user_id=user_id,
            text=text,
            timestamp=ts,
            thread_ts=thread_ts,
            raw_payload=event
        )

    except Exception as e:
        logger.error("Error handling message event: {}", str(e), exc_info=True)


async def handle_app_mention_event(
        event: Dict[str, Any],
        event_data: Dict[str, Any],
        db: AsyncSession,
        background_tasks: BackgroundTasks
):
    """Handle app mention events (when @reno is mentioned)."""
    try:
        # Extract event data
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        team_id = event_data.get("team_id")

        logger.info(f"Bot mentioned in channel {channel_id} by user {user_id}: {text}")

        # Skip if no text or user
        if not text or not user_id:
            return

        # Get workspace and user
        workspace = await get_workspace_by_slack_id(team_id, db)
        if not workspace:
            logger.warning(f"Workspace not found for team: {team_id}")
            return

        user = await get_or_create_user(user_id, workspace.id, db)
        if not user:
            logger.warning(f"Could not create/find user: {user_id}")
            return

        # CRITICAL: Commit the user creation before calling Celery task
        await db.commit()

        # Extract the query text by removing the bot mention
        import re
        # Remove bot mention (e.g., "<@U1234567890>" or "@reno")
        query_text = re.sub(r'<@[^>]+>', '', text).strip()
        query_text = re.sub(r'@\w+', '', query_text).strip()

        if not query_text:
            # Send a helpful message if no query is provided
            slack_service = SlackService()
            if workspace.tokens.get('access_token'):
                await slack_service.send_message(
                    channel=channel_id,
                    text="👋 Hi! I'm Reno, your team knowledge assistant. Ask me anything about your team's conversations!\n\nExample: `@reno How do I restart the Kafka connector?`",
                    token=workspace.tokens.get('access_token')
                )
            return

        logger.info(f"Extracted query from mention: '{query_text}'")

        # Rate limiting check
        if not await check_rate_limit(user_id, workspace.id, db):
            slack_service = SlackService()
            if workspace.tokens.get('access_token'):
                await slack_service.send_message(
                    channel=channel_id,
                    text="⏳ Slow down there! You're asking questions too quickly. Please wait a moment before asking again.",
                    token=workspace.tokens.get('access_token')
                )
            return

        # Process the query asynchronously (same as /ask command)
        from ...workers.simplified_celery_app import celery_app

        task_result = celery_app.send_task(
            'app.workers.simple_query_processor.process_query_async',
            args=[
                None,  # query_id - will be generated
                workspace.id,  # workspace_id
                user.id,  # user_id
                channel_id,  # channel_id
                query_text,  # query_text (cleaned)
                None,  # response_url (not available for mentions)
                False  # is_slash_command (this is a mention)
            ]
        )

        logger.info(f"Queued query processing task for mention: {task_result.id}")

        # Send immediate acknowledgment
        slack_service = SlackService()
        if workspace.tokens.get('access_token'):
            await slack_service.send_message(
                channel=channel_id,
                text=f"🤔 Let me search for information about: *{query_text}*\n\nSearching through our team knowledge...",
                token=workspace.tokens.get('access_token')
            )

        # Also store the message for knowledge extraction (same as regular messages)
        await handle_message_event(event, event_data, db, background_tasks)

    except Exception as e:
        logger.error("Error handling app mention event: {}", str(e), exc_info=True)


async def handle_team_join_event(
        event: Dict[str, Any],
        event_data: Dict[str, Any],
        db: AsyncSession
):
    """Handle team join events (new users joining workspace)."""
    try:
        team_id = event_data.get("team_id")
        user_id = event.get("user", {}).get("id")

        if not team_id or not user_id:
            return

        # Get workspace
        workspace = await get_workspace_by_slack_id(team_id, db)
        if not workspace:
            logger.warning(f"Workspace not found for team: {team_id}")
            return

        # Create new user
        new_user = User(
            workspace_id=workspace.id,
            slack_id=user_id,
            name=event.get("user", {}).get("name", f"User_{user_id}"),
            role="user"
        )

        db.add(new_user)
        await db.commit()

        logger.info(f"Created new user {new_user.id} for {user_id} in workspace {workspace.id}")

    except Exception as e:
        logger.error("Error handling team join event: {}", str(e), exc_info=True)


@router.post("/commands/ask")
async def handle_ask_command(
        request: Request,
        db: AsyncSession = Depends(get_db)
):
    """Handle the /ask slash command from Slack."""
    try:
        # Parse the Slack command payload
        form_data = await request.form()
        command_data = dict(form_data)

        logger.info(f"Received /ask command: {command_data}")

        # Extract command data
        user_id = command_data.get("user_id")
        channel_id = command_data.get("channel_id")
        team_id = command_data.get("team_id")
        text = command_data.get("text", "").strip()
        response_url = command_data.get("response_url")

        if not all([user_id, channel_id, team_id, text]):
            return {
                "response_type": "ephemeral",
                "text": "❌ Missing required command parameters. Please try again."
            }

        # Get workspace and user
        workspace = await get_workspace_by_slack_id(team_id, db)
        if not workspace:
            return {
                "response_type": "ephemeral",
                "text": "❌ Workspace not found. Please contact your administrator."
            }

        user = await get_or_create_user(user_id, workspace.id, db)
        if not user:
            return {
                "response_type": "ephemeral",
                "text": "❌ User not found. Please contact your administrator."
            }

        # CRITICAL: Commit the user creation before calling Celery task
        await db.commit()

        # Rate limiting check
        if not await check_rate_limit(user_id, workspace.id, db):
            return {
                "response_type": "ephemeral",
                "text": "⏳ Rate limit exceeded. Please wait a moment before asking another question."
            }

        # Process the query asynchronously
        from ...workers.simplified_celery_app import celery_app

        task_result = celery_app.send_task(
            'app.workers.simple_query_processor.process_query_async',
            args=[
                None,  # query_id - will be generated
                workspace.id,  # workspace_id
                user.id,  # user_id
                channel_id,  # channel_id
                text,  # query_text
                response_url,  # response_url
                True  # is_slash_command
            ]
        )

        logger.info(f"Queued query processing task: {task_result.id}")

        # Return immediate acknowledgment
        return {
            "response_type": "in_channel",
            "text": f"🤔 Processing your question: *{text}*\n\nI'm searching through our knowledge base...",
            "attachments": [
                {
                    "text": "This may take a few seconds. I'll respond with the best answers I can find.",
                    "color": "#36a64f"
                }
            ]
        }

    except Exception as e:
        logger.error("Error handling /ask command: {}", str(e), exc_info=True)
        return {
            "response_type": "ephemeral",
            "text": "❌ Sorry, I encountered an error processing your request. Please try again in a moment."
        }


@router.post("/interactive")
async def handle_slack_interactive(
        request: Request,
        db: AsyncSession = Depends(get_db)
):
    """Handle Slack interactive components (buttons, modals, etc.)."""
    try:
        # Read request body
        body = await request.body()

        # Verify Slack signature
        if not verify_slack_signature(request, body):
            logger.warning("Invalid Slack signature received for interactive component")
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse the payload (Slack sends form-encoded data for interactive components)
        form_data = await request.form()
        payload_str = form_data.get("payload")

        if not payload_str:
            raise HTTPException(status_code=400, detail="No payload found")

        payload = json.loads(payload_str)
        logger.info(f"Received interactive component: {payload.get('type', 'unknown')}")

        # Process the interaction
        interaction_service = InteractionService()
        result = await interaction_service.process_interaction(payload, db)

        if "error" in result:
            logger.error(f"Interaction processing error: {result['error']}")
            return {
                "response_type": "ephemeral",
                "text": f"❌ {result['error']}"
            }

        # Return the result (could be a message update, modal, etc.)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse interactive payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload format")

    except Exception as e:
        logger.error(f"Error handling interactive component: {e}", exc_info=True)
        return {
            "response_type": "ephemeral",
            "text": "❌ Sorry, I encountered an error processing your interaction. Please try again."
        }


async def check_rate_limit(slack_user_id: str, workspace_id: int, db: AsyncSession) -> bool:
    """Check if user has exceeded rate limits for queries."""
    try:
        # First, get the User record to get the integer user_id
        from ...models.base import Query, User

        user_result = await db.execute(
            select(User).where(
                and_(
                    User.slack_id == slack_user_id,
                    User.workspace_id == workspace_id
                )
            )
        )
        user = user_result.scalar_one_or_none()

        if not user:
            logger.warning(f"User not found for rate limiting: {slack_user_id}")
            return True  # Allow if user not found

        # Check queries in the last minute
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)

        result = await db.execute(
            select(func.count(Query.id))
            .where(
                and_(
                    Query.user_id == user.id,  # Use integer user_id
                    Query.workspace_id == workspace_id,
                    Query.created_at >= one_minute_ago
                )
            )
        )

        recent_query_count = result.scalar() or 0

        # Allow max 10 queries per minute per user (increased for testing)
        max_queries_per_minute = 10
        return recent_query_count < max_queries_per_minute

    except Exception as e:
        logger.error("Error checking rate limit: {}", str(e))
        # Default to allowing the request if rate limiting fails
        return True


@router.get("/channels")
async def list_channels(db: AsyncSession = Depends(get_db)):
    """List all channels where the bot is active."""
    try:
        # Get unique channels from messages
        result = await db.execute(
            select(Message.channel_id, Message.workspace_id)
            .distinct()
        )
        channels = result.fetchall()

        channel_list = []
        for channel_id, workspace_id in channels:
            # Get workspace info
            workspace_result = await db.execute(
                select(Workspace.name, Workspace.slack_id)
                .where(Workspace.id == workspace_id)
            )
            workspace_data = workspace_result.fetchone()

            if workspace_data:
                workspace_name, workspace_slack_id = workspace_data
                channel_list.append({
                    "channel_id": channel_id,
                    "workspace_name": workspace_name,
                    "workspace_slack_id": workspace_slack_id
                })

        return {"channels": channel_list}

    except Exception as e:
        logger.error("Error listing channels: {}", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/test")
async def test_slack_integration():
    """Test endpoint to verify Slack integration setup."""
    try:
        from ...core.config import settings

        return {
            "status": "ok",
            "slack_config": {
                "client_id": settings.slack_client_id,
                "bot_token_set": bool(settings.slack_bot_token),
                "signing_secret_set": bool(settings.slack_signing_secret),
                "app_base_url": settings.app_base_url,
                "events_url": f"{settings.app_base_url}/api/v1/slack/events"
            },
            "message": "Slack integration test endpoint. Check your Slack app configuration."
        }
    except Exception as e:
        logger.error("Error in test endpoint: {}", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/messages/{channel_id}")
async def get_channel_messages(
        channel_id: str,
        limit: int = 50,
        offset: int = 0,
        db: AsyncSession = Depends(get_db)
):
    """Get messages from a specific channel."""
    try:
        result = await db.execute(
            select(Message, User.name, Workspace.name)
            .join(User, Message.user_id == User.id)
            .join(Workspace, Message.workspace_id == Workspace.id)
            .where(Message.channel_id == channel_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        messages = []
        for message, user_name, workspace_name in result.fetchall():
            messages.append({
                "id": message.id,
                "text": message.raw_payload.get("text", ""),
                "user_name": user_name,
                "workspace_name": workspace_name,
                "timestamp": message.raw_payload.get("ts"),
                "thread_ts": message.raw_payload.get("thread_ts"),
                "created_at": message.created_at
            })

        return {"messages": messages, "channel_id": channel_id}

    except Exception as e:
        logger.error("Error getting channel messages: {}", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
