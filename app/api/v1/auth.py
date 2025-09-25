"""Slack OAuth authentication endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError
import json
from typing import Optional
import httpx
from loguru import logger
from datetime import datetime

from ...core.database import get_db
from ...utils.env_loader import get_env_var, get_env_var_required

router = APIRouter()


@router.get("/slack/install")
async def slack_install():
    """Redirect to Slack OAuth installation page."""
    logger.info("Starting Slack OAuth installation flow")
    
    try:
        # Get fresh environment variables on each request
        client_id = get_env_var_required("SLACK_CLIENT_ID")
        app_base_url = get_env_var("APP_BASE_URL", "http://localhost:8000")
        
        scope = "channels:history,channels:read,chat:write,chat:write.public,commands,groups:history,groups:read,im:history,im:write,mpim:history,search:read.users,team:read,users:read,users:read.email"
        redirect_uri = f"{app_base_url}/api/v1/auth/slack/callback"
        
        slack_oauth_url = f"https://slack.com/oauth/v2/authorize?client_id={client_id}&scope={scope}&redirect_uri={redirect_uri}"
        
        logger.info(f"redirect_uri: {redirect_uri}")
        logger.info(f"Redirecting to Slack OAuth URL with client_id: {client_id[:8]}...")
        logger.debug(f"Full OAuth URL: {slack_oauth_url}")
        
        return RedirectResponse(url=slack_oauth_url)
        
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate Slack OAuth URL: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate OAuth URL")


@router.get("/slack/callback")
async def slack_callback(
    code: str,
    state: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Handle Slack OAuth callback and create workspace."""
    logger.info("Received Slack OAuth callback")
    logger.debug(f"OAuth code received: {code[:8]}...")
    logger.debug(f"OAuth state: {state}")
    
    try:
        # Get fresh environment variables on each request
        client_id = get_env_var_required("SLACK_CLIENT_ID")
        client_secret = get_env_var_required("SLACK_CLIENT_SECRET")
        
        # Exchange authorization code for access token
        logger.info("Exchanging OAuth code for access token")
        client = AsyncWebClient()
        oauth_response = await client.oauth_v2_access(
            client_id=client_id,
            client_secret=client_secret,
            code=code
        )
        
        if not oauth_response["ok"]:
            logger.error(f"Slack OAuth failed: {oauth_response}")
            raise HTTPException(status_code=400, detail="Failed to get access token from Slack")
        
        logger.info("Successfully obtained access token from Slack")
        
        # Extract workspace information
        workspace_info = oauth_response["team"]
        access_token = oauth_response["access_token"]
        
        logger.info(f"Workspace ID: {workspace_info['id']}, Name: {workspace_info.get('name', 'Unknown')}")
        logger.debug(f"Access token received: {access_token[:8]}...")
        
        # Get additional workspace details using the access token
        logger.info("Fetching additional workspace details")
        workspace_client = AsyncWebClient(token=access_token)
        team_info = await workspace_client.team_info()
        
        if not team_info["ok"]:
            logger.error(f"Failed to get team info: {team_info}")
            raise HTTPException(status_code=400, detail="Failed to get workspace info from Slack")
        
        team_data = team_info["team"]
        logger.info(f"Team details: {team_data['name']} (ID: {team_data['id']})")
        
        # Check if workspace already exists using ORM
        logger.info("Checking if workspace already exists")
        from ...models.base import Workspace
        
        result = await db.execute(
            select(Workspace).where(Workspace.slack_id == workspace_info["id"])
        )
        existing_workspace = result.scalar_one_or_none()
        
        if existing_workspace:
            logger.info(f"Updating existing workspace: {existing_workspace.id}")
            # Update existing workspace tokens directly
            existing_workspace.tokens = {
                "access_token": access_token,
                "bot_user_id": oauth_response.get("bot_user_id"),
                "scope": oauth_response.get("scope"),
                "installed_at": oauth_response.get("installed_at")
            }
            existing_workspace.updated_at = func.now()
            await db.flush()
            
            workspace_id = existing_workspace.id
            workspace_name = existing_workspace.name
        else:
            logger.info("Creating new workspace")
            # Create new workspace using ORM model
            from ...models.base import Workspace
            
            new_workspace = Workspace(
                name=team_data["name"],
                slack_id=workspace_info["id"],
                tokens={
                    "access_token": access_token,
                    "bot_user_id": oauth_response.get("bot_user_id"),
                    "scope": oauth_response.get("scope"),
                    "installed_at": oauth_response.get("installed_at")
                }
            )
            
            db.add(new_workspace)
            await db.flush()  # This will populate the ID
            workspace_id = new_workspace.id
            workspace_name = team_data["name"]
        
        # Commit to database
        logger.info("Committing workspace changes to database")
        await db.commit()
        logger.info(f"Workspace saved successfully: ID={workspace_id}, Name={workspace_name}")
        
        # Return success page
        logger.info("Returning success page to user")
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Slack App Installation Successful</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
                .success {{ color: #36a64f; font-size: 24px; margin-bottom: 20px; }}
                .workspace {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .token-info {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107; }}
            </style>
        </head>
        <body>
            <div class="success">✅ Installation Successful!</div>
            <h2>Workspace: {workspace_name}</h2>
            <div class="workspace">
                <p><strong>Workspace ID:</strong> {workspace_info["id"]}</p>
                <p><strong>Created:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="token-info">
                <p><strong>Bot User ID:</strong> {oauth_response.get('bot_user_id', 'N/A')}</p>
                <p><strong>Scopes:</strong> {oauth_response.get('scope', 'N/A')}</p>
                <p><em>Access token has been securely stored in the database.</em></p>
            </div>
            <p>Your Slack app is now installed and ready to use!</p>
        </body>
        </html>
        """)
        
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Slack API error: {e.response['error']}")
    except Exception as e:
        logger.error(f"Unexpected error during OAuth callback: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/workspaces")
async def list_workspaces(db: AsyncSession = Depends(get_db)):
    """List all installed workspaces (for debugging/admin purposes)."""
    from ...models.base import Workspace
    logger.info("Listing all workspaces")
    
    try:
        result = await db.execute(select(Workspace))
        workspaces = result.scalars().all()
        
        logger.info(f"Found {len(workspaces)} workspaces")
        
        workspace_list = [
            {
                "id": w.id,
                "name": w.name,
                "slack_id": w.slack_id,
                "created_at": w.created_at,
                "updated_at": w.updated_at,
                "has_tokens": bool(w.tokens.get("access_token")),
                "bot_user_id": w.tokens.get("bot_user_id"),
                "scope": w.tokens.get("scope")
            }
            for w in workspaces
        ]
        
        logger.debug(f"Workspace list: {workspace_list}")
        return {"workspaces": workspace_list}
        
    except Exception as e:
        logger.error(f"Failed to fetch workspaces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch workspaces: {str(e)}")


@router.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Get specific workspace details."""
    from ...models.base import Workspace
    logger.info(f"Fetching workspace details for ID: {workspace_id}")
    
    try:
        result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
        workspace = result.scalar_one_or_none()
        
        if not workspace:
            logger.warning(f"Workspace not found with ID: {workspace_id}")
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        logger.info(f"Found workspace: {workspace.name} (ID: {workspace.id})")
        
        workspace_data = {
            "id": workspace.id,
            "name": workspace.name,
            "slack_id": workspace.slack_id,
            "created_at": workspace.created_at,
            "updated_at": workspace.updated_at,
            "has_tokens": bool(workspace.tokens.get("access_token")),
            "bot_user_id": workspace.tokens.get("bot_user_id"),
            "scope": workspace.tokens.get("scope")
        }
        
        logger.debug(f"Workspace data: {workspace_data}")
        return workspace_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch workspace {workspace_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
