"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging first
from .core.logging import logger

from .api.v1 import health, auth, slack, knowledge, search, accuracy
from .utils.env_loader import get_env_var

# Create FastAPI app
app = FastAPI(
    title="SlackBot Backend",
    description="FastAPI-based Slack bot backend with AI capabilities",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(slack.router, prefix="/api/v1/slack", tags=["slack"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(accuracy.router, prefix="/api/v1/accuracy", tags=["accuracy"])

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("🚀 SlackBot Backend starting up...")
    logger.info(f"📚 API Documentation available at: {app.docs_url}")
    logger.info(f"🔍 Alternative docs at: {app.redoc_url}")
    
    # Log current configuration
    app_base_url = get_env_var("APP_BASE_URL", "http://localhost:8000")
    logger.info(f"🌐 App Base URL: {app_base_url}")
    
    slack_client_id = get_env_var("SLACK_CLIENT_ID", "Not configured")
    if slack_client_id != "Not configured":
        logger.info(f"🔑 Slack Client ID: {slack_client_id[:8]}...")
    else:
        logger.warning("⚠️ SLACK_CLIENT_ID not configured")
    
    # Trigger automatic conversation backfill
    try:
        from .core.startup import trigger_automatic_backfill
        logger.info("🔄 Triggering automatic conversation backfill...")
        await trigger_automatic_backfill()
        logger.info("✅ Automatic backfill check completed")
    except Exception as e:
        logger.warning(f"⚠️ Automatic backfill failed: {e}")
        # Don't fail startup for backfill errors

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("🛑 SlackBot Backend shutting down...")

@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("Root endpoint accessed")
    
    # Get fresh environment variables
    app_base_url = get_env_var("APP_BASE_URL", "http://localhost:8000")
    
    return {
        "message": "SlackBot Backend API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "auth": "/api/v1/auth",
        "slack": "/api/v1/slack",
        "knowledge": "/api/v1/knowledge",
        "search": "/api/v1/search",
        "app_base_url": app_base_url
    }

@app.get("/hello/{name}")
async def say_hello(name: str):
    """Hello endpoint."""
    logger.info(f"Hello endpoint accessed with name: {name}")
    return {"message": f"Hello {name}"}

@app.get("/health/backfill")
async def backfill_health_check():
    """Check conversation backfill status."""
    try:
        from .core.startup import check_backfill_status
        await check_backfill_status()
        return {"status": "healthy", "message": "Backfill status checked"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
