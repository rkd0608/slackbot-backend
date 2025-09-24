from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.startup import lifespan
from app.core.logging import logger
from app.utils.env_loader import get_env_var

# Import all routers
from app.api.v1 import health, auth, slack, knowledge, search, queries, accuracy

app = FastAPI(
    title="SlackBot Backend",
    description="FastAPI-based Slack bot backend with AI capabilities",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(slack.router, prefix="/api/v1/slack", tags=["slack"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(queries.router, prefix="/api/v1/queries", tags=["queries"])
app.include_router(accuracy.router, prefix="/api/v1/accuracy", tags=["accuracy"])

@app.get("/")
async def root():
    """Root endpoint with API information."""
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
    return {"message": f"Hello {name}"}
