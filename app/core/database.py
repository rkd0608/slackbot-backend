"""Database setup with SQLAlchemy async engine."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from .config import settings

# Base class for all models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass

# Global variables for engine and session factory
_engine = None
_AsyncSessionLocal = None

def get_engine():
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.db_echo,  # SQL query logging
            poolclass=NullPool,  # Use NullPool for async engine
            future=True,  # Use SQLAlchemy 2.0 style
        )
    return _engine

def get_session_factory():
    """Get or create the async session factory."""
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _AsyncSessionLocal

# Dependency to get database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

# Export AsyncSessionLocal for worker scripts
AsyncSessionLocal = get_session_factory

# Database health check
async def check_db_health() -> bool:
    """Check if database is accessible."""
    try:
        engine = get_engine()
        print(f"DEBUG: Attempting to connect to database with URL: {settings.database_url}")
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("DEBUG: Database connection successful")
        return True
    except Exception as e:
        print(f"DEBUG: Database connection failed with error: {str(e)}")
        return False

# Database initialization
async def init_db() -> None:
    """Initialize database tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        # This will be done when models are imported elsewhere
        await conn.run_sync(Base.metadata.create_all)

# Database cleanup
async def close_db() -> None:
    """Close database connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
