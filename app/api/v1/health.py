"""Health check endpoint."""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...core.database import get_db, check_db_health

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "slackbot-backend",
        "timestamp": "2025-01-28T23:00:00Z"
    }

@router.get("/health/db")
async def database_health_check():
    """Database connection health check endpoint."""
    try:
        # Check database connection
        is_healthy = await check_db_health()
        
        if is_healthy:
            return {
                "status": "healthy",
                "database": "connected",
                "message": "Database connection successful"
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Database connection failed"
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database health check failed: {str(e)}"
        )

@router.get("/health/db/query")
async def database_query_check(db: AsyncSession = Depends(get_db)):
    """Database query health check endpoint."""
    try:
        # Execute a simple query
        result = await db.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        
        if row and row.test == 1:
            return {
                "status": "healthy",
                "database": "query_successful",
                "message": "Database query executed successfully",
                "result": row.test
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Database query returned unexpected result"
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database query failed: {str(e)}"
        )
