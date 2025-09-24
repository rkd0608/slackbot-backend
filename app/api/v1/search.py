"""Vector search API endpoints for semantic knowledge retrieval."""

import time
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from loguru import logger

from ...core.database import get_db
from ...services.vector_service import VectorService
from ...services.embedding_service import EmbeddingService

router = APIRouter()

@router.get("/semantic")
async def semantic_search(
    query: str = Query(..., description="Search query"),
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    min_confidence: Optional[float] = Query(0.0, description="Minimum confidence score"),
    limit: int = Query(10, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
):
    """Perform semantic search using vector embeddings."""
    try:
        start_time = time.time()
        
        # Initialize services
        vector_service = VectorService()
        
        # Perform hybrid search (vector + text)
        search_results = await vector_service.hybrid_search(
            query=query,
            workspace_id=workspace_id,
            knowledge_type=knowledge_type,
            min_confidence=min_confidence,
            limit=limit,
            db=db
        )
        
        search_time = time.time() - start_time
        
        # Performance validation
        if search_time > 0.5:  # 500ms threshold
            logger.warning(f"Search response time ({search_time:.3f}s) exceeded 500ms threshold")
        
        return {
            "query": query,
            "results": search_results,
            "total_found": len(search_results),
            "search_time": round(search_time, 3),
            "performance": "optimal" if search_time <= 0.5 else "suboptimal",
            "filters": {
                "workspace_id": workspace_id,
                "knowledge_type": knowledge_type,
                "min_confidence": min_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/vector")
async def vector_search(
    query: str = Query(..., description="Search query"),
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    min_confidence: Optional[float] = Query(0.0, description="Minimum confidence score"),
    limit: int = Query(10, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
):
    """Perform pure vector similarity search."""
    try:
        start_time = time.time()
        
        # Initialize services
        vector_service = VectorService()
        embedding_service = EmbeddingService()
        
        # Generate embedding for query
        query_embedding = await embedding_service.generate_embedding(query, "question")
        if not query_embedding:
            raise HTTPException(status_code=400, detail="Failed to generate query embedding")
        
        # Perform vector search
        search_results = await vector_service.search_similar(
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            knowledge_type=knowledge_type,
            min_confidence=min_confidence,
            limit=limit,
            db=db
        )
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "results": search_results,
            "total_found": len(search_results),
            "search_time": round(search_time, 3),
            "search_method": "vector",
            "filters": {
                "workspace_id": workspace_id,
                "knowledge_type": knowledge_type,
                "min_confidence": min_confidence
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in vector search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/text")
async def text_search(
    query: str = Query(..., description="Search query"),
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    min_confidence: Optional[float] = Query(0.0, description="Minimum confidence score"),
    limit: int = Query(10, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
):
    """Perform text-based search as fallback."""
    try:
        start_time = time.time()
        
        # Initialize services
        vector_service = VectorService()
        
        # Perform text search
        search_results = await vector_service._text_search(
            query=query,
            workspace_id=workspace_id,
            knowledge_type=knowledge_type,
            min_confidence=min_confidence,
            limit=limit,
            db=db
        )
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "results": search_results,
            "total_found": len(search_results),
            "search_time": round(search_time, 3),
            "search_method": "text",
            "filters": {
                "workspace_id": workspace_id,
                "knowledge_type": knowledge_type,
                "min_confidence": min_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Error in text search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/hybrid")
async def hybrid_search(
    query: str = Query(..., description="Search query"),
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    min_confidence: Optional[float] = Query(0.0, description="Minimum confidence score"),
    limit: int = Query(10, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
):
    """Perform hybrid search combining vector and text methods."""
    try:
        start_time = time.time()
        
        # Initialize services
        vector_service = VectorService()
        
        # Perform hybrid search
        search_results = await vector_service.hybrid_search(
            query=query,
            workspace_id=workspace_id,
            knowledge_type=knowledge_type,
            min_confidence=min_confidence,
            limit=limit,
            db=db
        )
        
        search_time = time.time() - start_time
        
        # Analyze result distribution
        vector_count = len([r for r in search_results if r.get("search_method") == "vector"])
        text_count = len([r for r in search_results if r.get("search_method") == "text"])
        hybrid_count = len([r for r in search_results if r.get("search_method") == "hybrid"])
        
        return {
            "query": query,
            "results": search_results,
            "total_found": len(search_results),
            "search_time": round(search_time, 3),
            "search_method": "hybrid",
            "result_distribution": {
                "vector_only": vector_count,
                "text_only": text_count,
                "hybrid": hybrid_count
            },
            "filters": {
                "workspace_id": workspace_id,
                "knowledge_type": knowledge_type,
                "min_confidence": min_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/suggest")
async def search_suggestions(
    query: str = Query(..., description="Partial search query"),
    limit: int = Query(5, description="Maximum number of suggestions"),
    db: AsyncSession = Depends(get_db)
):
    """Get search suggestions based on partial query."""
    try:
        start_time = time.time()
        
        # Initialize services
        vector_service = VectorService()
        
        # Generate embedding for partial query
        embedding_service = EmbeddingService()
        query_embedding = await embedding_service.generate_embedding(query, "question")
        
        if query_embedding:
            # Get vector-based suggestions
            vector_suggestions = await vector_service.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                db=db
            )
            
            # Extract titles as suggestions
            suggestions = [result["title"] for result in vector_suggestions if result.get("title")]
        else:
            # Fallback to text-based suggestions
            suggestions = []
        
        # Add common patterns if we don't have enough suggestions
        if len(suggestions) < limit:
            common_patterns = [
                "How to",
                "What is",
                "Why does",
                "When should",
                "Where can I"
            ]
            
            for pattern in common_patterns:
                if pattern.lower().startswith(query.lower()) and len(suggestions) < limit:
                    suggestions.append(pattern)
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "suggestions": suggestions[:limit],
            "total_suggestions": len(suggestions),
            "search_time": round(search_time, 3)
        }
        
    except Exception as e:
        logger.error(f"Error in search suggestions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/performance")
async def search_performance_metrics(
    db: AsyncSession = Depends(get_db)
):
    """Get search performance metrics and statistics."""
    try:
        # Initialize services
        vector_service = VectorService()
        embedding_service = EmbeddingService()
        
        # Get embedding statistics
        embedding_stats = await vector_service.get_embedding_stats(db)
        
        # Get service status
        vector_status = await vector_service.get_service_status()
        embedding_status = await embedding_service.get_service_status()
        
        return {
            "embedding_stats": embedding_stats,
            "vector_service": vector_status,
            "embedding_service": embedding_status,
            "performance_targets": {
                "search_response_time_ms": 500,
                "embedding_generation_time_ms": 1000,
                "similarity_threshold": 0.7
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/test")
async def test_search_performance(
    test_queries: List[str] = Query(..., description="List of test queries"),
    db: AsyncSession = Depends(get_db)
):
    """Test search performance with multiple queries."""
    try:
        if not test_queries or len(test_queries) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 test queries allowed")
        
        # Initialize services
        vector_service = VectorService()
        
        performance_results = []
        total_time = 0
        
        for query in test_queries:
            query_start = time.time()
            
            # Perform hybrid search
            results = await vector_service.hybrid_search(
                query=query,
                limit=5,
                db=db
            )
            
            query_time = time.time() - query_start
            total_time += query_time
            
            performance_results.append({
                "query": query,
                "results_count": len(results),
                "response_time": round(query_time, 3),
                "performance": "optimal" if query_time <= 0.5 else "suboptimal"
            })
        
        avg_time = total_time / len(test_queries)
        
        return {
            "test_queries": len(test_queries),
            "performance_results": performance_results,
            "total_time": round(total_time, 3),
            "average_time": round(avg_time, 3),
            "overall_performance": "optimal" if avg_time <= 0.5 else "suboptimal"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in performance test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
