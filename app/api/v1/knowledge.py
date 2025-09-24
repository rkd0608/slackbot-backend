"""Knowledge management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from typing import List, Optional, Dict, Any
from loguru import logger

from ...core.database import get_db
from ...models.base import KnowledgeItem, Message, User, Workspace
from ...services.openai_service import OpenAIService

router = APIRouter()

@router.get("/")
async def list_knowledge(
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    min_confidence: Optional[float] = Query(0.0, description="Minimum confidence score"),
    max_confidence: Optional[float] = Query(1.0, description="Maximum confidence score"),
    limit: int = Query(50, description="Maximum number of items to return"),
    offset: int = Query(0, description="Number of items to skip"),
    db: AsyncSession = Depends(get_db)
):
    """List knowledge items with filtering options."""
    try:
        # Build query
        query = select(KnowledgeItem, Workspace.name)
        query = query.join(Workspace, KnowledgeItem.workspace_id == Workspace.id)
        
        # Apply filters
        if workspace_id:
            query = query.where(KnowledgeItem.workspace_id == workspace_id)
        
        if knowledge_type:
            # Use JSONB containment operator for type filtering
            query = query.where(KnowledgeItem.metadata.contains({"type": knowledge_type}))
        
        if min_confidence > 0:
            query = query.where(KnowledgeItem.confidence >= min_confidence)
        
        if max_confidence < 1.0:
            query = query.where(KnowledgeItem.confidence <= max_confidence)
        
        # Order by confidence and creation date
        query = query.order_by(KnowledgeItem.confidence.desc(), KnowledgeItem.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        # Execute query
        result = await db.execute(query)
        rows = result.fetchall()
        
        # Format response
        knowledge_items = []
        for item, workspace_name in rows:
            # Safely access metadata fields
            metadata = item.item_metadata or {}
            knowledge_items.append({
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "content": item.content,
                "confidence": item.confidence,
                "type": metadata.get("type", "unknown"),
                "participants": metadata.get("participants", []),
                "tags": metadata.get("tags", []),
                "workspace_name": workspace_name,
                "source_user": metadata.get("source_user_id", "unknown"),
                "source_channel": metadata.get("source_channel_id"),
                "created_at": item.created_at,
                "updated_at": item.updated_at
            })
        
        return {
            "knowledge_items": knowledge_items,
            "total_returned": len(knowledge_items),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing knowledge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{knowledge_id}")
async def get_knowledge_item(
    knowledge_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific knowledge item by ID."""
    try:
        result = await db.execute(
            select(KnowledgeItem, Workspace.name)
            .join(Workspace, KnowledgeItem.workspace_id == Workspace.id)
            .where(KnowledgeItem.id == knowledge_id)
        )
        
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        item, workspace_name = row
        
        # Safely access metadata fields
        metadata = item.item_metadata or {}
        
        return {
            "id": item.id,
            "title": item.title,
            "summary": item.summary,
            "content": item.content,
            "confidence": item.confidence,
            "type": metadata.get("type", "unknown"),
            "participants": metadata.get("participants", []),
            "tags": metadata.get("tags", []),
            "source_context": metadata.get("source_context", ""),
            "workspace_name": workspace_name,
            "source_user": metadata.get("source_user_id", "unknown"),
            "source_channel": metadata.get("source_channel_id"),
            "source_message_id": metadata.get("source_message_id"),
            "extraction_metadata": metadata.get("extraction_metadata", {}),
            "verification_metadata": metadata.get("verification_metadata", {}),
            "verification_result": metadata.get("verification_result", {}),
            "created_at": item.created_at,
            "updated_at": item.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge item {knowledge_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/search/")
async def search_knowledge(
    query: str = Query(..., description="Search query"),
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    min_confidence: Optional[float] = Query(0.0, description="Minimum confidence score"),
    limit: int = Query(50, description="Maximum number of items to return"),
    db: AsyncSession = Depends(get_db)
):
    """Search knowledge items by text content."""
    try:
        # Build search query
        search_query = select(KnowledgeItem, Workspace.name)
        search_query = search_query.join(Workspace, KnowledgeItem.workspace_id == Workspace.id)
        
        # Apply text search (simple LIKE search for now, will be enhanced with embeddings later)
        search_query = search_query.where(
            or_(
                KnowledgeItem.title.ilike(f"%{query}%"),
                KnowledgeItem.summary.ilike(f"%{query}%"),
                KnowledgeItem.content.ilike(f"%{query}%"),
                KnowledgeItem.metadata["tags"].astext.ilike(f"%{query}%")
            )
        )
        
        # Apply filters
        if workspace_id:
            search_query = search_query.where(KnowledgeItem.workspace_id == workspace_id)
        
        if knowledge_type:
            search_query = search_query.where(KnowledgeItem.metadata["type"].astext == knowledge_type)
        
        if min_confidence > 0:
            search_query = search_query.where(KnowledgeItem.confidence >= min_confidence)
        
        # Order by relevance (confidence) and creation date
        search_query = search_query.order_by(KnowledgeItem.confidence.desc(), KnowledgeItem.created_at.desc())
        search_query = search_query.limit(limit)
        
        # Execute query
        result = await db.execute(search_query)
        rows = result.fetchall()
        
        # Format response
        knowledge_items = []
        for item, workspace_name in rows:
            # Safely access metadata fields
            metadata = item.metadata or {}
            knowledge_items.append({
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "content": item.content,
                "confidence": item.confidence,
                "type": metadata.get("type", "unknown"),
                "participants": metadata.get("participants", []),
                "tags": metadata.get("tags", []),
                "workspace_name": workspace_name,
                "source_user": metadata.get("source_user_id", "unknown"),
                "created_at": item.created_at
            })
        
        return {
            "query": query,
            "knowledge_items": knowledge_items,
            "total_found": len(knowledge_items),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats/overview")
async def get_knowledge_stats(
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    db: AsyncSession = Depends(get_db)
):
    """Get overview statistics for knowledge items."""
    try:
        # Build base query
        base_query = select(KnowledgeItem)
        if workspace_id:
            base_query = base_query.where(KnowledgeItem.workspace_id == workspace_id)
        
        # Total count
        total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
        total_count = total_result.scalar()
        
        # Count by type
        type_query = base_query.add_columns(
            KnowledgeItem.metadata["type"].astext.label("type"),
            func.count().label("count")
        ).group_by(KnowledgeItem.metadata["type"].astext)
        
        type_result = await db.execute(type_query)
        type_counts = {row.type or "unknown": row.count for row in type_result.fetchall()}
        
        # Confidence distribution
        confidence_query = base_query.add_columns(
            func.avg(KnowledgeItem.confidence).label("avg_confidence"),
            func.min(KnowledgeItem.confidence).label("min_confidence"),
            func.max(KnowledgeItem.confidence).label("max_confidence")
        )
        
        confidence_result = await db.execute(confidence_query)
        confidence_stats = confidence_result.fetchone()
        
        # Recent activity
        recent_query = base_query.add_columns(
            func.count().label("count")
        ).where(KnowledgeItem.created_at >= func.now() - func.interval('7 days'))
        
        recent_result = await db.execute(recent_query)
        recent_count = recent_result.scalar()
        
        return {
            "total_knowledge_items": total_count,
            "by_type": type_counts,
            "confidence_stats": {
                "average": float(confidence_stats.avg_confidence or 0),
                "minimum": float(confidence_stats.min_confidence or 0),
                "maximum": float(confidence_stats.max_confidence or 0)
            },
            "recent_activity": {
                "last_7_days": recent_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats/quality")
async def get_knowledge_quality_stats(
    workspace_id: Optional[int] = Query(None, description="Filter by workspace ID"),
    db: AsyncSession = Depends(get_db)
):
    """Get quality metrics for knowledge items."""
    try:
        # Build base query
        base_query = select(KnowledgeItem)
        if workspace_id:
            base_query = base_query.where(KnowledgeItem.workspace_id == workspace_id)
        
        # Verification statistics
        verification_query = base_query.add_columns(
            func.count().label("total"),
            func.sum(
                func.case(
                    (KnowledgeItem.metadata["verification_result"]["hallucination_detected"].astext == "true", 1),
                    else_=0
                )
            ).label("hallucinations_detected"),
            func.avg(
                func.case(
                    (KnowledgeItem.metadata["verification_result"]["overall_verification_score"].astext.cast(float), 
                     KnowledgeItem.metadata["verification_result"]["overall_verification_score"].astext.cast(float)),
                    else_=0.0
                )
            ).label("avg_verification_score")
        )
        
        verification_result = await db.execute(verification_query)
        verification_stats = verification_result.fetchone()
        
        # Confidence distribution
        confidence_ranges = [
            (0.0, 0.3, "low"),
            (0.3, 0.7, "medium"),
            (0.7, 1.0, "high")
        ]
        
        confidence_distribution = {}
        for min_conf, max_conf, label in confidence_ranges:
            range_query = base_query.where(
                and_(
                    KnowledgeItem.confidence >= min_conf,
                    KnowledgeItem.confidence < max_conf
                )
            )
            count_result = await db.execute(select(func.count()).select_from(range_query.subquery()))
            confidence_distribution[label] = count_result.scalar()
        
        return {
            "verification_stats": {
                "total_items": verification_stats.total,
                "hallucinations_detected": verification_stats.hallucinations_detected or 0,
                "hallucination_rate": (verification_stats.hallucinations_detected or 0) / max(verification_stats.total, 1),
                "average_verification_score": float(verification_stats.avg_verification_score or 0)
            },
            "confidence_distribution": confidence_distribution,
            "quality_metrics": {
                "high_confidence_rate": confidence_distribution.get("high", 0) / max(verification_stats.total, 1),
                "verification_coverage": "partial"  # Will be enhanced with more metrics
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting quality stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{knowledge_id}/verify")
async def verify_knowledge_item(
    knowledge_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Manually trigger verification of a knowledge item."""
    try:
        # Get the knowledge item
        result = await db.execute(
            select(KnowledgeItem).where(KnowledgeItem.id == knowledge_id)
        )
        item = result.scalar_one_or_none()
        
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        # Get source message for context
        source_message_id = item.metadata.get("source_message_id")
        if not source_message_id:
            raise HTTPException(status_code=400, detail="No source message found")
        
        source_message = await db.execute(
            select(Message).where(Message.id == source_message_id)
        ).scalar_one_or_none()
        
        if not source_message:
            raise HTTPException(status_code=400, detail="Source message not found")
        
        # Get conversation context
        from ...workers.knowledge_extractor import get_conversation_context
        conversation_context = await get_conversation_context(
            item.workspace_id,
            source_message.channel_id,
            source_message_id,
            db
        )
        
        # Perform verification
        openai_service = OpenAIService()
        verification_result = await openai_service.verify_extraction(
            extracted_knowledge={"knowledge_items": [item.metadata]},
            source_messages=conversation_context
        )
        
        # Update verification metadata
        item.metadata["verification_metadata"] = verification_result.get("verification_metadata", {})
        item.metadata["verification_result"] = verification_result
        
        # Update confidence based on verification
        if verification_result.get("overall_verification_score"):
            verification_score = verification_result["overall_verification_score"]
            item.confidence = (item.confidence * 0.7) + (verification_score * 0.3)
        
        await db.commit()
        
        return {
            "status": "success",
            "knowledge_id": knowledge_id,
            "verification_result": verification_result,
            "updated_confidence": item.confidence
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying knowledge item {knowledge_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{knowledge_id}")
async def delete_knowledge_item(
    knowledge_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a knowledge item."""
    try:
        result = await db.execute(
            select(KnowledgeItem).where(KnowledgeItem.id == knowledge_id)
        )
        item = result.scalar_one_or_none()
        
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        await db.delete(item)
        await db.commit()
        
        logger.info(f"Deleted knowledge item {knowledge_id}")
        
        return {"status": "success", "message": "Knowledge item deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge item {knowledge_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
