"""Embedding generator worker for creating vector embeddings for knowledge items."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import select, and_, update
from sqlalchemy.orm import sessionmaker
from loguru import logger
import json
import time

from .celery_app import celery_app
from ..core.config import settings
from ..models.base import KnowledgeItem
from ..services.embedding_service import EmbeddingService
from ..services.vector_service import VectorService

def get_async_session():
    """Create a new async session for each task."""
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal

@celery_app.task
def generate_embedding(
    knowledge_id: int, 
    text: str, 
    content_type: str = "knowledge"
):
    """Generate embedding for a single knowledge item using Celery."""
    try:
        logger.info(f"Starting embedding generation for knowledge item {knowledge_id}")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                generate_embedding_async(
                    knowledge_id=knowledge_id,
                    text=text,
                    content_type=content_type
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in embedding generation task: {e}", exc_info=True)
        raise

async def generate_embedding_async(
    knowledge_id: int,
    text: str,
    content_type: str = "knowledge"
) -> Dict[str, Any]:
    """Generate embedding for a knowledge item asynchronously."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            start_time = time.time()
            
            # Initialize services
            embedding_service = EmbeddingService()
            vector_service = VectorService()
            
            # Generate embedding
            embedding = await embedding_service.generate_embedding(text, content_type)
            
            if not embedding:
                logger.error(f"Failed to generate embedding for knowledge item {knowledge_id}")
                return {"status": "error", "message": "Failed to generate embedding"}
            
            # Validate embedding
            if not await embedding_service.validate_embedding(embedding):
                logger.error(f"Generated invalid embedding for knowledge item {knowledge_id}")
                return {"status": "error", "message": "Invalid embedding generated"}
            
            # Store embedding
            storage_success = await vector_service.store_embedding(knowledge_id, embedding, db)
            
            if not storage_success:
                logger.error(f"Failed to store embedding for knowledge item {knowledge_id}")
                return {"status": "error", "message": "Failed to store embedding"}
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully generated and stored embedding for knowledge item {knowledge_id} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "knowledge_id": knowledge_id,
                "embedding_dimensions": len(embedding),
                "processing_time": processing_time,
                "content_type": content_type
            }
            
        except Exception as e:
            logger.error(f"Error generating embedding for knowledge item {knowledge_id}: {e}", exc_info=True)
            await db.rollback()
            raise

@celery_app.task
def generate_embeddings_batch():
    """Generate embeddings for multiple knowledge items in batch."""
    try:
        logger.info("Starting batch embedding generation...")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(generate_embeddings_batch_async())
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in batch embedding generation: {e}", exc_info=True)
        raise

async def generate_embeddings_batch_async() -> Dict[str, Any]:
    """Generate embeddings for multiple knowledge items asynchronously."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            start_time = time.time()
            
            # Find knowledge items without embeddings
            result = await db.execute(
                select(KnowledgeItem)
                .where(
                    and_(
                        KnowledgeItem.embedding.is_(None),
                        KnowledgeItem.content.isnot(None),
                        KnowledgeItem.content != ""
                    )
                )
                .limit(50)  # Process in batches
            )
            
            knowledge_items = result.scalars().all()
            
            if not knowledge_items:
                logger.info("No knowledge items need embeddings")
                return {"status": "success", "processed": 0, "total_time": 0}
            
            logger.info(f"Found {len(knowledge_items)} knowledge items needing embeddings")
            
            # Initialize services
            embedding_service = EmbeddingService()
            vector_service = VectorService()
            
            # Prepare batch data
            texts = []
            content_types = []
            knowledge_ids = []
            
            for item in knowledge_items:
                # Create text for embedding
                text_parts = []
                if item.title:
                    text_parts.append(item.title)
                if item.summary:
                    text_parts.append(item.summary)
                if item.content:
                    text_parts.append(item.content)
                
                if text_parts:
                    combined_text = " | ".join(text_parts)
                    texts.append(combined_text)
                    content_types.append("knowledge")
                    knowledge_ids.append(item.id)
            
            if not texts:
                logger.info("No valid texts found for embedding generation")
                return {"status": "success", "processed": 0, "total_time": 0}
            
            # Generate embeddings in batch
            embeddings = await embedding_service.generate_embeddings_batch(texts, content_types)
            
            # Store embeddings
            embeddings_data = []
            for knowledge_id, embedding in zip(knowledge_ids, embeddings):
                if embedding:
                    embeddings_data.append((knowledge_id, embedding))
            
            stored_count = await vector_service.store_embeddings_batch(embeddings_data, db)
            
            total_time = time.time() - start_time
            
            logger.info(f"Batch embedding generation completed: {stored_count} embeddings generated and stored in {total_time:.2f}s")
            
            return {
                "status": "success",
                "processed": stored_count,
                "total_items": len(knowledge_items),
                "total_time": total_time,
                "avg_time_per_item": total_time / max(stored_count, 1)
            }
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            await db.rollback()
            raise

@celery_app.task
def regenerate_embeddings():
    """Regenerate embeddings for existing knowledge items."""
    try:
        logger.info("Starting embedding regeneration...")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(regenerate_embeddings_async())
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in embedding regeneration: {e}", exc_info=True)
        raise

async def regenerate_embeddings_async() -> Dict[str, Any]:
    """Regenerate embeddings for existing knowledge items asynchronously."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            start_time = time.time()
            
            # Find knowledge items with existing embeddings
            result = await db.execute(
                select(KnowledgeItem)
                .where(KnowledgeItem.embedding.isnot(None))
                .limit(20)  # Process in smaller batches for regeneration
            )
            
            knowledge_items = result.scalars().all()
            
            if not knowledge_items:
                logger.info("No knowledge items with embeddings found for regeneration")
                return {"status": "success", "processed": 0, "total_time": 0}
            
            logger.info(f"Found {len(knowledge_items)} knowledge items for embedding regeneration")
            
            # Initialize services
            embedding_service = EmbeddingService()
            vector_service = VectorService()
            
            regenerated_count = 0
            
            for item in knowledge_items:
                try:
                    # Create text for embedding
                    text_parts = []
                    if item.title:
                        text_parts.append(item.title)
                    if item.summary:
                        text_parts.append(item.summary)
                    if item.content:
                        text_parts.append(item.content)
                    
                    if not text_parts:
                        continue
                    
                    combined_text = " | ".join(text_parts)
                    
                    # Generate new embedding
                    new_embedding = await embedding_service.generate_embedding(combined_text, "knowledge")
                    
                    if new_embedding and await embedding_service.validate_embedding(new_embedding):
                        # Store new embedding
                        if await vector_service.store_embedding(item.id, new_embedding, db):
                            regenerated_count += 1
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error regenerating embedding for knowledge item {item.id}: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            logger.info(f"Embedding regeneration completed: {regenerated_count} embeddings regenerated in {total_time:.2f}s")
            
            return {
                "status": "success",
                "processed": regenerated_count,
                "total_items": len(knowledge_items),
                "total_time": total_time,
                "avg_time_per_item": total_time / max(regenerated_count, 1)
            }
            
        except Exception as e:
            logger.error(f"Error in embedding regeneration: {e}")
            await db.rollback()
            raise

@celery_app.task
def cleanup_embeddings():
    """Clean up invalid or corrupted embeddings."""
    try:
        logger.info("Starting embedding cleanup...")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(cleanup_embeddings_async())
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in embedding cleanup: {e}", exc_info=True)
        raise

async def cleanup_embeddings_async() -> Dict[str, Any]:
    """Clean up invalid or corrupted embeddings asynchronously."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            start_time = time.time()
            
            # Initialize services
            embedding_service = EmbeddingService()
            vector_service = VectorService()
            
            # Find knowledge items with embeddings
            result = await db.execute(
                select(KnowledgeItem)
                .where(KnowledgeItem.embedding.isnot(None))
                .limit(100)  # Process in batches
            )
            
            knowledge_items = result.scalars().all()
            
            if not knowledge_items:
                logger.info("No knowledge items with embeddings found for cleanup")
                return {"status": "success", "cleaned": 0, "total_time": 0}
            
            logger.info(f"Found {len(knowledge_items)} knowledge items for embedding cleanup")
            
            cleaned_count = 0
            
            for item in knowledge_items:
                try:
                    # Parse and validate embedding
                    if item.embedding:
                        try:
                            embedding_data = json.loads(item.embedding)
                            
                            # Validate embedding
                            if not await embedding_service.validate_embedding(embedding_data):
                                logger.warning(f"Invalid embedding found for knowledge item {item.id}, removing...")
                                
                                # Remove invalid embedding
                                await db.execute(
                                    update(KnowledgeItem)
                                    .where(KnowledgeItem.id == item.id)
                                    .values(embedding=None, updated_at=datetime.utcnow())
                                )
                                
                                cleaned_count += 1
                                
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Corrupted embedding found for knowledge item {item.id}, removing...")
                            
                            # Remove corrupted embedding
                            await db.execute(
                                update(KnowledgeItem)
                                .where(KnowledgeItem.id == item.id)
                                .values(embedding=None, updated_at=datetime.utcnow())
                            )
                            
                            cleaned_count += 1
                            
                except Exception as e:
                    logger.error(f"Error cleaning embedding for knowledge item {item.id}: {e}")
                    continue
            
            await db.commit()
            
            total_time = time.time() - start_time
            
            logger.info(f"Embedding cleanup completed: {cleaned_count} embeddings cleaned in {total_time:.2f}s")
            
            return {
                "status": "success",
                "cleaned": cleaned_count,
                "total_items": len(knowledge_items),
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in embedding cleanup: {e}")
            await db.rollback()
            raise

@celery_app.task
def optimize_embeddings():
    """Optimize embeddings for better search performance."""
    try:
        logger.info("Starting embedding optimization...")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(optimize_embeddings_async())
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in embedding optimization: {e}", exc_info=True)
        raise

async def optimize_embeddings_async() -> Dict[str, Any]:
    """Optimize embeddings for better search performance asynchronously."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            start_time = time.time()
            
            # Initialize services
            vector_service = VectorService()
            
            # Get embedding statistics
            stats = await vector_service.get_embedding_stats(db)
            
            # For now, just return stats
            # In the future, this could include:
            # - Reindexing for better performance
            # - Clustering analysis
            # - Quality assessment
            
            total_time = time.time() - start_time
            
            logger.info(f"Embedding optimization completed in {total_time:.2f}s")
            
            return {
                "status": "success",
                "stats": stats,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in embedding optimization: {e}")
            await db.rollback()
            raise

if __name__ == "__main__":
    celery_app.start()
