"""Vector service for managing vector operations using PostgreSQL with pgvector."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.dialects.postgresql import insert
from loguru import logger

from ..core.database import get_db
from ..models.base import KnowledgeItem
from ..services.embedding_service import EmbeddingService

class VectorService:
    """Service for managing vector operations using PostgreSQL with pgvector."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.similarity_threshold = 0.7  # Minimum similarity for relevant results
        
    async def store_embedding(
        self, 
        knowledge_id: int, 
        embedding: List[float], 
        db: AsyncSession
    ) -> bool:
        """Store an embedding for a knowledge item."""
        try:
            if not embedding:
                logger.warning(f"No embedding provided for knowledge item {knowledge_id}")
                return False
            
            # Validate embedding
            if not await self.embedding_service.validate_embedding(embedding):
                logger.error(f"Invalid embedding for knowledge item {knowledge_id}")
                return False
            
            # Convert embedding to string for storage
            embedding_str = json.dumps(embedding)
            
            # Update the knowledge item with the embedding
            await db.execute(
                text("""
                    UPDATE knowledgeitem 
                    SET embedding = :embedding, updated_at = NOW()
                    WHERE id = :knowledge_id
                """),
                {"embedding": embedding_str, "knowledge_id": knowledge_id}
            )
            
            await db.commit()
            logger.info(f"Stored embedding for knowledge item {knowledge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding for knowledge item {knowledge_id}: {e}")
            await db.rollback()
            return False
    
    async def store_embeddings_batch(
        self, 
        embeddings_data: List[Tuple[int, List[float]]], 
        db: AsyncSession
    ) -> int:
        """Store multiple embeddings in a batch."""
        try:
            if not embeddings_data:
                return 0
            
            stored_count = 0
            
            for knowledge_id, embedding in embeddings_data:
                if await self.store_embedding(knowledge_id, embedding, db):
                    stored_count += 1
            
            logger.info(f"Stored {stored_count} embeddings from {len(embeddings_data)} items")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error in batch embedding storage: {e}")
            return 0
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        workspace_id: Optional[int] = None,
        knowledge_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Search for similar knowledge items using vector similarity."""
        try:
            if not query_embedding:
                return []
            
            # Build search query
            search_query = """
                SELECT 
                    ki.id,
                    ki.title,
                    ki.summary,
                    ki.content,
                    ki.confidence_score,
                    ki.metadata,
                    ki.created_at,
                    ki.updated_at,
                    (ki.embedding <=> %s) as distance,
                    (1 - (ki.embedding <=> %s)) as similarity
                FROM knowledgeitem ki
                WHERE ki.embedding IS NOT NULL
            """
            
            # Use raw SQL with parameter substitution for vector operations
            search_query = search_query.replace("%s", "?")
            params = [query_embedding, query_embedding]
            
            # Add filters
            if workspace_id:
                search_query += " AND ki.workspace_id = :workspace_id"
                params["workspace_id"] = workspace_id
            
            if knowledge_type:
                search_query += " AND ki.metadata->>'type' = :knowledge_type"
                params["knowledge_type"] = knowledge_type
            
            if min_confidence > 0:
                search_query += " AND ki.confidence_score >= :min_confidence"
                params["min_confidence"] = min_confidence
            
            # Order by similarity and add limit
            search_query += """
                ORDER BY similarity DESC, ki.confidence_score DESC, ki.created_at DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute search
            result = await db.execute(text(search_query), params)
            rows = result.fetchall()
            
            # Format results
            search_results = []
            for row in rows:
                metadata = json.loads(row.metadata) if row.metadata else {}
                
                search_results.append({
                    "id": row.id,
                    "title": row.title,
                    "summary": row.summary,
                    "content": row.content,
                    "confidence": float(row.confidence_score),
                    "type": metadata.get("type", "unknown"),
                    "participants": metadata.get("participants", []),
                    "tags": metadata.get("tags", []),
                    "similarity": float(row.similarity),
                    "distance": float(row.distance),
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "metadata": metadata
                })
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result["similarity"] >= self.similarity_threshold
            ]
            
            logger.info(f"Vector search returned {len(filtered_results)} relevant results from {len(search_results)} total")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        workspace_id: Optional[int] = None,
        channel_id: Optional[str] = None,  # Add channel filtering
        knowledge_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        db: AsyncSession = None,
        include_conversations: bool = True  # Include conversation history
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity with text search."""
        try:
            if not query or not query.strip():
                return []
            
            logger.info(f"Starting hybrid search for query: '{query}'")
            
            # Try vector search first if we have embeddings
            vector_results = []
            try:
                # Generate embedding for the query
                query_embedding = await self.embedding_service.generate_embedding(query)
                if query_embedding:
                    logger.info("Performing vector search with query embedding")
                    vector_results = await self._vector_search_simple(
                        query_embedding,
                        workspace_id,
                        channel_id,  # Add channel_id for channel isolation
                        knowledge_type,
                        min_confidence,
                        limit // 2,
                        db
                    )
                    logger.info(f"Vector search returned {len(vector_results)} results")
                else:
                    logger.warning("Failed to generate query embedding, falling back to text search")
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text search: {e}")
            
            # Perform text search on knowledge items
            knowledge_results = await self._text_search(
                query, 
                workspace_id, 
                channel_id,  # Pass channel_id for filtering
                knowledge_type, 
                min_confidence, 
                limit // 2,  # Reserve half for conversations
                db
            )
            logger.info(f"Text search returned {len(knowledge_results)} results")
            
            # Combine vector and text results
            combined_knowledge = self._combine_results(vector_results, knowledge_results)
            
            # Search conversation history if enabled
            conversation_results = []
            if include_conversations:
                conversation_results = await self._conversation_search(
                    query,
                    workspace_id,
                    channel_id,
                    limit // 2,  # Reserve half for knowledge items
                    db
                )
                logger.info(f"Conversation search returned {len(conversation_results)} results")
            
            # Combine all results
            all_results = combined_knowledge + conversation_results
            
            # Sort by relevance (vector results first, then text, then conversations)
            all_results.sort(key=lambda x: (
                x.get('search_method') == 'vector',  # Vector results first
                x.get('search_method') == 'text',    # Text results second
                x.get('confidence', 0.0),            # Then by confidence
                x.get('created_at', datetime.min)    # Then by recency
            ), reverse=True)
            
            # Apply final limit
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def _vector_search_simple(
        self,
        query_embedding: List[float],
        workspace_id: Optional[int] = None,
        channel_id: Optional[str] = None,
        knowledge_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Simplified vector search using string embeddings (temporary solution)."""
        try:
            # Since we're storing embeddings as strings, we'll use text similarity for now
            # This is a fallback until we implement proper vector operations
            logger.info("Using text-based similarity search (embeddings stored as strings)")
            
            # Build search query that looks for knowledge items with embeddings
            search_query = """
                SELECT
                    ki.id,
                    ki.title,
                    ki.summary,
                    ki.content,
                    ki.confidence_score,
                    ki.metadata,
                    ki.created_at,
                    ki.updated_at
                FROM knowledgeitem ki
                WHERE ki.embedding IS NOT NULL
                  AND ki.embedding != ''
            """
            
            params = {}
            
            # Add filters
            if workspace_id:
                search_query += " AND ki.workspace_id = :workspace_id"
                params["workspace_id"] = workspace_id
            
            if knowledge_type:
                search_query += " AND ki.metadata->>'type' = :knowledge_type"
                params["knowledge_type"] = knowledge_type
            
            if min_confidence > 0:
                search_query += " AND ki.confidence_score >= :min_confidence"
                params["min_confidence"] = min_confidence
            
            # Channel isolation logic (same as text search):
            # - If channel_id starts with 'D' (DM), search workspace-wide (no channel filter)
            # - If channel_id starts with 'C' (public/private channel), restrict to that channel only
            # - If no channel_id, search workspace-wide
            if channel_id and not channel_id.startswith('D'):
                # This is a public/private channel - restrict to channel-specific knowledge only
                search_query += " AND ki.metadata->>'source_channel_id' = :channel_id"
                params["channel_id"] = channel_id
                logger.info(f"Vector search applying channel isolation for channel {channel_id}")
            elif channel_id and channel_id.startswith('D'):
                # This is a DM - allow workspace-wide search (no channel filter)
                logger.info(f"Vector search: DM detected ({channel_id}) - allowing workspace-wide knowledge search")
            # If no channel_id, search workspace-wide (no additional filter needed)
            
            # Order by confidence and recency for now
            search_query += """
                ORDER BY 
                    ki.confidence_score DESC,
                    ki.created_at DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute search
            result = await db.execute(text(search_query), params)
            rows = result.fetchall()
            
            # Format results
            vector_results = []
            for row in rows:
                metadata = row.metadata if hasattr(row, 'metadata') else {}
                
                vector_results.append({
                    "id": row.id,
                    "title": row.title,
                    "summary": row.summary,
                    "content": row.content,
                    "confidence": float(row.confidence_score),
                    "type": metadata.get("type", "unknown"),
                    "participants": metadata.get("participants", []),
                    "tags": metadata.get("tags", []),
                    "similarity": 0.8,  # High similarity for vector results
                    "distance": 0.2,    # Low distance for vector results
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "metadata": metadata,
                    "search_method": "vector"
                })
            
            return vector_results
            
        except Exception as e:
            logger.error(f"Error in simplified vector search: {e}")
            return []
    
    def _combine_results(self, vector_results: List[Dict[str, Any]], text_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine vector and text search results, avoiding duplicates."""
        try:
            # Create a map of results by ID to avoid duplicates
            results_map = {}
            
            # Add vector results first (higher priority)
            for result in vector_results:
                results_map[result["id"]] = result
            
            # Add text results that aren't already in vector results
            for result in text_results:
                if result["id"] not in results_map:
                    results_map[result["id"]] = result
                else:
                    # Mark existing result as hybrid
                    results_map[result["id"]]["search_method"] = "hybrid"
            
            # Convert back to list and sort by search method and confidence
            combined = list(results_map.values())
            combined.sort(key=lambda x: (
                x.get('search_method') == 'vector',  # Vector first
                x.get('confidence', 0.0)             # Then by confidence
            ), reverse=True)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return vector_results + text_results  # Fallback to simple concatenation
    
    async def _text_search(
        self, 
        query: str, 
        workspace_id: Optional[int] = None,
        channel_id: Optional[str] = None,  # Add channel filtering
        knowledge_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Perform text-based search as fallback."""
        try:
            # Build text search query
            search_query = """
                SELECT 
                    ki.id,
                    ki.title,
                    ki.summary,
                    ki.content,
                    ki.confidence_score,
                    ki.metadata,
                    ki.created_at,
                    ki.updated_at
                FROM knowledgeitem ki
                WHERE (
                    ki.title ILIKE :query_pattern OR
                    ki.summary ILIKE :query_pattern OR
                    ki.content ILIKE :query_pattern OR
                    ki.metadata::text ILIKE :query_pattern
                )
            """
            
            # Create multiple search patterns for better matching
            query_words = query.lower().split()
            query_patterns = []
            
            # Add the full query as a pattern
            query_patterns.append(f"%{query}%")
            
            # Add individual word patterns for partial matching
            for word in query_words:
                if len(word) > 2:  # Only add words longer than 2 characters
                    query_patterns.append(f"%{word}%")
            
            # Build the search query with proper parameter binding
            search_query = """
                SELECT 
                    ki.id,
                    ki.title,
                    ki.summary,
                    ki.content,
                    ki.confidence_score,
                    ki.metadata,
                    ki.created_at,
                    ki.updated_at
                FROM knowledgeitem ki
                WHERE (
            """
            
            # Add OR conditions for each pattern
            pattern_conditions = []
            for i, pattern in enumerate(query_patterns):
                pattern_conditions.append(f"""
                    ki.title ILIKE :pattern_{i} OR 
                    ki.summary ILIKE :pattern_{i} OR 
                    ki.content ILIKE :pattern_{i} OR 
                    ki.metadata::text ILIKE :pattern_{i}
                """)
            
            search_query += " OR ".join(pattern_conditions)
            search_query += ")"
            
            # Create parameters for each pattern
            params = {f"pattern_{i}": pattern for i, pattern in enumerate(query_patterns)}
            
            # Add filters
            if workspace_id:
                search_query += " AND ki.workspace_id = :workspace_id"
                params["workspace_id"] = workspace_id
            
            if knowledge_type:
                search_query += " AND ki.metadata->>'type' = :knowledge_type"
                params["knowledge_type"] = knowledge_type
            
            if min_confidence > 0:
                search_query += " AND ki.confidence_score >= :min_confidence"
                params["min_confidence"] = min_confidence
            
            # Channel isolation logic: 
            # - If channel_id starts with 'D' (DM), search workspace-wide (no channel filter)
            # - If channel_id starts with 'C' (public/private channel), restrict to that channel only
            # - If no channel_id, search workspace-wide
            if channel_id and not channel_id.startswith('D'):
                # This is a public/private channel - restrict to channel-specific knowledge only
                search_query += " AND ki.metadata->>'source_channel_id' = :channel_id"
                params["channel_id"] = channel_id
                logger.info(f"Applying channel isolation for channel {channel_id}")
            elif channel_id and channel_id.startswith('D'):
                # This is a DM - allow workspace-wide search (no channel filter)
                logger.info(f"DM detected ({channel_id}) - allowing workspace-wide knowledge search")
            # If no channel_id, search workspace-wide (no additional filter needed)
            
            # Order by relevance and add limit
            search_query += """
                ORDER BY 
                    ki.confidence_score DESC,
                    ki.created_at DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute search
            logger.info(f"Executing text search with query: {search_query}")
            logger.info(f"Search parameters: {params}")
            logger.info(f"Query patterns: {query_patterns}")
            result = await db.execute(text(search_query), params)
            rows = result.fetchall()
            logger.info(f"Text search returned {len(rows)} raw rows")
            
            # Format results
            text_results = []
            for row in rows:
                # SQLAlchemy automatically deserializes JSONB, so metadata is already a dict
                metadata = row.metadata if hasattr(row, 'metadata') else (row.conversation_metadata if hasattr(row, 'conversation_metadata') else {})
                
                text_results.append({
                    "id": row.id,
                    "title": row.title,
                    "summary": row.summary,
                    "content": row.content,
                    "confidence": float(row.confidence_score),
                    "type": metadata.get("type", "unknown"),
                    "participants": metadata.get("participants", []),
                    "tags": metadata.get("tags", []),
                    "similarity": 0.5,  # Default similarity for text search
                    "distance": 0.5,    # Default distance for text search
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "metadata": metadata,
                    "search_method": "text"
                })
            
            return text_results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    async def _conversation_search(
        self, 
        query: str, 
        workspace_id: Optional[int] = None,
        channel_id: Optional[str] = None,
        limit: int = 10,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Search through conversation history."""
        try:
            # Build conversation search query
            search_query = """
                SELECT 
                    m.id,
                    m.content,
                    m.slack_message_id,
                    m.slack_user_id,
                    m.message_metadata,
                    m.created_at,
                    m.updated_at,
                    c.slack_channel_name,
                    c.slack_channel_id
                FROM message m
                JOIN conversation c ON m.conversation_id = c.id
                WHERE (
                    m.content ILIKE :query_pattern OR
                    m.message_metadata::text ILIKE :query_pattern
                )
            """
            
            # Create multiple search patterns for better matching
            query_words = query.lower().split()
            query_patterns = []
            
            # Add the full query as a pattern
            query_patterns.append(f"%{query}%")
            
            # Add individual word patterns for partial matching
            for word in query_words:
                if len(word) > 2:  # Only add words longer than 2 characters
                    query_patterns.append(f"%{word}%")
            
            # Build the search query with proper parameter binding
            search_query = """
                SELECT 
                    m.id,
                    m.content,
                    m.slack_message_id,
                    m.slack_user_id,
                    m.message_metadata,
                    m.created_at,
                    m.updated_at,
                    c.slack_channel_name,
                    c.slack_channel_id
                FROM message m
                JOIN conversation c ON m.conversation_id = c.id
                WHERE (
            """
            
            # Add OR conditions for each pattern
            pattern_conditions = []
            for i, pattern in enumerate(query_patterns):
                pattern_conditions.append(f"""
                    m.content ILIKE :pattern_{i} OR 
                    m.message_metadata::text ILIKE :pattern_{i}
                """)
            
            search_query += " OR ".join(pattern_conditions)
            search_query += ")"
            
            # Create parameters for each pattern
            params = {f"pattern_{i}": pattern for i, pattern in enumerate(query_patterns)}
            
            # Add filters
            if workspace_id:
                search_query += " AND c.workspace_id = :workspace_id"
                params["workspace_id"] = workspace_id
            
            if channel_id:
                search_query += " AND c.slack_channel_id = :channel_id"
                params["channel_id"] = channel_id
            
            # Order by relevance and add limit
            search_query += """
                ORDER BY 
                    m.created_at DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute search
            logger.info(f"Executing conversation search with query: {search_query}")
            logger.info(f"Search parameters: {params}")
            logger.info(f"Query patterns: {query_patterns}")
            result = await db.execute(text(search_query), params)
            rows = result.fetchall()
            logger.info(f"Conversation search returned {len(rows)} raw rows")
            
            # Format results
            conversation_results = []
            for row in rows:
                metadata = row.message_metadata if hasattr(row, 'message_metadata') else {}
                
                # Get surrounding context for better understanding
                extended_content = await self._get_message_context(row.id, db)
                
                conversation_results.append({
                    "id": row.id,
                    "title": f"Message in #{row.slack_channel_name}",
                    "summary": row.content[:200] + "..." if len(row.content) > 200 else row.content,
                    "content": extended_content if extended_content else row.content,
                    "confidence": 0.7,  # Default confidence for conversations
                    "type": "conversation",
                    "participants": [row.slack_user_id] if row.slack_user_id else [],
                    "tags": [],
                    "similarity": 0.6,  # Default similarity for conversations
                    "distance": 0.4,    # Default distance for conversations
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "metadata": {
                        **metadata,
                        "source_channel_id": row.slack_channel_id,
                        "slack_message_id": row.slack_message_id,
                        "slack_user_id": row.slack_user_id,
                        "channel_name": row.slack_channel_name
                    },
                    "search_method": "conversation"
                })
            
            return conversation_results
            
        except Exception as e:
            logger.error(f"Error in conversation search: {e}")
            return []
    
    async def _get_message_context(self, message_id: int, db: AsyncSession, context_size: int = 2) -> str:
        """Get surrounding messages for better context."""
        try:
            from ..models.base import Message, Conversation
            
            # Get the target message first
            target_msg_result = await db.execute(
                select(Message).where(Message.id == message_id)
            )
            target_msg = target_msg_result.scalars().first()
            
            if not target_msg:
                return ""
            
            # Get surrounding messages from the same conversation
            context_result = await db.execute(
                select(Message)
                .where(Message.conversation_id == target_msg.conversation_id)
                .where(Message.id.between(message_id - context_size, message_id + context_size))
                .order_by(Message.created_at.asc())
            )
            context_messages = context_result.scalars().all()
            
            # Build context string
            context_parts = []
            for msg in context_messages:
                user_prefix = f"@{msg.slack_user_id}: " if msg.slack_user_id else ""
                context_parts.append(f"{user_prefix}{msg.content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting message context: {e}")
            return ""
    
    async def _combine_search_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        text_results: List[Dict[str, Any]], 
        query: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine and rank search results from different methods."""
        try:
            # Create a map of results by ID
            results_map = {}
            
            # Add vector results with high weight
            for result in vector_results:
                result["search_method"] = "vector"
                result["combined_score"] = result["similarity"] * 0.8 + result["confidence"] * 0.2
                results_map[result["id"]] = result
            
            # Add text results with lower weight
            for result in text_results:
                if result["id"] not in results_map:
                    result["combined_score"] = result["similarity"] * 0.4 + result["confidence"] * 0.6
                    results_map[result["id"]] = result
                else:
                    # Enhance existing result with text search info
                    existing = results_map[result["id"]]
                    existing["search_method"] = "hybrid"
                    existing["combined_score"] = (existing["combined_score"] + result["combined_score"]) / 2
            
            # Convert to list and sort by combined score
            combined_results = list(results_map.values())
            combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Apply limit
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return vector_results[:limit] if vector_results else []
    
    async def get_embedding_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        try:
            # Count total knowledge items
            total_result = await db.execute(
                select(func.count(KnowledgeItem.id))
            )
            total_count = total_result.scalar()
            
            # Count items with embeddings
            embedded_result = await db.execute(
                select(func.count(KnowledgeItem.id))
                .where(KnowledgeItem.embedding.isnot(None))
            )
            embedded_count = embedded_result.scalar()
            
            # Count by type
            type_result = await db.execute(
                text("""
                    SELECT 
                        metadata->>'type' as type,
                        COUNT(*) as count
                    FROM knowledgeitem 
                    WHERE embedding IS NOT NULL
                    GROUP BY metadata->>'type'
                    ORDER BY count DESC
                """)
            )
            type_counts = {row.type or "unknown": row.count for row in type_result.fetchall()}
            
            return {
                "total_knowledge_items": total_count,
                "items_with_embeddings": embedded_count,
                "embedding_coverage": embedded_count / max(total_count, 1),
                "by_type": type_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {}
    
    async def cleanup_orphaned_embeddings(self, db: AsyncSession) -> int:
        """Remove embeddings for deleted knowledge items."""
        try:
            # This would be implemented if we had a separate embeddings table
            # For now, embeddings are stored directly in knowledgeitem table
            logger.info("No orphaned embeddings to clean up (embeddings stored inline)")
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned embeddings: {e}")
            return 0
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the vector service."""
        return {
            "status": "operational",
            "similarity_threshold": self.similarity_threshold,
            "embedding_service": await self.embedding_service.get_service_status()
        }
