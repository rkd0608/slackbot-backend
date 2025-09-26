"""Embedding service for generating and managing vector embeddings for knowledge search."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import openai
from loguru import logger
from openai import AsyncOpenAI

from ..core.config import settings


class EmbeddingService:
    """Service for generating and managing vector embeddings."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = "text-embedding-3-small"  # Fast and cost-effective
        self.max_batch_size = 100  # OpenAI batch limit
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Embedding dimensions for different models
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        logger.info(f"Embedding service initialized with model: {self.embedding_model}")
    
    async def generate_embedding(
        self, 
        text: str, 
        content_type: str = "knowledge"
    ) -> Optional[List[float]]:
        """Generate a single embedding for text content."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding generation")
                return None
            
            # Clean and prepare text
            cleaned_text = self._prepare_text_for_embedding(text, content_type)
            
            # Generate embedding
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text
            )
            
            embedding = response.data[0].embedding
            
            logger.debug(f"Generated embedding for {content_type}: {len(embedding)} dimensions")
            return embedding
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            await asyncio.sleep(1)  # Wait before retry
            raise
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        content_types: Optional[List[str]] = None
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches."""
        try:
            if not texts:
                return []
            
            if content_types is None:
                content_types = ["knowledge"] * len(texts)
            
            if len(texts) != len(content_types):
                raise ValueError("Texts and content_types must have the same length")
            
            # Clean and prepare texts
            prepared_texts = []
            for text, content_type in zip(texts, content_types):
                cleaned_text = self._prepare_text_for_embedding(text, content_type)
                if cleaned_text:
                    prepared_texts.append(cleaned_text)
                else:
                    prepared_texts.append("")  # Keep alignment
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(prepared_texts), self.max_batch_size):
                batch_texts = prepared_texts[i:i + self.max_batch_size]
                
                # Filter out empty texts
                valid_batch = [(j, text) for j, text in enumerate(batch_texts) if text.strip()]
                
                if not valid_batch:
                    # Add None embeddings for empty texts
                    all_embeddings.extend([None] * len(batch_texts))
                    continue
                
                # Generate embeddings for valid batch
                try:
                    response = await self.client.embeddings.create(
                        model=self.embedding_model,
                        input=[text for _, text in valid_batch]
                    )
                    
                    # Map embeddings back to original positions
                    batch_embeddings = [None] * len(batch_texts)
                    for (orig_idx, _), embedding_data in zip(valid_batch, response.data):
                        batch_embeddings[orig_idx] = embedding_data.embedding
                    
                    all_embeddings.extend(batch_embeddings)
                    
                    # Rate limiting
                    if i + self.max_batch_size < len(prepared_texts):
                        await asyncio.sleep(self.rate_limit_delay)
                        
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit exceeded in batch {i//self.max_batch_size}: {e}")
                    await asyncio.sleep(2)  # Wait longer for rate limits
                    # Add None embeddings for failed batch
                    all_embeddings.extend([None] * len(batch_texts))
                    
                except Exception as e:
                    logger.error(f"Error in batch {i//self.max_batch_size}: {e}")
                    # Add None embeddings for failed batch
                    all_embeddings.extend([None] * len(batch_texts))
            
            logger.info(f"Generated {len([e for e in all_embeddings if e is not None])} embeddings from {len(texts)} texts")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            raise
    
    def _prepare_text_for_embedding(self, text: str, content_type: str) -> str:
        """Prepare text for optimal embedding generation."""
        if not text:
            return ""
        
        # Clean text
        cleaned = text.strip()
        
        # Truncate if too long (OpenAI has limits)
        max_length = 8000  # Conservative limit
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        # Add content type prefix for better semantic understanding
        if content_type == "knowledge":
            cleaned = f"Knowledge: {cleaned}"
        elif content_type == "question":
            cleaned = f"Question: {cleaned}"
        elif content_type == "conversation":
            cleaned = f"Conversation: {cleaned}"
        
        return cleaned
    
    async def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in valid range
            return max(-1.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find the most similar embeddings to a query embedding."""
        try:
            if not query_embedding or not candidate_embeddings:
                return []
            
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                if candidate:
                    similarity = await self.calculate_similarity(query_embedding, candidate)
                    similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {e}")
            return []
    
    async def generate_hybrid_embedding(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Generate embedding that combines text content with metadata."""
        try:
            # Create enhanced text with metadata
            enhanced_text = self._create_enhanced_text(text, metadata)
            
            # Generate embedding for enhanced text
            return await self.generate_embedding(enhanced_text, "hybrid")
            
        except Exception as e:
            logger.error(f"Error generating hybrid embedding: {e}")
            return None
    
    def _create_enhanced_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """Create enhanced text that includes metadata context."""
        enhanced_parts = [text]
        
        # Add metadata context
        if metadata.get("type"):
            enhanced_parts.append(f"Type: {metadata['type']}")
        
        if metadata.get("tags"):
            tags_text = ", ".join(metadata["tags"])
            enhanced_parts.append(f"Tags: {tags_text}")
        
        if metadata.get("participants"):
            participants_text = ", ".join(metadata["participants"])
            enhanced_parts.append(f"Participants: {participants_text}")
        
        # Combine all parts
        enhanced_text = " | ".join(enhanced_parts)
        
        return enhanced_text
    
    async def get_embedding_dimensions(self) -> int:
        """Get the dimensions of the current embedding model."""
        return self.model_dimensions.get(self.embedding_model, 1536)
    
    async def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding is properly formatted."""
        try:
            if not embedding:
                return False
            
            expected_dimensions = await self.get_embedding_dimensions()
            
            # Check dimensions
            if len(embedding) != expected_dimensions:
                logger.warning(f"Embedding dimensions mismatch: expected {expected_dimensions}, got {len(embedding)}")
                return False
            
            # Check for NaN or infinite values
            if any(not np.isfinite(val) for val in embedding):
                logger.warning("Embedding contains NaN or infinite values")
                return False
            
            # Check for all-zero embeddings
            if all(val == 0.0 for val in embedding):
                logger.warning("Embedding is all zeros")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating embedding: {e}")
            return False
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the embedding service."""
        return {
            "status": "operational",
            "model": self.embedding_model,
            "dimensions": await self.get_embedding_dimensions(),
            "max_batch_size": self.max_batch_size,
            "rate_limit_delay": self.rate_limit_delay
        }
