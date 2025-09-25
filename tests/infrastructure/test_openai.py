"""Mock OpenAI service for testing."""

import json
import hashlib
from typing import Dict, Any, List, Optional
from loguru import logger


class MockOpenAIService:
    """Mock OpenAI service for testing."""
    
    def __init__(self, response_mapping: Optional[Dict[str, str]] = None):
        self.response_mapping = response_mapping or self._get_default_responses()
        self.call_history = []
    
    def _get_default_responses(self) -> Dict[str, str]:
        """Get default mock responses for common queries."""
        return {
            "kafka_restart": "To restart Kafka connector: docker restart kafka-connector",
            "database_migration": "Database migration steps: 1. Backup data 2. Run migration 3. Verify",
            "deployment_process": "Deployment process: 1. Build image 2. Test 3. Deploy to staging 4. Deploy to prod",
            "error_troubleshooting": "Check logs first: docker logs <service-name>",
            "default": "I found some information about your query. Please check the sources for details."
        }
    
    def _get_response_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a key for response lookup based on messages."""
        # Extract the user message content
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break
        
        # Simple keyword matching for response selection
        if "kafka" in user_message and "restart" in user_message:
            return "kafka_restart"
        elif "migration" in user_message and "database" in user_message:
            return "database_migration"
        elif "deploy" in user_message:
            return "deployment_process"
        elif "error" in user_message or "troubleshoot" in user_message:
            return "error_troubleshooting"
        else:
            return "default"
    
    async def _make_request(self, **kwargs) -> Dict[str, Any]:
        """Mock OpenAI API request."""
        try:
            # Log the request
            self.call_history.append(kwargs)
            
            # Extract messages
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "gpt-3.5-turbo")
            max_tokens = kwargs.get("max_tokens", 500)
            temperature = kwargs.get("temperature", 0.1)
            
            # Get response based on messages
            response_key = self._get_response_key(messages)
            response_text = self.response_mapping.get(response_key, self.response_mapping["default"])
            
            # Create mock response structure
            mock_response = {
                "id": f"chatcmpl-{hashlib.md5(str(messages).encode()).hexdigest()[:8]}",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(messages)),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(str(messages)) + len(response_text.split())
                }
            }
            
            logger.info(f"Mock OpenAI response generated for key: {response_key}")
            return mock_response
            
        except Exception as e:
            logger.error(f"Mock OpenAI request failed: {e}")
            raise
    
    def add_custom_response(self, key: str, response: str):
        """Add custom response for specific queries."""
        self.response_mapping[key] = response
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of all API calls made."""
        return self.call_history
    
    def clear_call_history(self):
        """Clear API call history."""
        self.call_history = []


class MockEmbeddingService:
    """Mock embedding service for testing."""
    
    def __init__(self):
        self.embedding_cache = {}
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for text."""
        # Use simple hash-based embedding for consistency
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Generate mock 1536-dimensional embedding
        embedding = [float((ord(c) - 48) % 10) / 10.0 for c in text_hash[:1536]]
        if len(embedding) < 1536:
            embedding.extend([0.0] * (1536 - len(embedding)))
        
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache = {}


# Global mock services
mock_openai_service = MockOpenAIService()
mock_embedding_service = MockEmbeddingService()
