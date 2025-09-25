"""Isolated test Redis management."""

import redis
import asyncio
from typing import Optional, Dict, Any
from loguru import logger


class TestRedisManager:
    """Manages isolated Redis instances for testing."""
    
    def __init__(self):
        self.test_redis_db = 15  # Use different DB number
        self.test_redis_url = f"redis://localhost:6380/{self.test_redis_db}"
        self._redis_client = None
    
    def setup_test_redis(self) -> str:
        """Setup isolated Redis instance for testing."""
        try:
            self._redis_client = redis.Redis(
                host='localhost',
                port=6380,
                db=self.test_redis_db,
                decode_responses=True
            )
            
            # Test connection
            self._redis_client.ping()
            
            # Clear any existing data
            self._redis_client.flushdb()
            
            logger.info(f"Test Redis setup on DB {self.test_redis_db}")
            return self.test_redis_url
            
        except Exception as e:
            logger.error(f"Failed to setup test Redis: {e}")
            raise
    
    def get_client(self) -> redis.Redis:
        """Get Redis client for testing."""
        if not self._redis_client:
            raise RuntimeError("Test Redis not initialized. Call setup_test_redis() first.")
        return self._redis_client
    
    def cleanup_test_redis(self):
        """Clean up test Redis data."""
        try:
            if self._redis_client:
                self._redis_client.flushdb()
                self._redis_client.close()
                logger.info("Test Redis cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup test Redis: {e}")
    
    def seed_test_data(self, data: Dict[str, Any]):
        """Seed Redis with test data."""
        try:
            # Seed Celery task data
            celery_data = data.get("celery_tasks", {})
            for task_id, task_data in celery_data.items():
                self._redis_client.hset(f"celery-task-meta-{task_id}", mapping=task_data)
            
            # Seed cache data
            cache_data = data.get("cache", {})
            for key, value in cache_data.items():
                self._redis_client.set(key, value, ex=3600)  # 1 hour expiry
            
            # Seed rate limiting data
            rate_limit_data = data.get("rate_limits", {})
            for key, value in rate_limit_data.items():
                self._redis_client.set(f"rate_limit:{key}", value, ex=300)  # 5 minutes
            
            logger.info("Test Redis data seeded successfully")
            
        except Exception as e:
            logger.error(f"Failed to seed test Redis data: {e}")
            raise


# Global test Redis manager instance
test_redis_manager = TestRedisManager()
