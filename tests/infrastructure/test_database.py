"""Isolated test database management."""

import asyncio
import uuid
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from loguru import logger

from app.core.config import settings


class TestDatabaseManager:
    """Manages isolated test database instances."""
    
    def __init__(self):
        self.test_db_name = f"slackbot_test_{uuid.uuid4().hex[:8]}"
        self.test_db_url = f"postgresql+asyncpg://postgres:admin@localhost:5433/{self.test_db_name}"
        self.sync_db_url = f"postgresql://postgres:admin@localhost:5433/{self.test_db_name}"
        self._engine = None
        self._session_factory = None
    
    async def setup_test_db(self) -> str:
        """Create isolated test database and run migrations."""
        try:
            # Create test database using sync connection
            sync_engine = create_engine(
                "postgresql://postgres:admin@localhost:5433/postgres",
                echo=False
            )
            
            with sync_engine.connect() as conn:
                conn.execute(text("COMMIT"))  # End any existing transaction
                conn.execute(text(f"DROP DATABASE IF EXISTS {self.test_db_name}"))
                conn.execute(text(f"CREATE DATABASE {self.test_db_name}"))
            
            sync_engine.dispose()
            
            # Create async engine for the test database
            self._engine = create_async_engine(
                self.test_db_url,
                echo=False,
                poolclass=None  # Use NullPool for async
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                self._engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # Run migrations
            await self._run_migrations()
            
            logger.info(f"Test database '{self.test_db_name}' created successfully")
            return self.test_db_url
            
        except Exception as e:
            logger.error(f"Failed to setup test database: {e}")
            raise
    
    async def _run_migrations(self):
        """Run database migrations on test database."""
        try:
            # Import and run migrations
            from alembic.config import Config
            from alembic import command
            
            # Create alembic config for test database
            alembic_cfg = Config("alembic.ini")
            alembic_cfg.set_main_option("sqlalchemy.url", self.sync_db_url)
            
            # Run migrations
            command.upgrade(alembic_cfg, "head")
            logger.info("Test database migrations completed")
            
        except Exception as e:
            logger.error(f"Failed to run migrations: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        if not self._session_factory:
            raise RuntimeError("Test database not initialized. Call setup_test_db() first.")
        return self._session_factory()
    
    async def cleanup_test_db(self):
        """Drop test database and clean up connections."""
        try:
            # Close all connections
            if self._engine:
                await self._engine.dispose()
            
            # Drop test database
            sync_engine = create_engine(
                "postgresql://postgres:admin@localhost:5433/postgres",
                echo=False
            )
            
            with sync_engine.connect() as conn:
                conn.execute(text("COMMIT"))
                conn.execute(text(f"DROP DATABASE IF EXISTS {self.test_db_name}"))
            
            sync_engine.dispose()
            logger.info(f"Test database '{self.test_db_name}' cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup test database: {e}")
    
    async def seed_test_data(self, data: Dict[str, Any]):
        """Seed test database with test data."""
        async with self.get_session() as session:
            try:
                # Import models
                from app.models.base import Workspace, User, Conversation, Message, KnowledgeItem
                
                # Create test workspace
                workspace = Workspace(
                    name="Test Workspace",
                    slack_id="T1234567890",
                    tokens={"access_token": "test_token", "bot_token": "test_bot_token"}
                )
                session.add(workspace)
                await session.flush()
                
                # Create test user
                user = User(
                    workspace_id=workspace.id,
                    slack_id="U1234567890",
                    name="Test User",
                    role="admin"
                )
                session.add(user)
                await session.flush()
                
                # Create test conversation
                conversation = Conversation(
                    workspace_id=workspace.id,
                    slack_channel_id="C1234567890",
                    slack_channel_name="test-channel",
                    title="Test Conversation",
                    conversation_metadata={}
                )
                session.add(conversation)
                await session.flush()
                
                # Create test messages
                messages = data.get("messages", [])
                for msg_data in messages:
                    message = Message(
                        conversation_id=conversation.id,
                        slack_message_id=msg_data.get("ts", "1234567890.123456"),
                        slack_user_id=msg_data.get("user", "U1234567890"),
                        content=msg_data.get("text", ""),
                        message_metadata=msg_data
                    )
                    session.add(message)
                
                # Create test knowledge items
                knowledge_items = data.get("knowledge_items", [])
                for ki_data in knowledge_items:
                    knowledge_item = KnowledgeItem(
                        workspace_id=workspace.id,
                        title=ki_data.get("title", "Test Knowledge"),
                        summary=ki_data.get("summary", "Test Summary"),
                        content=ki_data.get("content", "Test Content"),
                        confidence_score=ki_data.get("confidence_score", 0.8),
                        knowledge_type=ki_data.get("knowledge_type", "test"),
                        metadata=ki_data.get("metadata", {}),
                        embedding=ki_data.get("embedding", None)
                    )
                    session.add(knowledge_item)
                
                await session.commit()
                logger.info("Test data seeded successfully")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to seed test data: {e}")
                raise


# Global test database manager instance
test_db_manager = TestDatabaseManager()
