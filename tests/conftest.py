"""Pytest configuration and fixtures."""

import pytest
import asyncio
import os
from typing import AsyncGenerator, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

# Import test infrastructure
from tests.infrastructure.test_database import test_db_manager
from tests.infrastructure.test_redis import test_redis_manager
from tests.infrastructure.test_openai import mock_openai_service, mock_embedding_service
from tests.infrastructure.test_slack import mock_slack_service

# Import fixtures
from tests.fixtures.conversations import ConversationFixtures
from tests.fixtures.queries import QueryFixtures
from tests.fixtures.ground_truth import GroundTruthFixtures


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_database():
    """Session-scoped test database."""
    await test_db_manager.setup_test_db()
    yield test_db_manager
    await test_db_manager.cleanup_test_db()


@pytest.fixture(scope="session")
def test_redis():
    """Session-scoped test Redis."""
    test_redis_manager.setup_test_redis()
    yield test_redis_manager
    test_redis_manager.cleanup_test_redis()


@pytest.fixture(scope="function")
async def isolated_test_env(test_database, test_redis):
    """Function-scoped isolated test environment."""
    # Clear any existing data
    test_redis.get_client().flushdb()
    
    # Seed with basic test data
    test_data = {
        "messages": ConversationFixtures.get_technical_conversation()["messages"],
        "knowledge_items": [ConversationFixtures.get_technical_conversation()["expected_knowledge"]]
    }
    
    await test_database.seed_test_data(test_data)
    
    yield {
        "database": test_database,
        "redis": test_redis,
        "openai": mock_openai_service,
        "slack": mock_slack_service
    }
    
    # Cleanup after each test
    test_redis.get_client().flushdb()
    mock_slack_service.clear_messages()


@pytest.fixture
async def db_session(test_database) -> AsyncGenerator[AsyncSession, None]:
    """Get a database session for testing."""
    async with test_database.get_session() as session:
        yield session


@pytest.fixture
def conversation_fixtures():
    """Conversation test fixtures."""
    return ConversationFixtures


@pytest.fixture
def query_fixtures():
    """Query test fixtures."""
    return QueryFixtures


@pytest.fixture
def ground_truth_fixtures():
    """Ground truth test fixtures."""
    return GroundTruthFixtures


@pytest.fixture
def mock_services():
    """Mock services for testing."""
    return {
        "openai": mock_openai_service,
        "slack": mock_slack_service,
        "embedding": mock_embedding_service
    }


@pytest.fixture
def test_workspace_data():
    """Test workspace data."""
    return {
        "name": "Test Workspace",
        "slack_id": "T1234567890",
        "tokens": {
            "access_token": "test_access_token",
            "bot_token": "test_bot_token"
        }
    }


@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "slack_id": "U1234567890",
        "name": "Test User",
        "role": "admin"
    }


@pytest.fixture
def test_channel_data():
    """Test channel data."""
    return {
        "id": "C1234567890",
        "name": "test-channel",
        "is_channel": True,
        "is_private": False
    }


@pytest.fixture
def quality_thresholds():
    """Quality thresholds for testing."""
    return GroundTruthFixtures.get_quality_thresholds()


@pytest.fixture
def evaluation_criteria():
    """Evaluation criteria for testing."""
    return GroundTruthFixtures.get_evaluation_criteria()


# Pytest markers
pytestmark = [
    pytest.mark.asyncio,
]


# --- Reporting hooks ---
def pytest_sessionfinish(session, exitstatus):
    """After the entire test session, export quality reports if any metrics exist."""
    try:
        import os
        from datetime import datetime
        from pathlib import Path
        from tests.monitoring.reporting import export_json, export_html
        # Only export if tests touched the quality dashboard (heuristic: always export)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        export_json(str(out_dir / f"quality_{ts}.json"))
        export_html(str(out_dir / f"quality_{ts}.html"))
    except Exception:
        # Do not fail test session on report export issues
        pass
