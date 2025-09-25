"""Basic tests to verify the testing framework is working."""

import pytest
from tests.infrastructure.test_database import test_db_manager
from tests.infrastructure.test_redis import test_redis_manager
from tests.infrastructure.test_openai import mock_openai_service
from tests.infrastructure.test_slack import mock_slack_service


class TestBasicFramework:
    """Basic tests to verify testing framework functionality."""
    
    def test_imports_work(self):
        """Test that all test modules can be imported."""
        assert test_db_manager is not None
        assert test_redis_manager is not None
        assert mock_openai_service is not None
        assert mock_slack_service is not None
    
    def test_mock_openai_service(self):
        """Test that mock OpenAI service works."""
        # Test that we can create the service
        assert mock_openai_service is not None
        assert hasattr(mock_openai_service, '_make_request')
        assert hasattr(mock_openai_service, 'add_custom_response')
    
    def test_mock_slack_service(self):
        """Test that mock Slack service works."""
        # Test that we can create the service
        assert mock_slack_service is not None
        assert hasattr(mock_slack_service, 'send_message')
        assert hasattr(mock_slack_service, 'get_user_info')
        assert hasattr(mock_slack_service, 'get_channel_info')
    
    def test_fixtures_import(self):
        """Test that fixtures can be imported."""
        from tests.fixtures.conversations import ConversationFixtures
        from tests.fixtures.queries import QueryFixtures
        from tests.fixtures.ground_truth import GroundTruthFixtures
        
        assert ConversationFixtures is not None
        assert QueryFixtures is not None
        assert GroundTruthFixtures is not None
    
    def test_conversation_fixtures(self):
        """Test that conversation fixtures work."""
        from tests.fixtures.conversations import ConversationFixtures
        
        # Test getting a conversation
        conv = ConversationFixtures.get_technical_conversation()
        assert conv is not None
        assert 'messages' in conv
        assert 'expected_knowledge' in conv
        assert len(conv['messages']) > 0
    
    def test_query_fixtures(self):
        """Test that query fixtures work."""
        from tests.fixtures.queries import QueryFixtures
        
        # Test getting queries
        queries = QueryFixtures.get_kafka_queries()
        assert queries is not None
        assert len(queries) > 0
        
        # Test query structure
        query = queries[0]
        assert 'query' in query
        assert 'expected_response' in query
        assert 'expected_sources' in query
        assert 'quality_score' in query
    
    def test_ground_truth_fixtures(self):
        """Test that ground truth fixtures work."""
        from tests.fixtures.ground_truth import GroundTruthFixtures
        
        # Test getting ground truth
        gt = GroundTruthFixtures.get_kafka_ground_truth()
        assert gt is not None
        assert 'conversation_id' in gt
        assert 'knowledge_items' in gt
        assert 'queries' in gt
        
        # Test quality thresholds
        thresholds = GroundTruthFixtures.get_quality_thresholds()
        assert thresholds is not None
        assert 'factual_accuracy' in thresholds
        assert 'completeness' in thresholds
        assert 'overall_quality' in thresholds
    
    @pytest.mark.asyncio
    async def test_mock_openai_request(self):
        """Test that mock OpenAI can handle requests."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I restart Kafka?"}
        ]
        
        response = await mock_openai_service._make_request(
            messages=messages,
            model="gpt-3.5-turbo",
            max_tokens=100
        )
        
        assert response is not None
        assert 'choices' in response
        assert len(response['choices']) > 0
        assert 'message' in response['choices'][0]
        assert 'content' in response['choices'][0]['message']
    
    @pytest.mark.asyncio
    async def test_mock_slack_send_message(self):
        """Test that mock Slack can send messages."""
        # Clear any existing messages
        mock_slack_service.clear_messages()
        
        # Send a test message
        result = await mock_slack_service.send_message(
            channel="C1234567890",
            text="Test message",
            token="test_token"
        )
        
        assert result is not None
        assert result.startswith("1234567890.")
        
        # Verify message was stored
        messages = mock_slack_service.get_sent_messages()
        assert len(messages) == 1
        assert messages[0]['channel'] == "C1234567890"
        assert messages[0]['text'] == "Test message"
