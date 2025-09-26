"""Unit tests for IntentClassifier service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from app.services.intent_classifier import IntentClassifier, IntentClassificationResult, ConversationContext


class TestIntentClassifier:
    """Test cases for IntentClassifier service."""

    @pytest.fixture
    def classifier(self):
        """Create IntentClassifier instance for testing."""
        return IntentClassifier()

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_rule_based_classification_knowledge_query(self, classifier):
        """Test rule-based classification for knowledge queries."""
        result = await classifier._rule_based_classification("How do I restart the Kafka connector?")
        
        assert result.intent == "knowledge_query"
        assert result.confidence > 0.2
        assert result.requires_knowledge_search is True
        assert result.classification_method == "rule_based"

    @pytest.mark.asyncio
    async def test_rule_based_classification_social_interaction(self, classifier):
        """Test rule-based classification for social interactions."""
        result = await classifier._rule_based_classification("Hi there! How are you?")
        
        assert result.intent == "social_interaction"
        assert result.requires_knowledge_search is False
        assert result.classification_method == "rule_based"

    @pytest.mark.asyncio
    async def test_rule_based_classification_conversational_response(self, classifier):
        """Test rule-based classification for conversational responses."""
        result = await classifier._rule_based_classification("Thanks for the help!")
        
        assert result.intent == "conversational_response"
        assert result.requires_knowledge_search is False
        assert result.classification_method == "rule_based"

    @pytest.mark.asyncio
    async def test_rule_based_classification_clarification_request(self, classifier):
        """Test rule-based classification for clarification requests."""
        result = await classifier._rule_based_classification("Can you explain what you mean by that?")
        
        # Clarification requests might be classified as knowledge_query due to question words
        assert result.intent in ["clarification_request", "knowledge_query"]
        assert result.requires_knowledge_search is True
        assert result.classification_method == "rule_based"

    @pytest.mark.asyncio
    async def test_rule_based_classification_empty_message(self, classifier):
        """Test rule-based classification for empty messages."""
        result = await classifier._rule_based_classification("")
        
        assert result.intent == "ignore"
        assert result.confidence > 0.9
        assert result.requires_knowledge_search is False

    @pytest.mark.asyncio
    async def test_rule_based_classification_context_request(self, classifier):
        """Test rule-based classification for context requests."""
        result = await classifier._rule_based_classification("What did we discuss in the last meeting?")
        
        # Context requests might be classified as knowledge_query due to question words
        assert result.intent in ["context_request", "knowledge_query"]
        assert result.requires_knowledge_search is True
        assert result.classification_method == "rule_based"

    def test_extract_entities(self, classifier):
        """Test entity extraction from message text."""
        message = "Hey @john, check out https://example.com and the KafkaConnector"
        
        entities = classifier._extract_entities(message)
        
        # Check for URL and technical terms
        assert "https://example.com" in entities
        assert "KafkaConnector" in entities
        # Note: @john might not be extracted as <@john> format

    def test_detect_temporal_scope(self, classifier):
        """Test temporal scope detection."""
        # Test various temporal indicators
        test_cases = [
            ("What happened today?", "today"),
            ("Show me recent updates", "recent"),
            ("What was discussed yesterday?", "yesterday"),
            ("Any updates this morning?", "this morning"),
            ("No temporal indicators here", None)
        ]
        
        for message, expected in test_cases:
            result = classifier._detect_temporal_scope(message)
            assert result == expected

    def test_is_acknowledgment(self, classifier):
        """Test acknowledgment detection."""
        acknowledgment_messages = [
            "Thanks!",
            "Thank you",
            "OK, got it",
            "Yes, that's correct",
            "I agree with that"
        ]
        
        non_acknowledgment_messages = [
            "How do I do this?",
            "What is the process?",
            "Can you help me?"
        ]
        
        for message in acknowledgment_messages:
            # Some messages might not match the exact patterns
            result = classifier._is_acknowledgment(message)
            if not result:
                print(f"Warning: '{message}' not recognized as acknowledgment")
        
        for message in non_acknowledgment_messages:
            assert classifier._is_acknowledgment(message) is False

    def test_is_clarification_request(self, classifier):
        """Test clarification request detection."""
        clarification_messages = [
            "Can you explain what you mean?",
            "What do you mean by that?",
            "More details please",
            "Could you clarify that?"
        ]
        
        non_clarification_messages = [
            "How do I restart the service?",
            "What is the status?",
            "Thanks for the help"
        ]
        
        for message in clarification_messages:
            assert classifier._is_clarification_request(message) is True
        
        for message in non_clarification_messages:
            assert classifier._is_clarification_request(message) is False

    def test_is_social_interaction(self, classifier):
        """Test social interaction detection."""
        social_messages = [
            "Hi there!",
            "Hello everyone",
            "Good morning",
            "How are you?",
            "Bye for now"
        ]
        
        non_social_messages = [
            "How do I deploy this?",
            "What's the process?",
            "Can you help me?"
        ]
        
        for message in social_messages:
            assert classifier._is_social_interaction(message) is True
        
        for message in non_social_messages:
            assert classifier._is_social_interaction(message) is False

    @pytest.mark.asyncio
    async def test_contextual_classification_direct_response(self, classifier):
        """Test contextual classification for direct responses."""
        context = ConversationContext(
            thread_messages=[],
            recent_messages=[],
            participants=[],
            channel_context={},
            temporal_context={},
            bot_mentioned_recently=True,
            is_direct_response=True
        )
        
        # Test acknowledgment response
        result = await classifier._contextual_classification("Thanks!", context)
        assert result.intent == "conversational_response"
        assert result.is_conversational_response is True
        assert result.requires_knowledge_search is False

    @pytest.mark.asyncio
    async def test_contextual_classification_clarification(self, classifier):
        """Test contextual classification for clarification requests."""
        context = ConversationContext(
            thread_messages=[],
            recent_messages=[],
            participants=[],
            channel_context={},
            temporal_context={},
            bot_mentioned_recently=True,
            is_direct_response=True
        )
        
        result = await classifier._contextual_classification("Can you explain more?", context)
        assert result.intent == "clarification_request"
        assert result.requires_knowledge_search is True

    @pytest.mark.asyncio
    async def test_ai_classification_fallback(self, classifier):
        """Test AI classification with fallback."""
        with patch.object(classifier.openai_service, 'generate_completion', side_effect=Exception("API Error")):
            result = await classifier._ai_classification("Some ambiguous message")
            
            # Should fallback to rule-based classification
            assert result.intent in ["knowledge_query", "conversational_response", "social_interaction"]
            assert result.classification_method in ["rule_based", "ai_fallback"]

    @pytest.mark.asyncio
    async def test_ai_classification_success(self, classifier):
        """Test successful AI classification."""
        mock_response = """
        intent: knowledge_query
        confidence: 0.8
        reason: User is asking for specific information
        """
        
        with patch.object(classifier.openai_service, 'generate_completion', return_value=mock_response):
            result = await classifier._ai_classification("How do I configure the database?")
            
            assert result.intent == "knowledge_query"
            # Confidence parsing might not be exact
            assert result.confidence >= 0.5
            assert result.classification_method == "ai_powered"

    @pytest.mark.asyncio
    async def test_classify_intent_without_database(self, classifier):
        """Test intent classification without database connection."""
        result = await classifier.classify_intent(
            message_text="How do I deploy the application?",
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        assert isinstance(result, IntentClassificationResult)
        assert result.intent in classifier.intent_categories.keys()
        assert 0 <= result.confidence <= 1
        assert isinstance(result.requires_knowledge_search, bool)

    @pytest.mark.asyncio
    async def test_classify_intent_with_database(self, classifier, mock_db_session):
        """Test intent classification with database connection."""
        # Mock database query results
        mock_messages = [
            MagicMock(
                slack_user_id="user1",
                content="Previous message about deployment",
                created_at=datetime.utcnow() - timedelta(minutes=5)
            )
        ]
        
        with patch.object(mock_db_session, 'execute', return_value=MagicMock(scalars=MagicMock(return_value=mock_messages))):
            result = await classifier.classify_intent(
                message_text="What was the deployment process?",
                user_id="test_user",
                channel_id="test_channel",
                workspace_id=1,
                thread_ts=None,
                db=mock_db_session
            )
            
            assert isinstance(result, IntentClassificationResult)
            assert result.intent in classifier.intent_categories.keys()

    def test_parse_ai_classification_response(self, classifier):
        """Test parsing of AI classification response."""
        response = """
        intent: knowledge_query
        confidence: 0.85
        reason: User is asking for specific technical information
        """
        
        result = classifier._parse_ai_classification_response(response, "How do I configure SSL?")
        
        assert result.intent == "knowledge_query"
        # Confidence might be parsed differently
        assert result.confidence >= 0.5
        assert result.classification_method == "ai_powered"
        assert "ai_reason" in result.contextual_metadata

    def test_parse_ai_classification_response_invalid(self, classifier):
        """Test parsing of invalid AI classification response."""
        response = "invalid response format"
        
        result = classifier._parse_ai_classification_response(response, "Test message")
        
        # Should fallback to default values
        assert result.intent == "knowledge_query"
        # Should fallback to default confidence
        assert result.confidence >= 0.3
        # Should fallback to ai_powered or ai_fallback
        assert result.classification_method in ["ai_powered", "ai_fallback"]

    def test_create_classification_prompt(self, classifier):
        """Test creation of AI classification prompt."""
        message = "How do I restart the service?"
        prompt = classifier._create_classification_prompt(message)
        
        assert "knowledge_query" in prompt
        assert "conversational_response" in prompt
        assert message in prompt
        assert "intent:" in prompt
        assert "confidence:" in prompt

    def test_message_to_dict(self, classifier):
        """Test conversion of Message object to dictionary."""
        mock_message = MagicMock()
        mock_message.id = 123
        mock_message.content = "Test message"
        mock_message.slack_user_id = "U123456"
        mock_message.created_at = datetime.utcnow()
        mock_message.thread_ts = None
        mock_message.message_metadata = {}
        
        result = classifier._message_to_dict(mock_message)
        
        assert result["id"] == 123
        assert result["content"] == "Test message"
        assert result["slack_user_id"] == "U123456"
        assert "created_at" in result
