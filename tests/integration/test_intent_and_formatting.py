"""Integration tests for intent classification and response formatting system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.intent_classifier import IntentClassifier, IntentClassificationResult
from app.services.response_formatter import ResponseFormatter, ContentData


class TestIntentAndFormattingIntegration:
    """Integration tests for the complete intent classification and response formatting system."""

    @pytest.fixture
    def classifier(self):
        """Create IntentClassifier instance."""
        return IntentClassifier()

    @pytest.fixture
    def formatter(self):
        """Create ResponseFormatter instance."""
        return ResponseFormatter()

    @pytest.mark.asyncio
    async def test_knowledge_query_workflow(self, classifier, formatter):
        """Test complete workflow for knowledge query."""
        # Test intent classification
        intent_result = await classifier.classify_intent(
            message_text="How do I restart the Kafka connector?",
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        # Verify intent classification
        assert intent_result.intent == "knowledge_query"
        assert intent_result.requires_knowledge_search is True
        assert intent_result.confidence > 0.5
        
        # Create mock content data
        content_data = ContentData(
            main_content="To restart the Kafka connector, run: `docker restart kafka-connector`",
            sources=[
                {
                    "title": "Kafka Connector Guide",
                    "date": "2024-01-15",
                    "channel": "#devops",
                    "excerpt": "Instructions for managing Kafka connectors"
                }
            ],
            next_steps=["Check connector status", "Monitor logs", "Verify data flow"],
            related_info=["Docker commands", "Kafka configuration"]
        )
        
        # Test response formatting
        response = await formatter.format_response(
            content_data=content_data,
            intent_result=intent_result,
            query_text="How do I restart the Kafka connector?",
            query_id=123
        )
        
        # Verify response structure
        assert response["response_type"] == "in_channel"
        assert "blocks" in response
        assert "text" in response
        assert len(response["blocks"]) > 0
        
        # Verify response contains expected content
        blocks_text = " ".join([block.get("text", {}).get("text", "") for block in response["blocks"] if "text" in block])
        assert "restart" in blocks_text.lower()
        assert "kafka" in blocks_text.lower()

    @pytest.mark.asyncio
    async def test_decision_query_workflow(self, classifier, formatter):
        """Test complete workflow for decision query."""
        # Test intent classification
        intent_result = await classifier.classify_intent(
            message_text="Why did we decide to migrate to PostgreSQL?",
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        # Verify intent classification
        assert intent_result.intent in ["knowledge_query", "decision_rationale"]
        assert intent_result.requires_knowledge_search is True
        
        # Create decision content data
        content_data = ContentData(
            main_content="The team decided to migrate to PostgreSQL for better JSON support and performance.",
            sources=[
                {
                    "title": "Database Migration Decision",
                    "date": "2024-01-10",
                    "channel": "#architecture",
                    "excerpt": "Discussion about database migration options"
                }
            ],
            next_steps=["Create migration plan", "Set up staging environment", "Test data migration"],
            related_info=["Current MySQL setup", "JSON performance requirements"],
            decision_info={
                "outcome": "Migrate to PostgreSQL",
                "decision_maker": "John (Senior Engineer)",
                "date": "January 10, 2024",
                "rationale": "Better JSON support needed for analytics features and improved query performance"
            }
        )
        
        # Test response formatting
        response = await formatter.format_response(
            content_data=content_data,
            intent_result=intent_result,
            query_text="Why did we decide to migrate to PostgreSQL?",
            query_id=123
        )
        
        # Verify response structure
        assert response["response_type"] == "in_channel"
        assert "blocks" in response
        
        # Verify decision-specific content
        blocks_text = " ".join([block.get("text", {}).get("text", "") for block in response["blocks"] if "text" in block])
        assert "Migrate to PostgreSQL" in blocks_text
        assert "John" in blocks_text

    @pytest.mark.asyncio
    async def test_social_interaction_workflow(self, classifier, formatter):
        """Test complete workflow for social interaction."""
        # Test intent classification
        intent_result = await classifier.classify_intent(
            message_text="Hi there! How are you?",
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        # Verify intent classification
        assert intent_result.intent == "social_interaction"
        assert intent_result.requires_knowledge_search is False
        
        # Create social content data
        content_data = ContentData(
            main_content="Hello! How can I help you today?",
            sources=[],
            next_steps=[],
            related_info=[]
        )
        
        # Test response formatting
        response = await formatter.format_response(
            content_data=content_data,
            intent_result=intent_result,
            query_text="Hi there!",
            query_id=123
        )
        
        # Verify response structure
        assert response["response_type"] == "in_channel"
        assert "blocks" in response
        assert len(response["blocks"]) == 1  # Simple social response

    @pytest.mark.asyncio
    async def test_conversational_response_workflow(self, classifier, formatter):
        """Test complete workflow for conversational response."""
        # Test intent classification
        intent_result = await classifier.classify_intent(
            message_text="Thanks for the help!",
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        # Verify intent classification (might be classified as social_interaction)
        assert intent_result.intent in ["conversational_response", "social_interaction"]
        assert intent_result.requires_knowledge_search is False
        
        # Create conversational content data
        content_data = ContentData(
            main_content="I understand. Is there anything else I can help you with?",
            sources=[],
            next_steps=[],
            related_info=[]
        )
        
        # Test response formatting
        response = await formatter.format_response(
            content_data=content_data,
            intent_result=intent_result,
            query_text="Thanks!",
            query_id=123
        )
        
        # Verify response structure
        assert response["response_type"] == "in_channel"
        assert "blocks" in response

    @pytest.mark.asyncio
    async def test_clarification_request_workflow(self, classifier, formatter):
        """Test complete workflow for clarification request."""
        # Test intent classification
        intent_result = await classifier.classify_intent(
            message_text="Can you explain what you mean by that?",
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        # Verify intent classification
        assert intent_result.intent == "clarification_request"
        assert intent_result.requires_knowledge_search is True
        
        # Create clarification content data
        content_data = ContentData(
            main_content="Let me clarify the previous response with more details...",
            sources=[
                {
                    "title": "Previous Discussion",
                    "date": "2024-01-15",
                    "channel": "#general",
                    "excerpt": "Original explanation that needs clarification"
                }
            ],
            next_steps=["Review the detailed explanation", "Ask follow-up questions if needed"],
            related_info=["Related concepts", "Additional context"]
        )
        
        # Test response formatting
        response = await formatter.format_response(
            content_data=content_data,
            intent_result=intent_result,
            query_text="Can you explain what you mean?",
            query_id=123
        )
        
        # Verify response structure
        assert response["response_type"] == "in_channel"
        assert "blocks" in response

    @pytest.mark.asyncio
    async def test_process_query_workflow(self, classifier, formatter):
        """Test complete workflow for process query."""
        # Test intent classification
        intent_result = await classifier.classify_intent(
            message_text="What are the steps to deploy the application?",
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        # Verify intent classification
        assert intent_result.intent in ["knowledge_query", "process"]
        assert intent_result.requires_knowledge_search is True
        
        # Create process content data
        content_data = ContentData(
            main_content="Here's the step-by-step process to deploy the application.",
            sources=[
                {
                    "title": "Deployment Process",
                    "date": "2024-01-12",
                    "channel": "#devops",
                    "excerpt": "Complete deployment guide"
                }
            ],
            next_steps=["Follow the steps", "Test in staging", "Deploy to production"],
            related_info=["Prerequisites", "Required tools"],
            process_steps=[
                {"description": "Run tests", "responsible": "Developer", "tools": "pytest"},
                {"description": "Build image", "responsible": "CI/CD", "tools": "Docker"},
                {"description": "Deploy", "responsible": "DevOps", "tools": "Kubernetes"}
            ]
        )
        
        # Test response formatting
        response = await formatter.format_response(
            content_data=content_data,
            intent_result=intent_result,
            query_text="What are the deployment steps?",
            query_id=123
        )
        
        # Verify response structure
        assert response["response_type"] == "in_channel"
        assert "blocks" in response
        
        # Verify process-specific content
        blocks_text = " ".join([block.get("text", {}).get("text", "") for block in response["blocks"] if "text" in block])
        assert "Process Guide" in blocks_text
        assert "Run tests" in blocks_text

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, classifier, formatter):
        """Test error handling in the complete workflow."""
        # Test with invalid input
        intent_result = await classifier.classify_intent(
            message_text="",  # Empty message
            user_id="test_user",
            channel_id="test_channel",
            workspace_id=1,
            thread_ts=None,
            db=None
        )
        
        # Should handle empty message gracefully
        assert intent_result.intent == "ignore"
        assert intent_result.requires_knowledge_search is False
        
        # Test response formatting with minimal data
        content_data = ContentData(
            main_content="",
            sources=[],
            next_steps=[],
            related_info=[]
        )
        
        response = await formatter.format_response(
            content_data=content_data,
            intent_result=intent_result,
            query_text="",
            query_id=123
        )
        
        # Should create appropriate response
        assert response["response_type"] == "in_channel"
        assert "blocks" in response

    @pytest.mark.asyncio
    async def test_performance_workflow(self, classifier, formatter):
        """Test performance of the complete workflow."""
        import time
        
        # Test multiple queries to measure performance
        queries = [
            "How do I restart the service?",
            "What was the decision about the database?",
            "Hi there!",
            "Thanks for the help!",
            "Can you explain more?",
            "What are the deployment steps?"
        ]
        
        start_time = time.time()
        
        for query in queries:
            # Intent classification
            intent_result = await classifier.classify_intent(
                message_text=query,
                user_id="test_user",
                channel_id="test_channel",
                workspace_id=1,
                thread_ts=None,
                db=None
            )
            
            # Create minimal content data
            content_data = ContentData(
                main_content=f"Response to: {query}",
                sources=[],
                next_steps=[],
                related_info=[]
            )
            
            # Response formatting
            response = await formatter.format_response(
                content_data=content_data,
                intent_result=intent_result,
                query_text=query,
                query_id=123
            )
            
            # Verify response is created
            assert response["response_type"] == "in_channel"
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete all queries in reasonable time (less than 10 seconds)
        assert total_time < 10.0
        print(f"Processed {len(queries)} queries in {total_time:.2f} seconds")
