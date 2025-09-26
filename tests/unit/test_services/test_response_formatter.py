"""Unit tests for ResponseFormatter service."""

import pytest
from unittest.mock import MagicMock

from app.services.response_formatter import ResponseFormatter, ContentData, ResponseTemplate
from app.services.intent_classifier import IntentClassificationResult


class TestResponseFormatter:
    """Test cases for ResponseFormatter service."""

    @pytest.fixture
    def formatter(self):
        """Create ResponseFormatter instance for testing."""
        return ResponseFormatter()

    @pytest.fixture
    def decision_content_data(self):
        """Create decision content data for testing."""
        return ContentData(
            main_content="The team decided to migrate to PostgreSQL for better JSON support.",
            sources=[
                {"title": "Architecture Discussion", "date": "2024-01-15", "channel": "#engineering"},
                {"title": "Database Migration Plan", "date": "2024-01-16", "channel": "#engineering"}
            ],
            next_steps=["Create migration timeline", "Set up staging environment"],
            related_info=["Current MySQL setup", "JSON performance requirements"],
            decision_info={
                "outcome": "Migrate to PostgreSQL",
                "decision_maker": "John (Senior Engineer)",
                "date": "January 15, 2024",
                "rationale": "Better JSON support needed for analytics features"
            }
        )

    @pytest.fixture
    def process_content_data(self):
        """Create process content data for testing."""
        return ContentData(
            main_content="Here's how to deploy the application to production.",
            sources=[
                {"title": "Deployment Guide", "date": "2024-01-10", "channel": "#devops"}
            ],
            next_steps=["Run tests", "Build Docker image", "Deploy to staging"],
            related_info=["Docker configuration", "Environment variables"],
            process_steps=[
                {"description": "Run full test suite", "responsible": "Developer", "tools": "pytest"},
                {"description": "Build Docker image", "responsible": "CI/CD", "tools": "Docker"},
                {"description": "Deploy to staging", "responsible": "DevOps", "tools": "Kubernetes"}
            ]
        )

    @pytest.fixture
    def technical_content_data(self):
        """Create technical content data for testing."""
        return ContentData(
            main_content="Here's how to fix the SSL certificate issue.",
            sources=[
                {"title": "SSL Troubleshooting", "date": "2024-01-12", "channel": "#security"}
            ],
            next_steps=["Update certificate", "Test HTTPS", "Monitor logs"],
            related_info=["Certificate authority", "SSL configuration"],
            technical_details={
                "problem": "SSL certificate expired",
                "solution": "Generate new certificate and update configuration",
                "implementation": ["Generate new cert", "Update nginx config", "Restart services"],
                "tools": ["openssl", "nginx", "certbot"],
                "verification": "Check certificate validity and HTTPS connectivity"
            }
        )

    @pytest.fixture
    def social_content_data(self):
        """Create social content data for testing."""
        return ContentData(
            main_content="Hello! How can I help you today?",
            sources=[],
            next_steps=[],
            related_info=[]
        )

    @pytest.fixture
    def knowledge_query_intent(self):
        """Create knowledge query intent result."""
        return IntentClassificationResult(
            intent="knowledge_query",
            confidence=0.9,
            classification_method="rule_based",
            contextual_metadata={},
            entities=[],
            temporal_scope=None,
            is_conversational_response=False,
            requires_knowledge_search=True
        )

    @pytest.fixture
    def social_intent(self):
        """Create social interaction intent result."""
        return IntentClassificationResult(
            intent="social_interaction",
            confidence=0.8,
            classification_method="rule_based",
            contextual_metadata={},
            entities=[],
            temporal_scope=None,
            is_conversational_response=False,
            requires_knowledge_search=False
        )

    @pytest.mark.asyncio
    async def test_format_response_decision(self, formatter, decision_content_data, knowledge_query_intent):
        """Test formatting decision response."""
        result = await formatter.format_response(
            content_data=decision_content_data,
            intent_result=knowledge_query_intent,
            query_text="What was the database decision?",
            query_id=123
        )
        
        assert result["response_type"] == "in_channel"
        assert "text" in result
        assert "blocks" in result
        assert len(result["blocks"]) > 0
        
        # Check for decision-specific content
        blocks_text = " ".join([block.get("text", {}).get("text", "") for block in result["blocks"] if "text" in block])
        assert "Migrate to PostgreSQL" in blocks_text
        assert "John (Senior Engineer)" in blocks_text

    @pytest.mark.asyncio
    async def test_format_response_process(self, formatter, process_content_data, knowledge_query_intent):
        """Test formatting process response."""
        result = await formatter.format_response(
            content_data=process_content_data,
            intent_result=knowledge_query_intent,
            query_text="How do I deploy the app?",
            query_id=123
        )
        
        assert result["response_type"] == "in_channel"
        assert "blocks" in result
        
        # Check for process-specific content
        blocks_text = " ".join([block.get("text", {}).get("text", "") for block in result["blocks"] if "text" in block])
        assert "Process Guide" in blocks_text
        assert "Run full test suite" in blocks_text

    @pytest.mark.asyncio
    async def test_format_response_technical(self, formatter, technical_content_data, knowledge_query_intent):
        """Test formatting technical solution response."""
        result = await formatter.format_response(
            content_data=technical_content_data,
            intent_result=knowledge_query_intent,
            query_text="How do I fix SSL issues?",
            query_id=123
        )
        
        assert result["response_type"] == "in_channel"
        assert "blocks" in result
        
        # Check for technical-specific content
        blocks_text = " ".join([block.get("text", {}).get("text", "") for block in result["blocks"] if "text" in block])
        assert "Technical Solution" in blocks_text
        assert "SSL certificate expired" in blocks_text

    @pytest.mark.asyncio
    async def test_format_response_social(self, formatter, social_content_data, social_intent):
        """Test formatting social interaction response."""
        result = await formatter.format_response(
            content_data=social_content_data,
            intent_result=social_intent,
            query_text="Hi there!",
            query_id=123
        )
        
        assert result["response_type"] == "in_channel"
        assert "blocks" in result
        assert len(result["blocks"]) == 1  # Simple social response

    @pytest.mark.asyncio
    async def test_format_response_no_information(self, formatter):
        """Test formatting no information response."""
        empty_content = ContentData(
            main_content="",
            sources=[],
            next_steps=[],
            related_info=[]
        )
        
        # Create a non-social intent for no information
        no_info_intent = IntentClassificationResult(
            intent="knowledge_query",
            confidence=0.5,
            classification_method="rule_based",
            contextual_metadata={},
            entities=[],
            temporal_scope=None,
            is_conversational_response=False,
            requires_knowledge_search=True
        )
        
        result = await formatter.format_response(
            content_data=empty_content,
            intent_result=no_info_intent,
            query_text="Unknown topic",
            query_id=123
        )
        
        assert result["response_type"] == "in_channel"
        assert "couldn't find specific information" in result["text"]

    def test_determine_template_type_decision(self, formatter, decision_content_data, knowledge_query_intent):
        """Test template type determination for decision content."""
        template_type = formatter._determine_template_type(decision_content_data, knowledge_query_intent)
        assert template_type == "decision_response"

    def test_determine_template_type_process(self, formatter, process_content_data, knowledge_query_intent):
        """Test template type determination for process content."""
        template_type = formatter._determine_template_type(process_content_data, knowledge_query_intent)
        assert template_type == "process_response"

    def test_determine_template_type_technical(self, formatter, technical_content_data, knowledge_query_intent):
        """Test template type determination for technical content."""
        template_type = formatter._determine_template_type(technical_content_data, knowledge_query_intent)
        assert template_type == "technical_solution"

    def test_determine_template_type_social(self, formatter, social_content_data, social_intent):
        """Test template type determination for social content."""
        template_type = formatter._determine_template_type(social_content_data, social_intent)
        assert template_type == "social_response"

    def test_determine_template_type_no_info(self, formatter):
        """Test template type determination for no information."""
        empty_content = ContentData(
            main_content="",
            sources=[],
            next_steps=[],
            related_info=[]
        )
        
        # Create a non-social intent for no information
        no_info_intent = IntentClassificationResult(
            intent="knowledge_query",
            confidence=0.5,
            classification_method="rule_based",
            contextual_metadata={},
            entities=[],
            temporal_scope=None,
            is_conversational_response=False,
            requires_knowledge_search=True
        )
        
        template_type = formatter._determine_template_type(empty_content, no_info_intent)
        assert template_type == "no_information"

    def test_create_decision_template(self, formatter, decision_content_data, knowledge_query_intent):
        """Test creation of decision response template."""
        template = formatter._create_decision_template(
            decision_content_data, knowledge_query_intent, "Test query", 123
        )
        
        assert isinstance(template, ResponseTemplate)
        assert template.template_type == "decision_response"
        assert len(template.blocks) > 0
        assert "Migrate to PostgreSQL" in template.text_fallback
        
        # Check for decision-specific blocks
        block_texts = [block.get("text", {}).get("text", "") for block in template.blocks if "text" in block]
        combined_text = " ".join(block_texts)
        assert "Migrate to PostgreSQL" in combined_text
        assert "John (Senior Engineer)" in combined_text

    def test_create_process_template(self, formatter, process_content_data, knowledge_query_intent):
        """Test creation of process response template."""
        template = formatter._create_process_template(
            process_content_data, knowledge_query_intent, "Test query", 123
        )
        
        assert isinstance(template, ResponseTemplate)
        assert template.template_type == "process_response"
        assert len(template.blocks) > 0
        
        # Check for process-specific blocks
        block_texts = [block.get("text", {}).get("text", "") for block in template.blocks if "text" in block]
        combined_text = " ".join(block_texts)
        assert "Process Guide" in combined_text
        assert "Run full test suite" in combined_text

    def test_create_technical_template(self, formatter, technical_content_data, knowledge_query_intent):
        """Test creation of technical solution template."""
        template = formatter._create_technical_template(
            technical_content_data, knowledge_query_intent, "Test query", 123
        )
        
        assert isinstance(template, ResponseTemplate)
        assert template.template_type == "technical_solution"
        assert len(template.blocks) > 0
        
        # Check for technical-specific blocks
        block_texts = [block.get("text", {}).get("text", "") for block in template.blocks if "text" in block]
        combined_text = " ".join(block_texts)
        assert "Technical Solution" in combined_text
        assert "SSL certificate expired" in combined_text

    def test_create_social_template(self, formatter, social_content_data, social_intent):
        """Test creation of social interaction template."""
        template = formatter._create_social_template(
            social_content_data, social_intent, "Test query", 123
        )
        
        assert isinstance(template, ResponseTemplate)
        assert template.template_type == "social_response"
        assert len(template.blocks) == 1  # Simple social response

    def test_create_no_info_template(self, formatter, social_intent):
        """Test creation of no information template."""
        empty_content = ContentData(
            main_content="",
            sources=[],
            next_steps=[],
            related_info=[]
        )
        
        template = formatter._create_no_info_template(
            empty_content, social_intent, "Unknown topic", 123
        )
        
        assert isinstance(template, ResponseTemplate)
        assert template.template_type == "no_information"
        assert "couldn't find specific information" in template.text_fallback

    def test_add_sources_section(self, formatter):
        """Test adding sources section to blocks."""
        blocks = []
        sources = [
            {"title": "Source 1", "date": "2024-01-01", "channel": "#general", "excerpt": "Short excerpt"},
            {"title": "Source 2", "date": "2024-01-02", "channel": "#engineering", "excerpt": "Another excerpt"}
        ]
        
        formatter._add_sources_section(blocks, sources)
        
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert "Sources:" in blocks[0]["text"]["text"]
        assert "Source 1" in blocks[0]["text"]["text"]
        assert "Source 2" in blocks[0]["text"]["text"]

    def test_add_action_buttons(self, formatter):
        """Test adding action buttons to blocks."""
        blocks = []
        
        formatter._add_action_buttons(blocks, 123, "#2E8B57")
        
        assert len(blocks) == 1
        assert blocks[0]["type"] == "actions"
        assert len(blocks[0]["elements"]) == 3  # Helpful, Not Helpful, View Sources
        
        button_texts = [button["text"]["text"] for button in blocks[0]["elements"]]
        assert "👍 Helpful" in button_texts
        assert "👎 Not Helpful" in button_texts
        assert "🔗 View Sources" in button_texts

    def test_create_text_fallback(self, formatter, decision_content_data, knowledge_query_intent):
        """Test creation of text fallback."""
        text_fallback = formatter._create_text_fallback(
            decision_content_data, knowledge_query_intent, "Test query"
        )
        
        assert isinstance(text_fallback, str)
        assert len(text_fallback) > 0
        assert "Migrate to PostgreSQL" in text_fallback

    def test_create_error_response(self, formatter):
        """Test creation of error response."""
        error_response = formatter._create_error_response("Test query", "Test error")
        
        assert error_response["response_type"] == "ephemeral"
        assert "Error processing your request" in error_response["text"]
        assert "Test error" in error_response["text"]

    def test_template_configs(self, formatter):
        """Test template configurations are properly set."""
        expected_configs = [
            "decision_response", "process_response", "technical_solution",
            "general_info", "no_information", "social_response"
        ]
        
        for config_name in expected_configs:
            assert config_name in formatter.template_configs
            config = formatter.template_configs[config_name]
            assert "title" in config
            assert "icon" in config
            assert "color" in config

    def test_formatting_standards(self, formatter):
        """Test formatting standards are properly set."""
        standards = formatter.formatting_standards
        
        assert "max_section_length" in standards
        assert "max_source_count" in standards
        assert "max_next_steps" in standards
        assert "max_related_info" in standards
        
        assert standards["max_section_length"] > 0
        assert standards["max_source_count"] > 0
        assert standards["max_next_steps"] > 0
        assert standards["max_related_info"] > 0
