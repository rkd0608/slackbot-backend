"""Conversation test fixtures."""

from typing import Dict, Any, List


class ConversationFixtures:
    """Predefined conversation test data."""
    
    @staticmethod
    def get_technical_conversation() -> Dict[str, Any]:
        """Technical troubleshooting conversation."""
        return {
            "messages": [
                {
                    "user": "U1234567890",
                    "text": "The Kafka connector is failing again",
                    "ts": "1234567890.001",
                    "type": "message"
                },
                {
                    "user": "U0987654321", 
                    "text": "Check the logs first: docker logs kafka-connector",
                    "ts": "1234567890.002",
                    "type": "message"
                },
                {
                    "user": "U1234567890",
                    "text": "Found the issue: connection timeout to the database",
                    "ts": "1234567890.003",
                    "type": "message"
                },
                {
                    "user": "U0987654321",
                    "text": "Try restarting with: docker restart kafka-connector",
                    "ts": "1234567890.004",
                    "type": "message"
                },
                {
                    "user": "U1234567890",
                    "text": "That worked! Thanks",
                    "ts": "1234567890.005",
                    "type": "message"
                }
            ],
            "expected_knowledge": {
                "title": "Kafka connector restart procedure",
                "summary": "Steps to restart Kafka connector when it fails",
                "content": "To restart Kafka connector: docker restart kafka-connector. Check logs first with: docker logs kafka-connector",
                "confidence_score": 0.9,
                "knowledge_type": "troubleshooting",
                "metadata": {
                    "source_channel_id": "C1234567890",
                    "participants": ["U1234567890", "U0987654321"],
                    "created_date": "2025-09-25T01:00:00Z",
                    "message_count": 5,
                    "extraction_method": "test"
                }
            }
        }
    
    @staticmethod
    def get_decision_conversation() -> Dict[str, Any]:
        """Decision-making conversation."""
        return {
            "messages": [
                {
                    "user": "U1111111111",
                    "text": "We need to decide on the database migration approach",
                    "ts": "1234567890.101",
                    "type": "message"
                },
                {
                    "user": "U2222222222",
                    "text": "I think we should use blue-green deployment",
                    "ts": "1234567890.102",
                    "type": "message"
                },
                {
                    "user": "U3333333333",
                    "text": "That's too risky. Let's do rolling updates instead",
                    "ts": "1234567890.103",
                    "type": "message"
                },
                {
                    "user": "U1111111111",
                    "text": "Rolling updates it is. We'll start with staging next week",
                    "ts": "1234567890.104",
                    "type": "message"
                }
            ],
            "expected_knowledge": {
                "title": "Database migration strategy decision",
                "summary": "Team decided to use rolling updates for database migration",
                "content": "Database migration approach: rolling updates instead of blue-green deployment. Starting with staging environment next week.",
                "confidence_score": 0.85,
                "knowledge_type": "decision",
                "metadata": {
                    "source_channel_id": "C1234567890",
                    "participants": ["U1111111111", "U2222222222", "U3333333333"],
                    "created_date": "2025-09-25T01:00:00Z",
                    "message_count": 4,
                    "extraction_method": "test"
                }
            }
        }
    
    @staticmethod
    def get_process_conversation() -> Dict[str, Any]:
        """Process definition conversation."""
        return {
            "messages": [
                {
                    "user": "U4444444444",
                    "text": "What's our deployment process?",
                    "ts": "1234567890.201",
                    "type": "message"
                },
                {
                    "user": "U5555555555",
                    "text": "1. Build Docker image 2. Run tests 3. Deploy to staging 4. Deploy to prod",
                    "ts": "1234567890.202",
                    "type": "message"
                },
                {
                    "user": "U4444444444",
                    "text": "Do we need approval for prod deployment?",
                    "ts": "1234567890.203",
                    "type": "message"
                },
                {
                    "user": "U5555555555",
                    "text": "Yes, get approval from @U6666666666 before prod",
                    "ts": "1234567890.204",
                    "type": "message"
                }
            ],
            "expected_knowledge": {
                "title": "Deployment process steps",
                "summary": "Standard deployment process with approval requirements",
                "content": "Deployment process: 1. Build Docker image 2. Run tests 3. Deploy to staging 4. Deploy to prod. Get approval from @U6666666666 before production deployment.",
                "confidence_score": 0.9,
                "knowledge_type": "process",
                "metadata": {
                    "source_channel_id": "C1234567890",
                    "participants": ["U4444444444", "U5555555555"],
                    "created_date": "2025-09-25T01:00:00Z",
                    "message_count": 4,
                    "extraction_method": "test"
                }
            }
        }
    
    @staticmethod
    def get_incomplete_conversation() -> Dict[str, Any]:
        """Incomplete conversation without clear resolution."""
        return {
            "messages": [
                {
                    "user": "U7777777777",
                    "text": "Having issues with the API",
                    "ts": "1234567890.301",
                    "type": "message"
                },
                {
                    "user": "U8888888888",
                    "text": "What kind of issues?",
                    "ts": "1234567890.302",
                    "type": "message"
                },
                {
                    "user": "U7777777777",
                    "text": "Not sure, need to investigate more",
                    "ts": "1234567890.303",
                    "type": "message"
                }
            ],
            "expected_knowledge": None  # No clear knowledge to extract
        }
    
    @staticmethod
    def get_all_conversations() -> List[Dict[str, Any]]:
        """Get all conversation fixtures."""
        return [
            ConversationFixtures.get_technical_conversation(),
            ConversationFixtures.get_decision_conversation(),
            ConversationFixtures.get_process_conversation(),
            ConversationFixtures.get_incomplete_conversation()
        ]
