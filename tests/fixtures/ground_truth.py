"""Ground truth test fixtures."""

from typing import Dict, Any, List


class GroundTruthFixtures:
    """Ground truth reference data for testing."""
    
    @staticmethod
    def get_kafka_ground_truth() -> Dict[str, Any]:
        """Ground truth for Kafka-related knowledge."""
        return {
            "conversation_id": "conv_kafka_001",
            "knowledge_items": [
                {
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
            ],
            "queries": [
                {
                    "query": "How do I restart the Kafka connector?",
                    "expected_response": "To restart Kafka connector: docker restart kafka-connector",
                    "expected_sources": ["C1234567890"],
                    "quality_metrics": {
                        "factual_accuracy": 0.95,
                        "completeness": 0.9,
                        "source_attribution": 0.9,
                        "actionability": 0.95
                    }
                },
                {
                    "query": "What's wrong with Kafka?",
                    "expected_response": "Check the logs first: docker logs kafka-connector",
                    "expected_sources": ["C1234567890"],
                    "quality_metrics": {
                        "factual_accuracy": 0.85,
                        "completeness": 0.8,
                        "source_attribution": 0.9,
                        "actionability": 0.85
                    }
                }
            ]
        }
    
    @staticmethod
    def get_deployment_ground_truth() -> Dict[str, Any]:
        """Ground truth for deployment-related knowledge."""
        return {
            "conversation_id": "conv_deployment_001",
            "knowledge_items": [
                {
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
            ],
            "queries": [
                {
                    "query": "What's our deployment process?",
                    "expected_response": "Deployment process: 1. Build Docker image 2. Run tests 3. Deploy to staging 4. Deploy to prod",
                    "expected_sources": ["C1234567890"],
                    "quality_metrics": {
                        "factual_accuracy": 0.95,
                        "completeness": 0.9,
                        "source_attribution": 0.9,
                        "actionability": 0.9
                    }
                },
                {
                    "query": "Who approves production deployments?",
                    "expected_response": "Get approval from @U6666666666 before production deployment",
                    "expected_sources": ["C1234567890"],
                    "quality_metrics": {
                        "factual_accuracy": 0.9,
                        "completeness": 0.85,
                        "source_attribution": 0.9,
                        "actionability": 0.8
                    }
                }
            ]
        }
    
    @staticmethod
    def get_migration_ground_truth() -> Dict[str, Any]:
        """Ground truth for migration-related knowledge."""
        return {
            "conversation_id": "conv_migration_001",
            "knowledge_items": [
                {
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
            ],
            "queries": [
                {
                    "query": "What migration approach did we decide on?",
                    "expected_response": "Database migration approach: rolling updates instead of blue-green deployment",
                    "expected_sources": ["C1234567890"],
                    "quality_metrics": {
                        "factual_accuracy": 0.9,
                        "completeness": 0.85,
                        "source_attribution": 0.9,
                        "actionability": 0.8
                    }
                },
                {
                    "query": "When are we starting the migration?",
                    "expected_response": "Starting with staging environment next week",
                    "expected_sources": ["C1234567890"],
                    "quality_metrics": {
                        "factual_accuracy": 0.85,
                        "completeness": 0.8,
                        "source_attribution": 0.9,
                        "actionability": 0.75
                    }
                }
            ]
        }
    
    @staticmethod
    def get_all_ground_truth() -> List[Dict[str, Any]]:
        """Get all ground truth fixtures."""
        return [
            GroundTruthFixtures.get_kafka_ground_truth(),
            GroundTruthFixtures.get_deployment_ground_truth(),
            GroundTruthFixtures.get_migration_ground_truth()
        ]
    
    @staticmethod
    def get_quality_thresholds() -> Dict[str, float]:
        """Get quality thresholds for evaluation."""
        return {
            "factual_accuracy": 0.8,
            "completeness": 0.7,
            "source_attribution": 0.8,
            "actionability": 0.75,
            "overall_quality": 0.75
        }
    
    @staticmethod
    def get_evaluation_criteria() -> Dict[str, Any]:
        """Get evaluation criteria for quality assessment."""
        return {
            "factual_accuracy": {
                "description": "Accuracy of factual information in responses",
                "weight": 0.3,
                "min_threshold": 0.8
            },
            "completeness": {
                "description": "How completely the response addresses the query",
                "weight": 0.25,
                "min_threshold": 0.7
            },
            "source_attribution": {
                "description": "Proper citation of information sources",
                "weight": 0.2,
                "min_threshold": 0.8
            },
            "actionability": {
                "description": "How actionable the response is for the user",
                "weight": 0.25,
                "min_threshold": 0.75
            }
        }
