"""Query test fixtures."""

from typing import Dict, Any, List


class QueryFixtures:
    """Predefined query test data."""
    
    @staticmethod
    def get_kafka_queries() -> List[Dict[str, Any]]:
        """Kafka-related test queries."""
        return [
            {
                "query": "How do I restart the Kafka connector?",
                "expected_response": "To restart Kafka connector: docker restart kafka-connector",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.9,
                "query_type": "factual_retrieval"
            },
            {
                "query": "What's wrong with Kafka?",
                "expected_response": "Check the logs first: docker logs kafka-connector",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.8,
                "query_type": "troubleshooting"
            },
            {
                "query": "Kafka connector troubleshooting steps",
                "expected_response": "1. Check logs: docker logs kafka-connector 2. Restart: docker restart kafka-connector",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.85,
                "query_type": "process_inquiry"
            }
        ]
    
    @staticmethod
    def get_deployment_queries() -> List[Dict[str, Any]]:
        """Deployment-related test queries."""
        return [
            {
                "query": "What's our deployment process?",
                "expected_response": "Deployment process: 1. Build Docker image 2. Run tests 3. Deploy to staging 4. Deploy to prod",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.9,
                "query_type": "process_inquiry"
            },
            {
                "query": "Who approves production deployments?",
                "expected_response": "Get approval from @U6666666666 before production deployment",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.85,
                "query_type": "attribution_query"
            }
        ]
    
    @staticmethod
    def get_migration_queries() -> List[Dict[str, Any]]:
        """Migration-related test queries."""
        return [
            {
                "query": "What migration approach did we decide on?",
                "expected_response": "Database migration approach: rolling updates instead of blue-green deployment",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.9,
                "query_type": "factual_retrieval"
            },
            {
                "query": "When are we starting the migration?",
                "expected_response": "Starting with staging environment next week",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.8,
                "query_type": "temporal_query"
            }
        ]
    
    @staticmethod
    def get_negative_queries() -> List[Dict[str, Any]]:
        """Queries with no expected answer in test data."""
        return [
            {
                "query": "How do I configure Redis?",
                "expected_response": None,  # No answer in test data
                "expected_sources": [],
                "quality_score": 0.0,
                "query_type": "factual_retrieval"
            },
            {
                "query": "What's the password for the database?",
                "expected_response": None,  # No answer in test data
                "expected_sources": [],
                "quality_score": 0.0,
                "query_type": "factual_retrieval"
            }
        ]
    
    @staticmethod
    def get_ambiguous_queries() -> List[Dict[str, Any]]:
        """Ambiguous queries that might have multiple interpretations."""
        return [
            {
                "query": "What's the issue?",
                "expected_response": "Check the logs first: docker logs kafka-connector",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.6,  # Lower score due to ambiguity
                "query_type": "troubleshooting"
            },
            {
                "query": "How do we do it?",
                "expected_response": "Deployment process: 1. Build Docker image 2. Run tests 3. Deploy to staging 4. Deploy to prod",
                "expected_sources": ["C1234567890"],
                "quality_score": 0.5,  # Lower score due to ambiguity
                "query_type": "process_inquiry"
            }
        ]
    
    @staticmethod
    def get_all_queries() -> List[Dict[str, Any]]:
        """Get all query fixtures."""
        all_queries = []
        all_queries.extend(QueryFixtures.get_kafka_queries())
        all_queries.extend(QueryFixtures.get_deployment_queries())
        all_queries.extend(QueryFixtures.get_migration_queries())
        all_queries.extend(QueryFixtures.get_negative_queries())
        all_queries.extend(QueryFixtures.get_ambiguous_queries())
        return all_queries
    
    @staticmethod
    def get_queries_by_type(query_type: str) -> List[Dict[str, Any]]:
        """Get queries filtered by type."""
        all_queries = QueryFixtures.get_all_queries()
        return [q for q in all_queries if q.get("query_type") == query_type]
    
    @staticmethod
    def get_queries_with_answers() -> List[Dict[str, Any]]:
        """Get queries that should have answers in test data."""
        all_queries = QueryFixtures.get_all_queries()
        return [q for q in all_queries if q.get("expected_response") is not None]
    
    @staticmethod
    def get_queries_without_answers() -> List[Dict[str, Any]]:
        """Get queries that should not have answers in test data."""
        all_queries = QueryFixtures.get_all_queries()
        return [q for q in all_queries if q.get("expected_response") is None]
