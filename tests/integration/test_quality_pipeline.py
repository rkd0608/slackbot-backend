"""Integration tests for quality evaluation pipeline and dashboard."""

import os
from typing import List

import pytest

from tests.fixtures.conversations import ConversationFixtures
from tests.fixtures.queries import QueryFixtures
from tests.monitoring.quality_dashboard import quality_dashboard


@pytest.mark.integration
def test_quality_evaluation_end_to_end():
    os.environ["TEST_MODE"] = "true"

    # Arrange test data
    conv = ConversationFixtures.get_technical_conversation()
    query = QueryFixtures.get_kafka_queries()[0]["query"]
    ground_truth = QueryFixtures.get_kafka_queries()[0]["expected_response"]

    # Simulate an AI response close to ground truth
    response = (
        "To restart Kafka connector: docker restart kafka-connector. "
        "Before that, check logs with `docker logs kafka-connector`."
    )

    # Build source material as list of message texts
    source_material: List[str] = [m["text"] for m in conv["messages"]]

    # Act
    metrics = quality_dashboard.evaluate_response(
        response_id="resp_kafka_001",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )

    # Assert basic expectations
    assert metrics.factual_accuracy["overall_score"] >= 0.6
    assert metrics.completeness["overall_completeness"] >= 0.6
    assert metrics.utility["overall_utility"] >= 0.6
    assert metrics.hallucination["overall_hallucination_score"] <= 0.5
    assert metrics.overall_score >= 0.6


@pytest.mark.integration
def test_hallucination_detection_with_contradiction():
    os.environ["TEST_MODE"] = "true"

    conv = ConversationFixtures.get_decision_conversation()
    source_material = [m["text"] for m in conv["messages"]]

    # Response contradicts decision (claims blue-green was chosen)
    response = (
        "Team decided to use blue-green deployment for database migration. "
        "Rollout to prod immediately."
    )

    query = "What migration approach did we decide on?"
    ground_truth = "Database migration approach: rolling updates instead of blue-green deployment"

    metrics = quality_dashboard.evaluate_response(
        response_id="resp_migration_001",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )

    assert metrics.hallucination["overall_hallucination_score"] >= 0.3
