"""Integration tests for negative and ambiguous queries."""

import os
import pytest

from tests.fixtures.conversations import ConversationFixtures
from tests.fixtures.queries import QueryFixtures
from tests.monitoring.quality_dashboard import quality_dashboard


@pytest.mark.integration
def test_negative_query_has_low_scores():
    os.environ["TEST_MODE"] = "true"

    conv = ConversationFixtures.get_technical_conversation()
    negative = QueryFixtures.get_negative_queries()[0]

    query = negative["query"]
    ground_truth = negative["expected_response"] or ""
    # Simulate a generic model response that shouldn't falsely claim facts
    response = (
        "I couldn't find this in prior discussions. Please share more details or point to a thread."
    )
    source_material = [m["text"] for m in conv["messages"]]

    metrics = quality_dashboard.evaluate_response(
        response_id="resp_negative_001",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )

    # Expect low completeness/utility but also low hallucination
    assert metrics.completeness["overall_completeness"] <= 0.5
    assert metrics.utility["overall_utility"] <= 0.6
    assert metrics.hallucination["overall_hallucination_score"] <= 0.4


@pytest.mark.integration
def test_ambiguous_query_prefers_actions_or_clarification():
    os.environ["TEST_MODE"] = "true"

    conv = ConversationFixtures.get_process_conversation()
    ambiguous = QueryFixtures.get_ambiguous_queries()[0]

    query = ambiguous["query"]
    ground_truth = ambiguous["expected_response"]
    response = (
        "You might be referring to deployment steps. Try: 1. Build image 2. Run tests 3. Deploy staging 4. Deploy prod"
    )
    source_material = [m["text"] for m in conv["messages"]]

    metrics = quality_dashboard.evaluate_response(
        response_id="resp_ambiguous_001",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )

    # Expect at least moderate utility due to actionable steps
    assert metrics.utility["overall_utility"] >= 0.4

