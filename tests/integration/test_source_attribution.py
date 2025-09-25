"""Integration tests for source attribution correctness."""

import os
import pytest

from tests.fixtures.conversations import ConversationFixtures
from tests.monitoring.quality_dashboard import quality_dashboard


@pytest.mark.integration
def test_source_verification_detects_unverified_claims():
    os.environ["TEST_MODE"] = "true"

    conv = ConversationFixtures.get_decision_conversation()
    query = "What migration approach did we decide on?"
    ground_truth = (
        "Database migration approach: rolling updates instead of blue-green deployment"
    )

    # Response invents a person not in sources and wrong approach
    response = (
        "Alice confirmed blue-green deployment yesterday at 5pm. Proceed immediately."
    )
    source_material = [m["text"] for m in conv["messages"]]

    metrics = quality_dashboard.evaluate_response(
        response_id="resp_source_001",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )

    # Expect low source verification score (i.e., high hallucination component)
    assert metrics.hallucination["overall_hallucination_score"] >= 0.2

