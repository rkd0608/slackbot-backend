"""Adversarial and edge-case testing suites."""

import os
import pytest

from tests.fixtures.conversations import ConversationFixtures
from tests.monitoring.quality_dashboard import quality_dashboard


@pytest.mark.integration
def test_noise_and_typos_tolerance():
    os.environ["TEST_MODE"] = "true"
    conv = ConversationFixtures.get_technical_conversation()
    query = "Hwo do I rsetrat the Kafak connetor?"  # noisy/typo query
    ground_truth = (
        "To restart Kafka connector: docker restart kafka-connector"
    )
    response = (
        "To restart Kafka connector: docker restart kafka-connector. Check logs using `docker logs kafka-connector`."
    )
    source_material = [m["text"] for m in conv["messages"]]

    m = quality_dashboard.evaluate_response(
        response_id="adv_noise_001",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )
    # We allow lower scores, but should not collapse entirely
    assert m.overall_score >= 0.3


@pytest.mark.integration
def test_multi_topic_conversation_asks_specific_query():
    os.environ["TEST_MODE"] = "true"
    conv = ConversationFixtures.get_process_conversation()
    # Query about one topic; conversation mentions another topic
    query = "What are the deployment steps?"
    ground_truth = (
        "Deployment process: 1. Build Docker image 2. Run tests 3. Deploy to staging 4. Deploy to prod"
    )
    response = ground_truth
    source_material = [m["text"] for m in conv["messages"]]

    m = quality_dashboard.evaluate_response(
        response_id="adv_multitopic_001",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )
    assert m.completeness["overall_completeness"] >= 0.4


