"""Consistency tests across paraphrases and small perturbations."""

import os
import pytest

from tests.fixtures.conversations import ConversationFixtures
from tests.fixtures.queries import QueryFixtures
from tests.generators.paraphrase import paraphrase_variants, small_perturbations
from tests.monitoring.quality_dashboard import quality_dashboard


@pytest.mark.integration
def test_semantic_consistency_across_paraphrases():
    os.environ["TEST_MODE"] = "true"
    conv = ConversationFixtures.get_technical_conversation()
    q = QueryFixtures.get_kafka_queries()[0]
    base_query = q["query"]
    ground_truth = q["expected_response"]
    response = (
        "To restart Kafka connector: docker restart kafka-connector. \n"
        "Check logs first with `docker logs kafka-connector`."
    )
    source_material = [m["text"] for m in conv["messages"]]

    base_metrics = quality_dashboard.evaluate_response(
        response_id="consistency_base",
        query=base_query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )

    base_score = base_metrics.overall_score
    variants = paraphrase_variants(base_query)[:5]

    deltas = []
    for i, v in enumerate(variants):
        m = quality_dashboard.evaluate_response(
            response_id=f"consistency_var_{i}",
            query=v,
            response=response,
            ground_truth=ground_truth,
            source_material=source_material,
        )
        deltas.append(abs(m.overall_score - base_score))

    # Expect small variance across paraphrases
    assert sum(deltas) / len(deltas) <= 0.2


@pytest.mark.integration
def test_deterministic_consistency_under_small_perturbations():
    os.environ["TEST_MODE"] = "true"
    conv = ConversationFixtures.get_process_conversation()
    query = "What's our deployment process?"
    ground_truth = (
        "Deployment process: 1. Build Docker image 2. Run tests 3. Deploy to staging 4. Deploy to prod"
    )
    response = ground_truth
    source_material = [m["text"] for m in conv["messages"]]

    base_metrics = quality_dashboard.evaluate_response(
        response_id="perturb_base",
        query=query,
        response=response,
        ground_truth=ground_truth,
        source_material=source_material,
    )
    base_score = base_metrics.overall_score

    variants = small_perturbations(query)
    deltas = []
    for i, v in enumerate(variants):
        m = quality_dashboard.evaluate_response(
            response_id=f"perturb_var_{i}",
            query=v,
            response=response,
            ground_truth=ground_truth,
            source_material=source_material,
        )
        deltas.append(abs(m.overall_score - base_score))

    assert sum(deltas) / len(deltas) <= 0.2


