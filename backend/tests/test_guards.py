"""
Unit & Security Guard Tests for DomainGuard and QuestionGuard.
Tests closed-domain boundary, feature boundary, and anti-hallucination controls.
"""

import pytest
from backend.app.agent.domain_guard import DomainGuard
from backend.app.agent.question_guard import QuestionGuard
from backend.app.agent.state import AgentState
from backend.app.ml.preprocessing import SELECTED_FEATURES


def test_domain_guard_off_topic_redirection():
    """Test G: Off-topic queries (jokes, weather, car buying, trivia) are intercepted."""
    off_topic_queries = [
        "Tell me a joke.",
        "What's the weather today?",
        "Which car should I buy?",
        "Which laptop should I buy?",
        "Write me a poem about dogs.",
    ]

    for query in off_topic_queries:
        is_allowed, redirect_msg = DomainGuard.is_in_domain(query)
        assert not is_allowed, f"Query '{query}' should be intercepted as off-topic"
        assert "maintenance" in redirect_msg.lower() or "autocare" in redirect_msg.lower()


def test_domain_guard_valid_vehicle_maintenance():
    """Valid vehicle maintenance queries pass domain guard."""
    valid_queries = [
        "My car is 10 years old.",
        "Brakes are making noise.",
        "Battery seems weak.",
        "I don't know my service history.",
        "How much risk does my vehicle have?",
    ]

    for query in valid_queries:
        is_allowed, redirect_msg = DomainGuard.is_in_domain(query)
        assert is_allowed, f"Query '{query}' should pass domain guard"
        assert redirect_msg == ""


def test_question_guard_legal_features():
    """Test C & Section 58: QuestionGuard strictly allows only FINAL_MODEL_FEATURES."""
    q_guard = QuestionGuard(SELECTED_FEATURES)

    # Allowed features
    for feat in SELECTED_FEATURES:
        assert q_guard.is_legal_model_feature(feat)

    # Illegal / Unsupported features MUST be rejected
    illegal_features = [
        "Suspension_Condition",
        "Suspension",
        "Engine_Oil",
        "Coolant",
        "Spark_Plugs",
        "Alternator",
        "Engine_Temperature",
    ]

    for illegal_feat in illegal_features:
        assert not q_guard.is_legal_model_feature(illegal_feat), (
            f"Feature '{illegal_feat}' MUST BE REJECTED by QuestionGuard!"
        )


def test_question_guard_already_asked_or_declined():
    """Test D: QuestionGuard rejects asking features already known or declined."""
    q_guard = QuestionGuard(SELECTED_FEATURES)
    state = AgentState(conversation_id="test-guard-session")

    # Mark Brake_Condition as declined
    state.mark_declined("Brake_Condition")

    # Mark Vehicle_Age as known
    state.update_feature("Vehicle_Age", 10)

    # Neither Brake_Condition nor Vehicle_Age should be in legal question candidates
    candidates = q_guard.get_legal_question_candidates(state)
    assert "Brake_Condition" not in candidates
    assert "Vehicle_Age" not in candidates
    assert q_guard.validate_question("Brake_Condition", state) is False
    assert q_guard.validate_question("Vehicle_Age", state) is False
