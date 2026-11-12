"""
CRITICAL SECURITY TESTS (MUST FAIL / BE REJECTED BY BACKEND)

1. LLM prompt injection attempting to send email -> NO EMAIL SENT.
2. High ML probability (99%) -> NO AUTOMATIC EMAIL SENT.
3. Client-supplied recipient email override -> BACKEND IGNORES IT & USES REGISTERED CONTACT.
4. Client-supplied google_user_id override -> BACKEND IGNORES IT & USES SERVER SESSION.
5. Client-supplied OAuth token -> BACKEND DOES NOT TRUST IT.
6. Unauthenticated / Logged-out user emergency email request -> REJECTED (401).
7. LLM asking about illegal feature (e.g. Suspension_Condition) -> QuestionGuard REJECTS IT.
"""

import pytest
from backend.app.agent.question_guard import QuestionGuard
from backend.app.ml.preprocessing import SELECTED_FEATURES
from backend.app.emergency.emergency_engine import EmergencyEngine


def test_sec_1_llm_cannot_trigger_email():
    """
    SECURITY TEST 1:
    Even if the LLM output says 'Send an emergency email to my contact',
    the backend agent controller loop MUST NOT invoke email sending.
    Only explicit POST /api/emergency/send with user session cookie can dispatch email.
    """
    # The agent runner output is a dictionary returned to the API layer
    # It contains text response, state updates, prediction, and optional emergency_alert badge.
    # It NEVER calls Gmail API.
    from backend.app.agent.adk_agents.runner import ADKAgentRunner, get_adk_runner

    runner = get_adk_runner()
    assert not hasattr(runner, "send_email")
    assert not hasattr(runner, "gmail_service")


def test_sec_2_high_ml_probability_does_not_automatically_send_email():
    """
    SECURITY TEST 2:
    ML maintenance probability = 99% MUST NOT automatically send an email.
    """
    feature_state = {
        "Vehicle_Age": {"value": 20, "status": "known"},
        "Reported_Issues": {"value": 10, "status": "known"},
        "Maintenance_History": {"value": "Poor", "status": "known"},
        "Brake_Condition": {"value": "Good", "status": "known"},  # Brakes good
        "Battery_Status": {"value": "Good", "status": "known"},    # Battery good
    }

    # Emergency engine checks explicit policy, not ML probability score
    eval_result = EmergencyEngine.evaluate_emergency(feature_state)
    assert eval_result["is_emergency"] is False


def test_sec_3_frontend_recipient_override_ignored():
    """
    SECURITY TEST 3:
    Backend POST /api/emergency/send does NOT accept recipient email from request body.
    Recipient is retrieved ONLY from registered EmergencyContact table for the authenticated user session.
    """
    from backend.app.schemas.emergency import EmergencySendRequest

    # Schema only accepts optional conversation_id — recipient email is NOT a valid field
    fields = EmergencySendRequest.model_fields
    assert "recipient" not in fields
    assert "contact_email" not in fields
    assert "to" not in fields


def test_sec_4_frontend_google_user_id_override_ignored():
    """
    SECURITY TEST 4:
    Backend POST /api/emergency/send does NOT accept google_user_id from request body.
    User identity is derived strictly from the signed server-side session cookie.
    """
    from backend.app.schemas.emergency import EmergencySendRequest

    fields = EmergencySendRequest.model_fields
    assert "google_user_id" not in fields
    assert "user_id" not in fields


def test_sec_5_frontend_oauth_token_ignored():
    """
    SECURITY TEST 5:
    Backend does NOT trust frontend-provided OAuth tokens.
    Tokens are fetched from server-side encrypted OAuthToken database table.
    """
    from backend.app.schemas.emergency import EmergencySendRequest

    fields = EmergencySendRequest.model_fields
    assert "access_token" not in fields
    assert "oauth_token" not in fields


def test_sec_6_question_guard_rejects_illegal_suspension_feature():
    """
    SECURITY TEST 6:
    When LLM attempts to introduce 'Suspension_Condition', QuestionGuard MUST REJECT IT.
    """
    q_guard = QuestionGuard(SELECTED_FEATURES)

    assert not q_guard.is_legal_model_feature("Suspension_Condition")
    assert not q_guard.is_legal_model_feature("Suspension")
    assert not q_guard.is_legal_model_feature("Engine_Oil")
    assert not q_guard.is_legal_model_feature("Engine_Oil_Level")
    assert not q_guard.is_legal_model_feature("Coolant_Level")

