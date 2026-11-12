"""
ADKAgentRunner — Main orchestrator bridging FastAPI, Google ADK, and deterministic gates.

Pipeline:
USER MESSAGE
  ↓
DomainGuard (fast regex check)
  ↓
ContextManager / Compactor (rolling window preservation)
  ↓
InputAgent (Input LLM: explicit facts extraction & current_question resolution)
  ↓
FeatureTools (Deterministic schema & categorical validation -> updates AgentState)
  ↓
FeatureSelector (Deterministic get_next_question_feature -> single next question)
  ↓
PredictionTools (Deterministic programmatic ML gate: known_count >= 2 -> Random Forest)
  ↓
ResponseAgent (Output LLM: 4-6 lines grounded in validated state)
  ↓
JudgeAgent (Deterministic guards + LLM evaluation; failure/timeout -> deterministic safe fallback)
  ↓
DELIVER TO USER
"""

from __future__ import annotations

import asyncio
import json
import uuid
import re
from typing import Dict, Any, Optional, List, AsyncIterator

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from backend.app.agent.state.state import (
    AgentState,
    MIN_FEATURES_FOR_PREDICTION,
    CHECKPOINT_INPUT_RECEIVED,
    CHECKPOINT_INTENT_EXTRACTED,
    CHECKPOINT_FEATURES_EXTRACTED,
    CHECKPOINT_FEATURES_VALIDATED,
    CHECKPOINT_STATE_UPDATED,
    CHECKPOINT_FEATURES_RANKED,
    CHECKPOINT_ASSESSMENT_FEATURES_SELECTED,
    CHECKPOINT_QUESTION_SELECTED,
    CHECKPOINT_PREDICTION_GENERATED,
    CHECKPOINT_RESPONSE_GENERATED,
    CHECKPOINT_JUDGE_PASSED,
)
from backend.app.agent.state.feature_selector import (
    get_next_question_feature,
    get_question_specification,
    get_askable_features,
)
from backend.app.agent.tools.feature_tools import (
    apply_structured_extraction,
    constrain_extraction_to_explicit_facts,
    classify_summary_intent,
    classify_post_assessment_intent,
)
from backend.app.agent.tools.prediction_tools import execute_prediction_gate
from backend.app.agent.feature_ranking import feature_ranking_engine
from backend.app.agent.hypothesis_grounding import (
    normalise_hypotheses, ground_hypotheses, grounded_features,
)
from backend.app.agent.adk_agents.judge_agent import (
    run_deterministic_guards,
    generate_deterministic_fallback_response,
)
from backend.app.agent.domain_guard import DomainGuard
from backend.app.context.manager import ContextManager
from backend.app.ml.preprocessing import FEATURE_DISPLAY_NAMES, SELECTED_FEATURES
from backend.app.ml.explainer import get_local_explainer, explanation_for_ui
from backend.app.core.logging import get_logger
from backend.app.core.config import get_settings

logger = get_logger(__name__)

APP_NAME = "vehicle_maintenance_ai"


class ADKAgentRunner:
    """
    Coordinates turn-by-turn execution of the Google ADK multi-agent system.
    """

    def __init__(self):
        self.settings = get_settings()
        self.context_mgr = ContextManager()
        self._session_service = InMemorySessionService()

        self._controller_runner: Optional[Runner] = None
        self._information_runner: Optional[Runner] = None
        self._response_runner: Optional[Runner] = None
        self._judge_runner: Optional[Runner] = None
        self._initialized = False

    def initialize(self):
        """Initialize all ADK agents and runners."""
        if self._initialized:
            return

        logger.info("[ADK] Initializing Google ADK agents and runners...")

        from backend.app.agent.adk_agents.controller_agent import build_controller_agent
        from backend.app.agent.adk_agents.information_agent import build_information_agent
        from backend.app.agent.adk_agents.response_agent import build_response_agent
        from backend.app.agent.adk_agents.judge_agent import build_judge_agent

        controller_agent = build_controller_agent()
        information_agent = build_information_agent()
        response_agent = build_response_agent()
        judge_agent = build_judge_agent()

        self._controller_runner = Runner(
            agent=controller_agent,
            app_name=APP_NAME,
            session_service=self._session_service,
        )
        self._information_runner = Runner(
            agent=information_agent,
            app_name=APP_NAME + "_information",
            session_service=InMemorySessionService(),
        )
        self._response_runner = Runner(
            agent=response_agent,
            app_name=APP_NAME + "_response",
            session_service=InMemorySessionService(),
        )
        self._judge_runner = Runner(
            agent=judge_agent,
            app_name=APP_NAME + "_judge",
            session_service=InMemorySessionService(),
        )

        self._initialized = True
        logger.info("[ADK] Multi-Agent System Initialized: ControllerAgent, InformationAgent, ResponseAgent, JudgeAgent.")

    async def run_turn(
        self,
        user_message: str,
        conversation_id: str,
        agent_state: AgentState,
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Process a single conversation turn through the full zero-hallucination pipeline.
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"\n{'='*60}")
        logger.info(f"[ADK] TURN START | Conv: {conversation_id[:8]} | Turn: {agent_state.turn_count + 1}")
        agent_state.turn_count += 1
        agent_state.reset_turn_checkpoints()
        agent_state.record_checkpoint(CHECKPOINT_INPUT_RECEIVED)

        result: Dict[str, Any] = {
            "response": "",
            "prediction": None,
            "state_changed": False,
            "features_updated": [],
            "judge_result": None,
        }

        # -----------------------------------------------------------------------
        # Step 1: Domain Guard Check
        # -----------------------------------------------------------------------
        is_in_domain, redirect_msg = DomainGuard.is_in_domain(user_message)
        if not is_in_domain:
            logger.info(f"[ADK] DomainGuard: Intercepted off-topic query.")
            result["response"] = redirect_msg
            return result

        # -----------------------------------------------------------------------
        # POST-ASSESSMENT HANDLING (When assessment is already complete)
        # -----------------------------------------------------------------------
        if agent_state.assessment_complete:
            prev_mode = getattr(agent_state, "assessment_mode", "AWAITING_SUMMARY_CONFIRMATION")
            intent = classify_post_assessment_intent(user_message, prev_mode)
            logger.info("[POST_ASSESSMENT] prev_mode=%s intent=%s msg=%r", prev_mode, intent, user_message)

            if intent == "NEW_ASSESSMENT":
                agent_state.begin_new_assessment()
                next_feature = get_next_question_feature(agent_state)
                if next_feature:
                    agent_state.mark_asked(next_feature)
                    display = FEATURE_DISPLAY_NAMES.get(next_feature, next_feature.replace("_", " ").lower())
                    result["response"] = (
                        "Starting a new maintenance assessment. Previous assessment details will not be used for this one. "
                        f"Could you share the vehicle's {display}?"
                    )
                else:
                    result["response"] = "Starting a new maintenance assessment. Please share your vehicle details to begin."
                result["state_changed"] = True
                return result

            if intent == "SUMMARY_REQUEST":
                agent_state.assessment_mode = "SUMMARY"
            elif intent in ("FACTOR_EXPLANATION", "RISK_EXPLANATION"):
                agent_state.assessment_mode = "EXPLANATION"
            elif intent == "SUMMARY_DECLINE":
                agent_state.assessment_mode = "FOLLOW_UP"
            else:
                agent_state.assessment_mode = "FOLLOW_UP"

            latest_pred = agent_state.get_latest_prediction()
            result["prediction"] = (
                {
                    **latest_pred,
                    "assessment_complete": True,
                    "relevant_factor_count": agent_state.known_feature_count(),
                }
                if latest_pred else None
            )
            result["state_changed"] = False
            result["features_updated"] = []

            # Retrieve or compute local SHAP explanation for this prediction.
            # Use cached explanation if it matches the current feature-state signature;
            # otherwise (re)compute a fresh SHAP explanation.
            local_expl: Optional[Dict[str, Any]] = None
            if agent_state.has_current_explanation():
                local_expl = agent_state.get_latest_explanation()
                logger.info(
                    "[POST_ASSESSMENT] Using cached SHAP explanation. "
                    "available=%s  main_factor=%s",
                    (local_expl or {}).get("available"),
                    (local_expl or {}).get("main_factor", {}).get("feature") if local_expl else None,
                )
            else:
                logger.info(
                    "[POST_ASSESSMENT] No current SHAP explanation cached — computing fresh. "
                    "pred=%s  known_features=%s",
                    latest_pred.get("risk_level") if latest_pred else None,
                    list(agent_state.get_known_features().keys()),
                )
                explainer = get_local_explainer()
                logger.info(
                    "[POST_ASSESSMENT] Explainer: model=%s  has_tree=%s",
                    type(explainer.model).__name__ if explainer.model else "None",
                    explainer._tree_explainer is not None,
                )
                local_expl = explainer.explain_prediction(agent_state.feature_state, latest_pred)
                logger.info(
                    "[POST_ASSESSMENT] SHAP result: available=%s  reason=%s  main_factor=%s",
                    (local_expl or {}).get("available"),
                    (local_expl or {}).get("reason"),
                    (local_expl or {}).get("main_factor", {}).get("feature") if local_expl else None,
                )
                if local_expl and local_expl.get("available"):
                    agent_state.add_explanation(local_expl)
            result["local_explanation"] = local_expl
            result["explanation"] = explanation_for_ui(local_expl)


            # If user explicitly declined summary
            if intent == "SUMMARY_DECLINE":
                closing_msg = (
                    "No problem. The maintenance assessment is complete. If you'd like to review the result or "
                    "ask about any of the confirmed factors later, just let me know."
                )
                result["response"] = closing_msg
                return result

            # Run ResponseAgent for POST_ASSESSMENT turn
            response_context = {
                "user_message": user_message,
                "assessment_mode": agent_state.assessment_mode,
                "post_assessment_intent": intent,
                "known_features": agent_state.get_known_features(),
                "newly_confirmed": [],
                "unavailable_features": list(agent_state.declined_features),
                "prediction": latest_pred,
                "local_explanation": local_expl,
                "next_feature_to_ask": None,
                "next_feature_spec": None,
                "grounded_hypotheses": agent_state.grounded_hypotheses,
                "known_feature_count": agent_state.known_feature_count(),
                "prediction_available": bool(latest_pred),
                "assessment_complete": True,
                "remaining_relevant_features": [],
            }
            logger.info("[POST_ASSESSMENT] mode=%s context=%s", agent_state.assessment_mode, response_context)

            proposed_response = await self._run_response_agent(response_context)
            if not proposed_response or not proposed_response.strip():
                proposed_response = generate_deterministic_fallback_response(
                    agent_state, latest_pred, None, post_assessment_intent=intent, local_explanation=local_expl
                )

            is_guard_valid, violations = run_deterministic_guards(
                response_text=proposed_response,
                state=agent_state,
                prediction=latest_pred,
                expected_next_feature=None,
                post_assessment_intent=intent,
                local_explanation=local_expl,
            )

            final_response = proposed_response
            judge_result = None

            if not is_guard_valid:
                logger.warning(f"[JUDGE] Deterministic Guard REJECTED post-assessment response: {violations}")
                final_response = generate_deterministic_fallback_response(
                    agent_state, latest_pred, None, post_assessment_intent=intent, local_explanation=local_expl
                )
                judge_result = {
                    "score": 1,
                    "passed": False,
                    "hallucination_detected": True,
                    "issues": violations,
                    "reason": "Deterministic guard rejected response violations.",
                }
            else:
                judge_eval = await self._run_judge_agent(
                    user_message=user_message,
                    proposed_response=proposed_response,
                    state=agent_state,
                    prediction=latest_pred,
                    next_feature=None,
                    local_explanation=local_expl,
                )
                judge_result = judge_eval
                if not judge_eval.get("passed", False) or judge_eval.get("score", 0) < 3:
                    logger.warning(f"[JUDGE] JudgeAgent FAILED post-assessment response: {judge_eval.get('reason')}")
                    final_response = generate_deterministic_fallback_response(
                        agent_state, latest_pred, None, post_assessment_intent=intent, local_explanation=local_expl
                    )
                else:
                    agent_state.record_checkpoint(CHECKPOINT_JUDGE_PASSED)

            if agent_state.assessment_mode in ("SUMMARY", "EXPLANATION"):
                agent_state.assessment_mode = "FOLLOW_UP"

            result["response"] = final_response
            result["judge_result"] = judge_result
            return result

        # -----------------------------------------------------------------------
        # Step 2: Context Compaction Check
        # -----------------------------------------------------------------------
        logger.info("[CONTROLLER_START]")
        updated_summary, remaining_history = await self.context_mgr.maybe_compact(
            messages=conversation_history,
            current_summary=agent_state.compacted_summary,
            feature_state_snapshot=agent_state.to_context_dict(),
        )
        if updated_summary and updated_summary != agent_state.compacted_summary:
            agent_state.compacted_summary = updated_summary
            logger.info("[CONTEXT] Updated compacted conversation summary.")

        # -----------------------------------------------------------------------
        # Step 3: InputAgent Extraction (Input LLM)
        # -----------------------------------------------------------------------
        active_question_feature = agent_state.current_question_feature
        logger.info(f"[INPUT] Active question context: {active_question_feature}")
        logger.info("[TURN] raw_user_message = %r", user_message)
        logger.info("[STATE] before_update = %s", agent_state.get_known_features())

        input_context = {
            "user_message": user_message,
            "current_question_feature": active_question_feature,
            "known_features": agent_state.get_known_features(),
            "declined_features": list(agent_state.declined_features),
            "history_summary": updated_summary or agent_state.compacted_summary or "",
            "recent_history": remaining_history[-self.context_mgr.recent_messages_limit:],
        }

        logger.info("[INFORMATION_AGENT_START]")
        llm_extraction = await self._run_information_agent(input_context)
        logger.info("[INFORMATION_AGENT_COMPLETE]")

        extraction_data = constrain_extraction_to_explicit_facts(
            llm_extraction, user_message, active_question_feature
        )
        logger.info("[INPUT] llm_extraction = %s", llm_extraction)
        logger.info("[INPUT] deterministic_extracted_facts = %s", extraction_data.get("extracted_features", []))
        extraction_data["hypotheses"] = normalise_hypotheses(
            llm_extraction.get("hypotheses", []), user_message
        )
        if not extraction_data["hypotheses"] and agent_state.current_hypotheses:
            extraction_data["hypotheses"] = agent_state.current_hypotheses
        agent_state.record_checkpoint(CHECKPOINT_FEATURES_EXTRACTED)
        agent_state.record_checkpoint(CHECKPOINT_INTENT_EXTRACTED)

        # -----------------------------------------------------------------------
        # Step 4: Deterministic Feature Validation & State Update
        # -----------------------------------------------------------------------
        apply_summary = apply_structured_extraction(agent_state, extraction_data)
        state_changed = apply_summary["state_changed"]
        result["state_changed"] = state_changed
        result["features_updated"] = apply_summary["updated_features"]
        logger.info("[STATE] validated_facts = %s", apply_summary["updated_features"])
        logger.info("[STATE] after_update = %s", agent_state.get_known_features())

        logger.info(
            f"[STATE] Known features: {agent_state.known_feature_count()}/{len(SELECTED_FEATURES)} "
            f"| Updated: {apply_summary['updated_features']} | Declined: {apply_summary['declined_features']}"
        )
        agent_state.record_checkpoint(CHECKPOINT_FEATURES_VALIDATED)
        agent_state.record_checkpoint(CHECKPOINT_STATE_UPDATED)

        # -----------------------------------------------------------------------
        # Step 5: Dynamic, schema-bounded feature ranking and selection
        # -----------------------------------------------------------------------
        logger.info("[FEATURE_SELECTION_START]")
        extracted_intent = extraction_data.get("user_intent")
        intent = extracted_intent if extracted_intent not in (None, "inform", "greeting") else agent_state.current_intent
        if extraction_data.get("raw_observations"):
            agent_state.current_symptoms.extend(
                observation for observation in extraction_data["raw_observations"]
                if observation not in agent_state.current_symptoms
            )
        agent_state.current_hypotheses = extraction_data["hypotheses"]
        agent_state.grounded_hypotheses = ground_hypotheses(agent_state.current_hypotheses)
        grounded_candidates = grounded_features(agent_state.grounded_hypotheses)

        assessed_feature_count = agent_state.known_feature_count()
        assessment_target = 8
        target_reached = assessed_feature_count >= assessment_target

        all_unsupported = (
            bool(agent_state.grounded_hypotheses)
            and all(h.get("status") == "unsupported" for h in agent_state.grounded_hypotheses)
            and agent_state.known_feature_count() == 0
            and (not intent or intent not in feature_ranking_engine._INTENT_FEATURES)
        )

        if all_unsupported:
            selected_features = []
            agent_state.set_feature_subset([], intent)
            next_feature = None
            agent_state.current_question_feature = None
        else:
            ranked_features = feature_ranking_engine.rank_features(agent_state, intent, grounded_candidates)
            selected_features = feature_ranking_engine.select_feature_subset(
                agent_state, intent, target_size=9, candidates=grounded_candidates
            )
            agent_state.set_feature_subset(selected_features, intent)
            agent_state.record_checkpoint(CHECKPOINT_FEATURES_RANKED)
            agent_state.record_checkpoint(CHECKPOINT_ASSESSMENT_FEATURES_SELECTED)

            agent_state.remaining_relevant_features = get_askable_features(agent_state)
            next_feature = None if target_reached else get_next_question_feature(agent_state)

        next_spec = get_question_specification(next_feature)
        if next_feature:
            agent_state.mark_asked(next_feature)
            agent_state.record_checkpoint(CHECKPOINT_QUESTION_SELECTED)
            agent_state.assessment_complete = False
            agent_state.assessment_mode = "ASSESSMENT"
            agent_state.assessment_completion_reason = None
            logger.info(f"[QUESTION] Deterministically selected next question: {next_feature}")
        else:
            agent_state.assessment_complete = True
            agent_state.assessment_mode = "AWAITING_SUMMARY_CONFIRMATION"
            agent_state.current_question_feature = None
            agent_state.remaining_relevant_features = []
            if not all_unsupported:
                agent_state.assessment_completion_reason = "Assessment complete — confirmed target features reached."
            else:
                agent_state.assessment_completion_reason = None
            logger.info("[QUESTION] Assessment complete or unsupported turn.")
        logger.info("[FEATURE_SELECTION_COMPLETE]")

        # -----------------------------------------------------------------------
        # Step 7: Deterministic Prediction Gate (known_count >= 2)
        # -----------------------------------------------------------------------
        logger.info("[ML_PREDICTION_START]")
        should_predict = bool(apply_summary["updated_features"]) or (
            agent_state.has_enough_for_prediction() and not agent_state.has_current_prediction()
        )
        pred_gate_result = execute_prediction_gate(agent_state) if should_predict else {
            "allowed": False, "prediction": None, "reason": "No validated feature change."
        }
        if pred_gate_result["allowed"] and pred_gate_result["prediction"]:
            result["prediction"] = pred_gate_result["prediction"]
            agent_state.record_checkpoint(CHECKPOINT_PREDICTION_GENERATED)
            logger.info(
                f"[PREDICTION] Real ML output: {result['prediction'].get('prediction')} "
                f"| Risk: {result['prediction'].get('risk_level')} "
                f"| Score: {result['prediction'].get('model_score')}"
            )
            logger.info("[PREDICTION] current_ml_input = %s", agent_state.get_known_features())
        else:
            if agent_state.has_enough_for_prediction():
                result["prediction"] = agent_state.get_latest_prediction()
            else:
                result["prediction"] = None
                logger.info(f"[PREDICTION] Blocked: only {agent_state.known_feature_count()} feature(s) known (need {MIN_FEATURES_FOR_PREDICTION}).")
        logger.info("[ML_PREDICTION_COMPLETE]")

        prediction_available = bool(result.get("prediction"))
        prediction_is_current = agent_state.has_current_prediction()

        # Compute real local SHAP explanation for EVERY turn where a prediction exists.
        # Uses the predictor's own explainer (same model instance as prediction) to
        # guarantee prediction and explanation are always from the same model input.
        local_expl: Optional[Dict[str, Any]] = None
        if result.get("prediction"):
            if agent_state.assessment_complete:
                result["prediction"] = {
                    **result["prediction"],
                    "assessment_complete": True,
                    "relevant_factor_count": agent_state.known_feature_count(),
                }
            logger.info("[SHAP_START]")
            try:
                explainer = get_local_explainer()
                local_expl = explainer.explain_prediction(
                    agent_state.feature_state, result["prediction"]
                )
                if local_expl and local_expl.get("available"):
                    agent_state.add_explanation(local_expl)
            except Exception as e:
                logger.exception(f"[SHAP_FAIL] SHAP explanation failed safely: {e}")
                local_expl = {"available": False, "reason": str(e)}
            logger.info("[SHAP_COMPLETE]")

            result["local_explanation"] = local_expl
            result["explanation"] = explanation_for_ui(local_expl)

        remaining_candidates = list(agent_state.remaining_relevant_features)
        logger.info("[ASSESSMENT] assessed_features = %s", list(agent_state.get_known_features()))
        logger.info("[PREDICTION] prediction_available = %s", prediction_available)
        logger.info("[PREDICTION] prediction_is_current = %s", prediction_is_current)
        logger.info("[QUESTIONS] remaining_candidates = %s", remaining_candidates)
        logger.info("[NEXT_QUESTION] selected = %s", next_feature)
        logger.info(
            "[COMPLETION] assessment_complete = %s reason = %s mode = %s",
            agent_state.assessment_complete, agent_state.assessment_completion_reason, agent_state.assessment_mode
        )

        # -----------------------------------------------------------------------
        # Step 7: Emergency Engine Check
        # -----------------------------------------------------------------------
        try:
            from backend.app.emergency.emergency_engine import EmergencyEngine
            emergency_eval = EmergencyEngine.evaluate_emergency(agent_state.feature_state)
            if emergency_eval.get("is_emergency"):
                result["emergency_alert"] = emergency_eval
                logger.info(f"[EMERGENCY] Emergency trigger: {emergency_eval.get('severity')}")
        except Exception as e:
            logger.debug(f"Emergency engine check skipped: {e}")

        # -----------------------------------------------------------------------
        # Step 8: ResponseAgent Generation (Output LLM)
        # -----------------------------------------------------------------------
        logger.info("[OUTPUT_LLM_START]")
        explanation_intent = classify_post_assessment_intent(user_message, agent_state.assessment_mode)

        response_context = {
            "user_message": user_message,
            "assessment_mode": agent_state.assessment_mode,
            "post_assessment_intent": explanation_intent,
            "known_features": agent_state.get_known_features(),
            "newly_confirmed": apply_summary["updated_features"],
            "unavailable_features": list(agent_state.declined_features),
            "prediction": result.get("prediction"),
            "local_explanation": local_expl,
            "next_feature_to_ask": next_feature,
            "next_feature_spec": next_spec,
            "grounded_hypotheses": agent_state.grounded_hypotheses,
            "known_feature_count": agent_state.known_feature_count(),
            "prediction_available": prediction_available,
            "assessment_complete": agent_state.assessment_complete,
            "remaining_relevant_features": remaining_candidates,
        }
        logger.info("[RESPONSE] state_signature = %s", agent_state.feature_state_signature())
        logger.info("[RESPONSE] context = %s", response_context)

        proposed_response = await self._run_response_agent(response_context)
        logger.info("[OUTPUT_LLM_COMPLETE]")

        if not proposed_response or not proposed_response.strip():
            logger.warning("[RUNNER] Proposed response is empty — falling back to deterministic response.")
            proposed_response = generate_deterministic_fallback_response(
                agent_state, result.get("prediction"), next_feature,
                post_assessment_intent=explanation_intent if explanation_intent in ("FACTOR_EXPLANATION", "RISK_EXPLANATION") else None,
                local_explanation=local_expl,
            )
        agent_state.record_checkpoint(CHECKPOINT_RESPONSE_GENERATED)

        # -----------------------------------------------------------------------
        # Step 9: JudgeAgent (Deterministic Guards + LLM Evaluation)
        # -----------------------------------------------------------------------
        logger.info("[JUDGE_START]")
        is_guard_valid, violations = run_deterministic_guards(
            response_text=proposed_response,
            state=agent_state,
            prediction=result.get("prediction"),
            expected_next_feature=next_feature,
            post_assessment_intent=explanation_intent if explanation_intent in ("FACTOR_EXPLANATION", "RISK_EXPLANATION") else None,
            local_explanation=local_expl,
        )

        final_response = proposed_response
        judge_result = None

        if not is_guard_valid:
            logger.warning(f"[JUDGE] Deterministic Guard REJECTED response: {violations}")
            final_response = generate_deterministic_fallback_response(
                agent_state, result.get("prediction"), next_feature,
                post_assessment_intent=explanation_intent if explanation_intent in ("FACTOR_EXPLANATION", "RISK_EXPLANATION") else None,
                local_explanation=local_expl,
            )
            judge_result = {
                "score": 1,
                "passed": False,
                "hallucination_detected": True,
                "issues": violations,
                "reason": "Deterministic guard rejected response violations.",
            }
        else:
            # 9b. LLM Judge Evaluation (Blocking)
            judge_eval = await self._run_judge_agent(
                user_message=user_message,
                proposed_response=proposed_response,
                state=agent_state,
                prediction=result.get("prediction"),
                next_feature=next_feature,
                local_explanation=local_expl,
            )
            judge_result = judge_eval

            # CRITICAL RULE: If judge fails or score < 3, NEVER default to pass
            if not judge_eval.get("passed", False) or judge_eval.get("score", 0) < 3:
                logger.warning(f"[JUDGE] JudgeAgent FAILED response: {judge_eval.get('reason')}")
                final_response = generate_deterministic_fallback_response(
                    agent_state, result.get("prediction"), next_feature,
                    post_assessment_intent=explanation_intent if explanation_intent in ("FACTOR_EXPLANATION", "RISK_EXPLANATION") else None,
                    local_explanation=local_expl,
                )
            else:
                agent_state.record_checkpoint(CHECKPOINT_JUDGE_PASSED)
                logger.info(f"[JUDGE] JudgeAgent PASSED (Score: {judge_eval.get('score', 5)})")
        logger.info("[JUDGE_COMPLETE]")

        logger.info("[RESPONSE_BUILD_START]")
        result["response"] = final_response
        result["judge_result"] = judge_result
        logger.info("[RESPONSE_BUILD_COMPLETE]")
        logger.info("[CONTROLLER_COMPLETE]")
        return result

    async def run_turn_stream(
        self,
        user_message: str,
        conversation_id: str,
        agent_state: AgentState,
        conversation_history: List[Dict[str, str]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream-compatible turn execution.
        Runs full turn through ADK, yields SSE events for state update, prediction, and tokens.
        """
        result = await self.run_turn(
            user_message=user_message,
            conversation_id=conversation_id,
            agent_state=agent_state,
            conversation_history=conversation_history,
        )

        # Emit state update
        if result.get("state_changed"):
            yield {"type": "state_update", "data": agent_state.to_context_dict()}

        # Emit prediction
        if result.get("prediction"):
            yield {"type": "prediction", "data": result["prediction"]}

        if result.get("explanation"):
            yield {"type": "explanation", "data": result["explanation"]}

        # Emit emergency alert if present
        if result.get("emergency_alert"):
            yield {"type": "emergency_alert", "data": result["emergency_alert"]}

        # Stream response text
        response_text = result.get("response", "")
        if response_text:
            words = response_text.split(" ")
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield {"type": "text", "content": chunk}
                await asyncio.sleep(0.01)

        yield {"type": "done"}

    # ---------------------------------------------------------------------------
    # Specialist Agent ADK Invocations
    # ---------------------------------------------------------------------------

    async def _run_input_agent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible delegate for _run_information_agent."""
        return await self._run_information_agent_impl(context)

    async def _run_information_agent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run InformationAgent (Input LLM) to extract structured facts."""
        if "_run_input_agent" in self.__dict__:
            return await self._run_input_agent(context)
        return await self._run_information_agent_impl(context)

    async def _run_information_agent_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = f"information_{uuid.uuid4().hex[:8]}"
        try:
            await self._information_runner.session_service.create_session(
                app_name=APP_NAME + "_information",
                user_id="user",
                session_id=session_id,
                state={},
            )
        except Exception:
            pass

        adk_input = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=json.dumps(context, indent=2))],
        )

        try:
            raw_output = await asyncio.wait_for(
                self._collect_agent_response(self._information_runner, session_id, adk_input),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[INFORMATION] InformationAgent timed out after 15s — falling back to deterministic extraction")
            raw_output = ""
        except Exception as e:
            logger.exception(f"[INFORMATION] InformationAgent exception: {e}")
            raw_output = ""

        return self._parse_json_safely(raw_output, default={
            "extracted_features": [],
            "unavailable_features": [],
            "raw_observations": [],
            "hypotheses": [],
            "user_intent": "inform",
        })

    async def _run_response_agent(self, context: Dict[str, Any]) -> str:
        """Run ResponseAgent (Output LLM) to generate conversational response."""
        session_id = f"response_{uuid.uuid4().hex[:8]}"
        try:
            await self._response_runner.session_service.create_session(
                app_name=APP_NAME + "_response",
                user_id="user",
                session_id=session_id,
                state={},
            )
        except Exception:
            pass

        adk_input = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=json.dumps(context, indent=2))],
        )

        try:
            raw_output = await asyncio.wait_for(
                self._collect_agent_response(self._response_runner, session_id, adk_input),
                timeout=18.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[RESPONSE] ResponseAgent timed out after 18s — falling back to deterministic response")
            raw_output = ""
        except Exception as e:
            logger.exception(f"[RESPONSE] ResponseAgent exception: {e}")
            raw_output = ""

        cleaned = raw_output.strip()

        # If wrapped in JSON, extract string
        if cleaned.startswith("{") and "response" in cleaned:
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "response" in parsed:
                    return str(parsed["response"]).strip()
            except Exception:
                pass

        return cleaned if cleaned else ""

    async def _run_judge_agent(
        self,
        user_message: str,
        proposed_response: str,
        state: AgentState,
        prediction: Optional[Dict[str, Any]],
        next_feature: Optional[str],
        local_explanation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run JudgeAgent to evaluate response grounding."""
        default_fail = {
            "score": 1,
            "passed": False,
            "hallucination_detected": True,
            "issues": ["Judge evaluation failed or timed out."],
            "reason": "Judge failed to complete evaluation.",
        }

        judge_context = {
            "user_message": user_message,
            "proposed_response": proposed_response,
            "known_features": state.get_known_features(),
            "declined_features": list(state.declined_features),
            "prediction": prediction,
            "local_explanation": local_explanation,
            "next_feature_to_ask": next_feature,
            "known_feature_count": state.known_feature_count(),
        }

        session_id = f"judge_{uuid.uuid4().hex[:8]}"
        try:
            await self._judge_runner.session_service.create_session(
                app_name=APP_NAME + "_judge",
                user_id="user",
                session_id=session_id,
                state={},
            )
        except Exception:
            pass

        adk_input = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=json.dumps(judge_context, indent=2))],
        )

        try:
            raw_output = await asyncio.wait_for(
                self._collect_agent_response(self._judge_runner, session_id, adk_input),
                timeout=15.0,
            )
            parsed = self._parse_json_safely(raw_output, default=default_fail)
            score = parsed.get("score", 1)
            parsed["passed"] = score >= 3
            return parsed
        except asyncio.TimeoutError:
            logger.warning("[JUDGE] JudgeAgent timed out — enforcing safe fallback response.")
            return default_fail
        except Exception as e:
            logger.error(f"[JUDGE] JudgeAgent execution error: {e}")
            return default_fail

    async def _collect_agent_response(
        self,
        runner: Runner,
        session_id: str,
        new_message: genai_types.Content,
    ) -> str:
        """Run an ADK Runner asynchronously and collect response text."""
        parts: List[str] = []
        try:
            async for event in runner.run_async(
                user_id="user",
                session_id=session_id,
                new_message=new_message,
            ):
                if event.is_final_response():
                    content = event.content
                    if content and content.parts:
                        for p in content.parts:
                            if p.text:
                                parts.append(p.text)
        except Exception as e:
            logger.error(f"[ADK] Error running agent: {e}", exc_info=True)

        return "".join(parts)

    def _parse_json_safely(self, text: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Safely parse JSON with markdown code-block extraction support."""
        if not text:
            return default

        cleaned = text.strip()
        # Strip markdown ```json ... ```
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if m:
            cleaned = m.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try to find outermost braces
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return default


# ---------------------------------------------------------------------------
# Singleton Accessor
# ---------------------------------------------------------------------------
_adk_runner: Optional[ADKAgentRunner] = None


def get_adk_runner() -> ADKAgentRunner:
    """Get or create the singleton ADKAgentRunner."""
    global _adk_runner
    if _adk_runner is None:
        _adk_runner = ADKAgentRunner()
    return _adk_runner


def initialize_adk_runner():
    """Initialize the runner at application startup."""
    runner = get_adk_runner()
    runner.initialize()
    return runner
