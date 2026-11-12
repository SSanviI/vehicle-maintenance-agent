# Vehicle Maintenance Agentic AI – Final Refactor Implementation Plan

**Goal**: Refactor the existing repository into a clean, hallucination‑prevention‑focused multi‑agent system that truly uses Google ADK primitives.

---

## 1. Repository Audit & Cleanup
- Walk through all files under `backend/app/agent/` and related modules to identify duplicate orchestration paths, legacy agents, and inconsistent state handling.
- Delete dead artifacts: `.venv`, `__pycache__`, `.pytest_cache`, experimental notebooks, unused scripts, obsolete model bridges.
- Restructure the package layout to match the target architecture (see section *Clean Backend Structure*).

---

## 2. Core ADK Architecture
| Component | Action |
|-----------|--------|
| **Runner** | Implement a true ADK `Runner` that creates an ADK session and holds the shared `AgentState`.
| **ControllerAgent** | Root `LlmAgent`; orchestrates sub‑agents via ADK `AgentTool` calls.
| **InputAgent** | Uses the **Input LLM** to extract only explicit feature/value pairs (or an explicit "I don't know").
| **FeatureSelector** | Deterministic function `get_next_question_feature(state)` that excludes known, declined, already‑asked, and unsupported features.
| **PredictionAgent** | Pure deterministic wrapper around the pre‑trained Random Forest model (`ml/predictor.py`). No LLM reasoning.
| **ResponseAgent** | Uses the **Output LLM** to format a 4‑6 line response from structured data.
| **JudgeAgent** | Performs deterministic validation of the response (feature list, values, prediction match, single‑question rule). Falls back to a safe deterministic reply on failure.

---

## 3. State Management Consolidation
Create a single `AgentState` dataclass (`backend/app/agent/state/state.py`):
```python
@dataclass
class AgentState:
    feature_state: Dict[str, Any]
    asked_features: Set[str]
    declined_features: Set[str]
    current_question_feature: Optional[str]
    predictions: List[Dict]
    compacted_summary: str
    turn_count: int
    checkpoint_log: List[str]
```
Remove all previous JSON‑string serializations and ensure the ADK session synchronises with this object after every sub‑agent execution.

---

## 4. Model Bridge Removal & Prediction Tool
- Delete `backend/app/agent/adk_agents/model_bridge.py`.
- Add `backend/app/agent/tools/prediction_tool.py` that loads the Random Forest model once at startup and returns a structured prediction dict:
```json
{"risk_score": <float>, "risk_level": "Low|Medium|High", "details": {...}}
```
Register this tool with ADK so `PredictionAgent` can invoke it directly.

---

## 5. Two‑LLM Roles
- **Input LLM**: Prompt engineered to output JSON with only explicit features, values, or an "unknown" flag. No invented values.
- **Output LLM**: Prompt engineered to produce ~4‑6 lines, acknowledge known info, present the latest prediction, and ask exactly one next question.
Both clients are created via `backend/app/llm/factory.py` using the selected provider (GPT‑OSS 120B).

---

## 6. Deterministic Feature Selector
Implement `backend/app/agent/state/feature_selector.py` exposing:
```python
def get_next_question_feature(state: AgentState) -> Optional[str]:
    # Load schema from ml/feature_schema.py, apply priority, filter out:
    #   - state.feature_state keys (known)
    #   - state.declined_features
    #   - state.current_question_feature (if awaiting answer)
    #   - any feature not present in the schema
    # Return the highest‑priority remaining feature or None.
```
Guarantees no repeated or unsupported questions.

---

## 7. Minimum‑Feature Gate & Re‑prediction
- In `PredictionAgent`, before calling the prediction tool, check `len(state.feature_state) >= 2`. If the condition is not met, skip prediction.
- After each successful feature update, automatically invoke the prediction tool and store the result in `state.predictions`.

---

## 8. Judge Agent
- Perform deterministic checks:
  * Every feature mentioned in the response exists in the schema.
  * Every value matches `state.feature_state`.
  * The risk score/level equals the latest ML prediction.
  * The asked question equals `state.current_question_feature`.
  * Only one question is present.
  * Response length ≤ 6 lines.
- If any check fails, return a safe fallback response (e.g., "I’m sorry, I need more information before I can continue.") and do **not** forward the unverified LLM output.

---

## 9. Rolling Context & Compactor
- `backend/app/context/rolling_window.py` builds the prompt for the Output LLM from recent messages, `state.compacted_summary`, serialized `state`, and the current user message.
- `backend/app/context/compactor.py` periodically consolidates older turns, preserving confirmed features, declined features, current question, latest prediction, and important user statements.

---

## 10. Persistence & Checkpoints
- Extend `ConversationService` to persist the full `AgentState` (JSON) after each successful turn in a SQLite DB (`conversation.db`).
- Store a checkpoint enum (`INPUT_RECEIVED`, `FEATURES_EXTRACTED`, `FEATURES_VALIDATED`, `PREDICTION_GENERATED`, `RESPONSE_GENERATED`, `JUDGE_PASSED`).
- On conversation resume, load the latest state and resume from the last checkpoint.

---

## 11. Tests (Hallucination Matrix)
Create `backend/tests/test_hallucination.py` covering scenarios A–L from the specification. Use fixtures to spin up the FastAPI app, mock LLM responses, and assert that:
- No unsupported features are added.
- No hallucinated values appear.
- Repeated questions are never asked.
- Predictions occur only after two valid features.
- Judge failures produce the safe fallback response.

---

## 12. Front‑end Integration
- Verify FastAPI endpoint `/conversation` forwards the raw user message to the ADK runner and returns the validated response JSON.
- Front‑end simply displays the `text` field.

---

## Verification Plan
1. Run the full test suite (`pytest -q`). All hallucination tests must pass.
2. Manually start the FastAPI server (`uvicorn backend.app.main:app --reload`) and perform a short conversation using `scripts/test_agent_full_flow.py`.
3. Confirm:
   - Only one question per turn.
   - No repeated questions.
   - Predictions appear only after ≥2 valid features.
   - Judge never lets a hallucinated response through.
4. Profile model loading to ensure it occurs only once.

---

**User Review Required**
- Please confirm that this plan and the proposed directory restructuring meet your expectations, or let me know any adjustments.
- Once approved, I will begin applying the changes.

---

*Reply with “Approved” to start the implementation.*
