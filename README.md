# 🚗 Vehicle Maintenance AI — Google ADK Multi-Agent System

A production-grade, full-stack **Vehicle Maintenance Multi-Agent System** powered by **Google Agent Development Kit (ADK)**. Combines genuine ADK multi-agent orchestration, pure ML maintenance prediction authority, dynamic feature questioning, context compaction, LLM-as-a-Judge evaluation, multi-LLM router failover, PostgreSQL/SQLite persistence, Redis caching, Google OAuth 2.0 emergency dispatch, and a ChatGPT-style React frontend.

> **CRITICAL ARCHITECTURAL GUARANTEE**: The LLM is **NEVER** the prediction authority. The trained ML model is the sole authority for maintenance predictions. The ADK `PredictionAgent` invokes the ML engine via an ADK `FunctionTool`.

---

## 🏗️ Multi-Agent Architecture (Google ADK)

The orchestration layer is built using real **Google Agent Development Kit (ADK)** primitives (`LlmAgent`, `FunctionTool`, `AgentTool`, `Runner`, `InMemorySessionService`):

```text
                           USER MESSAGE (FastAPI /api/chat)
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │           ADKAgentRunner            │
                       │     (Context Compaction Bridge)     │
                       └──────────────────┬──────────────────┘
                                          │ ADK Runner
                                          ▼
                       ┌─────────────────────────────────────┐
                       │           ControllerAgent           │
                       │     (Root ADK Orchestrator)         │
                       └─────────┬────────┬────────┬─────────┘
                                 │        │        │
            ┌────────────────────┘        │        └────────────────────┐
            ▼                             ▼                             ▼
 ┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
 │  InformationAgent   │       │   PredictionAgent   │       │    ResponseAgent    │
 │ (ADK Feature Extr.) │       │  (ADK ML Delegation)│       │(ADK Service Advisor)│
 └──────────┬──────────┘       └──────────┬──────────┘       └──────────┬──────────┘
            │                             │                             │
            ▼                             ▼                             ▼
    [Schema Validation]           [ML Engine Authority]         [Final Conversational]
   [State & Unavailable]         [Single Source of Truth]        [Natural Response]
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │     JudgeAgent      │
                               │(Post-Turn Evaluator)│
                               └─────────────────────┘
                                          │
                                          ▼
                            FINAL USER-FACING RESPONSE
```

---

## 🤖 Google ADK Agent Roles & Responsibilities

### 1. Controller Agent (`backend/app/agent/adk_agents/controller_agent.py`)
- **Type**: Root `LlmAgent`.
- **Orchestration**: Contains `InformationAgent`, `PredictionAgent`, and `ResponseAgent` registered via ADK `AgentTool`.
- **Responsibilities**:
  - Performs initial closed-domain verification via `check_domain`.
  - Coordinates feature extraction via `InformationAgent`.
  - Determines when ML prediction should occur (new info, explicit request, or all features exhausted).
  - Coordinates conversational response generation via `ResponseAgent`.
  - Never computes predictions or invents features.

### 2. Information / Feature Agent (`backend/app/agent/adk_agents/information_agent.py`)
- **Type**: Specialist `LlmAgent`.
- **ADK Tools**: `get_model_features`, `validate_feature`, `update_feature_state`, `mark_feature_unavailable`, `get_next_feature_to_ask`.
- **Responsibilities**:
  - Dynamically inspects `models/feature_schema.json` as the authoritative source of truth.
  - Extracts only explicitly provided feature values.
  - Maps natural language phrases to exact categorical and numerical values.
  - Rejects hallucinated features (e.g., `Engine_Oil_Level`, `Mileage`) via `validate_feature`.
  - Handles natural *"I don't know"* variants by marking features as `unavailable` (never asked again).
  - Determines the next most useful feature to query without hardcoded question counts.

### 3. ML Prediction Agent (`backend/app/agent/adk_agents/prediction_agent.py`)
- **Type**: Specialist `LlmAgent`.
- **ADK Tools**: `run_ml_prediction`.
- **Responsibilities**:
  - Solely invokes the authoritative ML prediction engine (`backend/app/ml/predict.py`).
  - Generates a unique `prediction_id` per turn.
  - Returns probability, risk level (`HIGH` / `LOW`), features used/missing, and confidence metadata.
  - The LLM never computes or alters model scores.

### 4. Response Agent (`backend/app/agent/adk_agents/response_agent.py`)
- **Type**: Specialist `LlmAgent`.
- **ADK Tools**: `get_current_state_summary`, `get_next_feature_to_ask`, `get_model_features`.
- **Responsibilities**:
  - Formulates natural, conversational responses acting as AutoCare AI service advisor.
  - Naturally phrases single feature questions derived dynamically from `models/feature_schema.json`.
  - Explains predictions without revealing raw numeric model scores (handled by UI cards).
  - Never asks about unavailable or already known features.

### 5. Judge Agent (`backend/app/agent/adk_agents/judge_agent.py`)
- **Type**: Independent Evaluator `LlmAgent`.
- **ADK Tools**: `get_model_features`.
- **Responsibilities**:
  - Evaluates the final response and agent execution trace post-turn.
  - Checks for hallucinations, invalid feature questions, repeated unavailable questions, and ML consistency.
  - Emits structured evaluation JSON (`score: 1-5`, `passed: bool`, `hallucination: bool`, `reason: str`).
  - Operates as a read-only auditor without modifying conversation or user state.

---

## 📊 Dataset & ML Pipeline

### Dataset Overview
- **File**: `data/vehicle_maintenance_data.csv`
- **Size**: 50,000 rows, 20 columns
- **Target**: `Need_Maintenance` (Binary: 0 = No, 1 = Yes)

### Exploratory Data Analysis & Feature Selection
1. **Selected Model Features (`models/feature_schema.json`)**:
   - `Vehicle_Age` (Numerical)
   - `Reported_Issues` (Numerical)
   - `Maintenance_History` (Categorical: Good, Average, Poor)
   - `Service_History` (Numerical)
   - `Accident_History` (Numerical)
   - `Tire_Condition` (Categorical: New, Good, Worn Out)
   - `Brake_Condition` (Categorical: New, Good, Worn Out)
   - `Battery_Status` (Categorical: New, Good, Weak)
2. **Missing-Feature Strategy & Partial Inputs**:
   - Categorical handling: `UNKNOWN` representation.
   - Numerical handling: `-1` sentinel value + binary `_missing` indicator columns.
   - Incremental predictions: Recalculates dynamically as new features are provided.

### Model Performance
| Classifier | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 84.08% | 96.20% | 83.65% | 0.8949 | 0.9337 |
| Random Forest | 98.21% | 100.00% | 97.79% | 0.9888 | 1.0000 |
| **XGBoost (Selected)** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** |
| LightGBM | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 |

---

## 🧠 Context Management & Compaction

- **Rolling Context Window**: Configurable recent message buffer (`max_recent_messages = 20`).
- **Context Compactor (`backend/app/context/compactor.py`)**:
  - Automatically activates when turn count exceeds threshold (`compaction_threshold = 30`).
  - Condenses older turns into a structured summary (vehicle identity, known features, unavailable features, predictions, key symptoms).
  - Enforces `STRUCTURED STATE > COMPACTED SUMMARY` precedence.

---

## 🔀 Multi-LLM Provider Router & ADK Model Bridge

- **`RouterBackedModel` (`backend/app/agent/adk_agents/model_bridge.py`)**: Custom ADK `BaseLlm` bridge that translates ADK `LlmRequest` to the project's multi-provider router.
- **Failover Chain**: Primary (Groq/OpenAI) → Secondary (Fallback) → Tertiary provider.
- **Circuit Breaker**: Exponential backoff on rate limits (HTTP 429), 5-minute circuit breaker after 3 failures.

---

## 🔐 Google OAuth 2.0 & Emergency Gmail Dispatch

- **Google OAuth 2.0 PKCE**: Secure Google Sign-In with server-side Fernet-encrypted token storage.
- **Emergency Engine (`backend/app/emergency/emergency_engine.py`)**:
  - Detects critical conditions (`Brake_Condition = 'Worn Out'`, `Battery_Status = 'Weak'`).
  - Renders an Emergency Alert modal in the frontend.
  - Dispatches emergency alert emails to saved emergency contacts via the user's Gmail API credentials.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm

### 1. Backend Setup

```bash
# Clone repository and enter project root
cd vehicle_maintainence_agentic

# Configure environment variables
cp .env.example .env
# Edit .env with your LLM API Key(s) (e.g., Groq / OpenAI) and Google OAuth credentials

# Install dependencies (from root or backend directory)
pip install -r backend/requirements.txt
# (or from root: pip install -r requirements.txt)

# Option A: From project root (Recommended)
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 7860

# Option B: From inside the backend directory
# (PowerShell)
$env:PYTHONPATH=".."
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 7860
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Comprehensive Test Suite

Run the full automated test suite (53/53 tests passing) covering Google ADK agents, OAuth PKCE, Token Encryption, Emergency Engine, Gmail Dispatch, ML prediction authority, and Security:

```bash
# Run all tests
python -m pytest backend/tests/ -v

# Run Google ADK Multi-Agent tests only
python -m pytest backend/tests/test_adk_agents.py -v
```

### Test Suite Summary (53 Passed):
- **Google ADK Multi-Agent Suite (`test_adk_agents.py`)**:
  - ADK agent structure verification (`ControllerAgent`, `InformationAgent`, `PredictionAgent`, `ResponseAgent`, `JudgeAgent`).
  - Multi-feature extraction and structured state updates.
  - Natural *"I don't know"* handling (`unavailable` status, never asked again).
  - Dynamic adaptive questioning without hardcoded counts.
  - Invalid feature hallucination rejection (non-schema features like `Engine_Oil_Level`).
  - Pure ML prediction authority via ADK tool.
  - Context compaction & rolling window verification.
  - Closed-domain guard validation.
- **Security & OAuth Suite (`test_security.py`, `test_oauth.py`, `test_token_manager.py`)**:
  - OAuth 2.0 PKCE, Fernet encryption at rest, token refresh.
  - Prompt injection immunity, user isolation, emergency contact CRUD.
- **ML & Consistency Suite (`test_ml.py`, `test_prediction_consistency.py`)**:
  - Single ML prediction per turn, probability score consistency.
- **Emergency Flow Suite (`test_emergency_engine.py`, `test_gmail_service.py`, `test_emergency_flow.py`)**:
  - Explicit rule-based emergency triggering and Gmail dispatch.

---

## 📁 Repository Structure

```text
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── adk_agents/          # Google ADK Multi-Agent Implementation
│   │   │   │   ├── controller_agent.py # Root ControllerAgent (LlmAgent + AgentTools)
│   │   │   │   ├── information_agent.py# InformationAgent (Feature extraction & validation)
│   │   │   │   ├── prediction_agent.py # PredictionAgent (ML prediction delegation)
│   │   │   │   ├── response_agent.py   # ResponseAgent (Conversational service advisor)
│   │   │   │   ├── judge_agent.py      # JudgeAgent (Post-turn evaluator)
│   │   │   │   ├── model_bridge.py     # RouterBackedModel (ADK BaseLlm bridge)
│   │   │   │   ├── tools.py            # ADK FunctionTool functions
│   │   │   │   └── runner.py           # ADKAgentRunner (Session & turn orchestrator)
│   │   │   ├── state.py                # AgentState (Structured vehicle & turn state)
│   │   │   ├── domain_guard.py         # Closed-domain boundary validator
│   │   │   └── question_guard.py       # Hard feature boundary validator
│   │   ├── api/                        # FastAPI REST & SSE Streaming Endpoints
│   │   ├── context/                    # ContextManager & ContextCompactor
│   │   ├── emergency/                  # Emergency Engine & Gmail Dispatch
│   │   ├── llm/                        # Multi-LLM Router & Providers
│   │   ├── ml/                         # ML Model Predictor & Preprocessing Pipeline
│   │   └── main.py                     # FastAPI Application Entry Point
│   ├── tests/                          # 53 Automated Tests
│   └── requirements.txt                # Python Backend Dependencies
├── frontend/                           # React + Vite + TypeScript Frontend
├── models/
│   ├── feature_schema.json             # Authoritative Model Feature Definition
│   └── maintenance_model.joblib        # Trained XGBoost Maintenance Model
├── data/                               # Dataset & Samples
├── requirements.txt                    # Root Requirements Pointer
└── README.md                           # Documentation
```
