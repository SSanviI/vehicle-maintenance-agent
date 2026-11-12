"""
ADK Multi-Agent package for Vehicle Maintenance AI.

This is the SOLE agent implementation. There is no other active agent architecture.

Exports the model bridge helpers and all agent builders.
"""

from backend.app.agent.adk_agents.model_bridge import (
    RouterBackedModel,
    get_primary_model,
    get_fast_model,
    get_judge_model,
)
from backend.app.agent.adk_agents.controller_agent import build_controller_agent
from backend.app.agent.adk_agents.information_agent import build_information_agent
from backend.app.agent.adk_agents.prediction_agent import build_prediction_agent
from backend.app.agent.adk_agents.response_agent import build_response_agent
from backend.app.agent.adk_agents.judge_agent import (
    build_judge_agent,
    run_deterministic_guards,
    generate_deterministic_fallback_response,
)

__all__ = [
    # Model bridge
    "RouterBackedModel",
    "get_primary_model",
    "get_fast_model",
    "get_judge_model",
    # Agent builders
    "build_controller_agent",
    "build_information_agent",
    "build_prediction_agent",
    "build_response_agent",
    "build_judge_agent",
    # Deterministic helpers (used by runner and evaluation)
    "run_deterministic_guards",
    "generate_deterministic_fallback_response",
]
