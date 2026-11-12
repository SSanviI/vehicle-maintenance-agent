"""
Pydantic schemas for agent-related data.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    VEHICLE_INFORMATION = "vehicle_information"
    ASK_PREDICTION = "ask_prediction"
    DONT_KNOW = "dont_know"
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    CHANGE_ANSWER = "change_answer"
    GENERAL_QUESTION = "general_question"
    FAREWELL = "farewell"
    OTHER = "other"


class ExtractedFeatures(BaseModel):
    """Features extracted from user message by LLM."""
    intent: IntentType = IntentType.OTHER
    extracted_features: Dict[str, Any] = Field(default_factory=dict)
    unknown_features: List[str] = Field(default_factory=list)
    responding_to_feature: Optional[str] = None
    user_sentiment: Optional[str] = None
    requires_clarification: bool = False
    clarification_target: Optional[str] = None


class AgentAction(str, Enum):
    PREDICT = "predict"
    ASK_QUESTION = "ask_question"
    PROVIDE_RESULT = "provide_result"
    UPDATE_STATE = "update_state"
    RETRIEVE_VEHICLE = "retrieve_vehicle"
    GREET = "greet"
    RESPOND = "respond"


class AgentPlan(BaseModel):
    """Agent's planned next action(s)."""
    primary_action: AgentAction
    question_to_ask: Optional[str] = None
    target_feature: Optional[str] = None
    should_predict: bool = False
    reasoning: Optional[str] = None


class ToolCall(BaseModel):
    """Structured tool call request."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from a tool call."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
