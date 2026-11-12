"""
RouterBackedModel — Clean Google ADK BaseLlm implementation bridging LLMRouter.

Provides native support for:
- System instructions
- Content/Parts translation
- Structured JSON generation (`response_mime_type="application/json"` / `response_format={"type": "json_object"}`)
- Structured OpenAI-compatible function calls (never stringified tool calls)
- Role-based model selection (FAST for extraction/judgment, REASONING for generation)
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Optional, List, Dict, Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse as ADKLlmResponse
from google.genai import types as genai_types

from backend.app.llm.base import LLMMessage, LLMRole
from backend.app.llm.router import get_llm_router
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class RouterBackedModel(BaseLlm):
    """
    Google ADK BaseLlm model backed by the application's Multi-LLM Router.
    """

    model: str = "router-backed"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [
            "router-backed",
            "router-backed-fast",
            "router-backed-reasoning",
            "router-backed-judge",
        ]

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[ADKLlmResponse, None]:
        """
        Translate ADK LlmRequest → LLMRouter call → ADK LlmResponse.
        """
        router = get_llm_router()

        # Determine logical role
        if "judge" in self.model:
            role = LLMRole.FAST
        elif "fast" in self.model:
            role = LLMRole.FAST
        else:
            role = LLMRole.REASONING

        # Extract system instruction
        messages: List[LLMMessage] = []
        sys_instruction = llm_request.config.system_instruction if llm_request.config else None
        if sys_instruction:
            if isinstance(sys_instruction, genai_types.Content):
                sys_text = " ".join(
                    part.text for part in (sys_instruction.parts or []) if part.text
                )
            elif isinstance(sys_instruction, str):
                sys_text = sys_instruction
            else:
                sys_text = str(sys_instruction)

            if sys_text and sys_text.strip():
                messages.append(LLMMessage(role="system", content=sys_text.strip()))

        # Convert ADK Contents to provider messages, preserving structured tool
        # responses rather than flattening them into prompt text.
        for content in (llm_request.contents or []):
            role_str = content.role or "user"
            if role_str == "model":
                role_str = "assistant"

            text_parts: List[str] = []
            for part in (content.parts or []):
                if part.text:
                    text_parts.append(part.text)

                if part.function_response:
                    response = part.function_response
                    messages.append(LLMMessage(
                        role="tool",
                        name=response.name,
                        content=json.dumps(response.response or {}),
                    ))

            combined_text = "\n".join(text_parts).strip()
            if combined_text:
                messages.append(LLMMessage(role=role_str, content=combined_text))

        if not messages:
            messages.append(LLMMessage(role="user", content=""))

        # Determine parameters
        temperature = 0.2
        if llm_request.config and llm_request.config.temperature is not None:
            temperature = float(llm_request.config.temperature)

        max_tokens = 800
        if llm_request.config and llm_request.config.max_output_tokens is not None:
            max_tokens = int(llm_request.config.max_output_tokens)

        # Check for JSON output request
        response_format = None
        if llm_request.config and getattr(llm_request.config, "response_mime_type", None) == "application/json":
            response_format = {"type": "json_object"}

        tool_definitions = []
        for tool in (llm_request.tools_dict or {}).values():
            try:
                declaration = tool._get_declaration()  # ADK's canonical schema
                schema = declaration.model_dump(exclude_none=True)
                name = schema.pop("name")
                description = schema.pop("description", "")
                parameters = schema.pop("parameters_json_schema", None) or schema.pop("parameters", {"type": "object", "properties": {}})
                tool_definitions.append({
                    "type": "function",
                    "function": {"name": name, "description": description, "parameters": parameters},
                })
            except Exception as exc:
                logger.warning("Unable to export ADK tool declaration: %s", exc)

        try:
            if stream:
                full_text: List[str] = []
                async for chunk in router.generate_stream(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    full_text.append(chunk)
                    yield ADKLlmResponse(
                        content=genai_types.Content(
                            role="model",
                            parts=[genai_types.Part(text=chunk)],
                        ),
                        partial=True,
                        turn_complete=False,
                    )

                yield ADKLlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="".join(full_text))],
                    ),
                    partial=False,
                    turn_complete=True,
                )
            else:
                response = await router.generate(
                    messages=messages,
                    role=role,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tool_definitions or None,
                )
                parts = [genai_types.Part(text=response.content)] if response.content else []
                for tool_call in response.tool_calls or []:
                    try:
                        args = json.loads(tool_call.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    parts.append(genai_types.Part(function_call=genai_types.FunctionCall(
                        name=tool_call["name"], args=args
                    )))
                yield ADKLlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=parts,
                    ),
                    partial=False,
                    turn_complete=True,
                )

        except Exception as e:
            logger.error(f"[RouterBackedModel] Error in generate_content_async: {e}", exc_info=True)
            yield ADKLlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=f"Error: {str(e)}")],
                ),
                partial=False,
                turn_complete=True,
                error_message=str(e),
            )


def get_primary_model() -> RouterBackedModel:
    """Get ADK model for primary conversational generation."""
    return RouterBackedModel(model="router-backed-reasoning")


def get_fast_model() -> RouterBackedModel:
    """Get ADK model for fast extraction and JSON parsing."""
    return RouterBackedModel(model="router-backed-fast")


def get_judge_model() -> RouterBackedModel:
    """Get ADK model for evaluation and judgment."""
    return RouterBackedModel(model="router-backed-judge")
