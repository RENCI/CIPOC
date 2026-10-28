import json
from dataclasses import dataclass, fields
from typing import Optional, Any, ClassVar, Callable, Iterable
import pandas as pd
from pydantic import BaseModel

from language_model_service_api.languagemodelservice_api_completion_v3 import (
    GptChatCompletionRequest,
    GptResponseFormat,
    GptResponseFormatType,
    ReasoningEffort,
    GptToolChoice,
    GptTool,
    GptNoneToolChoice,
    GptAutoToolChoice,
    GptRequiredToolChoice,
    GptSpecificToolChoice,
    GptFunctionTool,
    GptFunctionToolChoice,
    GptChatCompletionResponse,
    GptChatUsage,
    GptChatCompletionChoice,
    GptChatCompletionTokensDetails,
    GptChatPromptTokensDetails,
    GptChatCompletionChoiceMessage,
    GptToolCall,
    GptToolCallInfo,
    FunctionToolCallInfo,
)
from language_model_service_api.languagemodelservice_api import ChatMessage, ChatMessageRole, FunctionCall

from cipoc.llm import ChatCompletionStatusError


# OpenAI to Foundry request mapping
MESSAGE_ROLES = {
    "user": ChatMessageRole.USER,
    "system": ChatMessageRole.SYSTEM,
    "assistant": ChatMessageRole.ASSISTANT,
    "function": ChatMessageRole.FUNCTION,
    "tool": ChatMessageRole.TOOL,
}

EFFORT_LEVELS = {
    "none": ReasoningEffort.NONE,
    "minimal": ReasoningEffort.MINIMAL,
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
    "xhigh": ReasoningEffort.XHIGH,
}

TOOL_CHOICE_MODES = {
    "none": GptNoneToolChoice(),
    "auto": GptAutoToolChoice(),
    "required": GptRequiredToolChoice(),
}

LEGACY_FUNCTION_FIELDS = ("functions", "function_call")


def to_foundry_tool_call(tool_call: dict[str, Any]) -> GptToolCall:
    return GptToolCall(
        id=tool_call["id"], tool_call=GptToolCallInfo(function=FunctionToolCallInfo(**tool_call["function"]))
    )


def to_foundry_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[GptToolCall] | None:
    if not tool_calls:
        return
    return [to_foundry_tool_call(tool_call) for tool_call in tool_calls]


def to_foundry_message(message: dict[str, Any]) -> ChatMessage:
    if message.get("function_call"):
        raise ChatCompletionStatusError(
            422, f"The Foundry models API does not accept legacy function calls; use tools instead."
        )
    return ChatMessage(
        **{
            **message,
            "role": MESSAGE_ROLES.get((message.get("role") or "").lower(), ChatMessageRole.UNKNOWN),
            "tool_calls": to_foundry_tool_calls(message.get("tool_calls")),
        }
    )


def to_foundry_messages(messages: list[dict[str, str]] | None) -> list[ChatMessage]:
    if not messages:
        raise ValueError("No messages present in LLM request.")
    return [to_foundry_message(message) for message in messages]


def to_foundry_effort(reasoning_effort: str) -> ReasoningEffort:
    return EFFORT_LEVELS.get(reasoning_effort.lower(), ReasoningEffort.UNKNOWN)


def to_foundry_response_format(response_format: dict[str, Any] | type[BaseModel]) -> GptResponseFormat:
    if not isinstance(response_format, dict):
        response_format = response_format.model_json_schema()
    return GptResponseFormat(
        type=GptResponseFormatType.JSON_SCHEMA,
        json_schema=response_format["json_schema"],
    )


def to_foundry_tool_choice(tool_choice: str | dict[str, Any]) -> GptToolChoice:
    # OpenAI sends a mode string, or {"type": "function", "function": {"name": ...}} to force one tool.
    if isinstance(tool_choice, dict):
        name = tool_choice["function"]["name"]
        return GptToolChoice(specific=GptSpecificToolChoice(GptFunctionToolChoice(name)))
    mode = tool_choice.lower()
    if mode not in TOOL_CHOICE_MODES:
        raise ValueError(f"Unsupported tool_choice {tool_choice!r}.")
    return GptToolChoice(**{mode: TOOL_CHOICE_MODES[mode]})


def to_foundry_tools(tools: list[dict[str, Any]]) -> list[GptTool]:
    tools = [tool.get("function", tool) for tool in tools]
    return [GptTool(GptFunctionTool(**tool)) for tool in tools]


@dataclass(frozen=True)
class FoundryChatArgs:
    messages: list[ChatMessage]
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    reasoning_effort: ReasoningEffort | None = None
    response_format: GptResponseFormat | None = None
    seed: int | None = None
    stop: list[str] | None = None
    temperature: float | None = None
    tool_choice: GptToolChoice | None = None
    tools: list[GptTool] | None = None
    top_p: float | None = None

    _CONVERTERS: ClassVar[dict[str, Callable[[Any], Any]]] = {
        "messages": to_foundry_messages,
        "reasoning_effort": to_foundry_effort,
        "response_format": to_foundry_response_format,
        "tool_choice": to_foundry_tool_choice,
        "tools": to_foundry_tools,
    }

    @classmethod
    def from_openai(cls, body: dict[str, Any]) -> "FoundryChatArgs":
        names = {f.name for f in fields(cls)}
        values = {key: value for key, value in body.items() if key in names}
        for name, convert in cls._CONVERTERS.items():
            if name in values:
                values[name] = convert(values[name])
        return cls(**values)

    def to_foundry(self) -> GptChatCompletionRequest:
        # Not dataclasses.asdict(): it deep-copies and would turn any
        # dataclass-based Foundry objects into plain dicts.
        return GptChatCompletionRequest(**{f.name: getattr(self, f.name) for f in fields(self)})


def to_foundry_request(body: dict) -> GptChatCompletionRequest:
    legacy = [key for key in LEGACY_FUNCTION_FIELDS if body.get(key) is not None]
    if legacy:
        raise ChatCompletionStatusError(
            422, f"The Foundry models API does not accept {legacy}; use tools/tool_choice instead."
        )
    return FoundryChatArgs.from_openai(body).to_foundry()


# Foundry to OpenAI response mapping
RESPONSE_FIELDS = ("id", "created", "model", "object", "system_fingerprint")
CHOICE_FIELDS = ("index", "finish_reason")
MESSAGE_FIELDS = ("content", "reasoning_content", "refusal")
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
PROMPT_DETAIL_FIELDS = ("cached_tokens", "audio_tokens")
COMPLETION_DETAIL_FIELDS = (
    "reasoning_tokens",
    "audio_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
)
FUNCTION_FIELDS = ("name", "arguments")
FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})


def foundry_model_to_dict(model, fields: Iterable[str]) -> dict[str, Any]:
    return {attr: getattr(model, attr) for attr in fields}


def from_foundry_usage(usage: GptChatUsage | None) -> dict[str, Any] | None:
    # GptChatUsage.prompt_tokens_details: GptChatPromptTokensDetails
    # GptChatUsage.completion_tokens_details: GptChatCompletionTokensDetails
    if usage is None:
        return
    return {
        **foundry_model_to_dict(usage, USAGE_FIELDS),
        "prompt_tokens_details": foundry_model_to_dict(usage.prompt_tokens_details, PROMPT_DETAIL_FIELDS)
        if usage.prompt_tokens_details
        else None,
        "completion_tokens_details": foundry_model_to_dict(usage.completion_tokens_details, COMPLETION_DETAIL_FIELDS)
        if usage.completion_tokens_details
        else None,
    }


def from_foundry_tool_call(tool_call: GptToolCall) -> dict[str, Any]:
    # GptToolCall.tool_call: GptToolCallInfo
    # GptToolCallInfo.function: FunctionToolCallInfo
    return {
        "id": tool_call.id,
        "type": "function",
        "function": foundry_model_to_dict(tool_call.tool_call.function, FUNCTION_FIELDS),
    }


def from_foundry_choice_message(message: GptChatCompletionChoiceMessage) -> dict[str, Any]:
    # message.role: GptChatMessageRole, but this always should be assistant
    return {
        "role": "assistant",
        **foundry_model_to_dict(message, MESSAGE_FIELDS),
        "tool_calls": [from_foundry_tool_call(call) for call in message.tool_calls or ()],
        "function_call": foundry_model_to_dict(message.function_call, FUNCTION_FIELDS)
        if message.function_call
        else None,
    }


def from_foundry_choice(choice: GptChatCompletionChoice) -> dict[str, Any]:
    return {
        **foundry_model_to_dict(choice, CHOICE_FIELDS),
        "message": from_foundry_choice_message(choice.message),
    }


def to_openai_response(response: GptChatCompletionResponse) -> dict[str, Any]:
    return {
        **foundry_model_to_dict(response, RESPONSE_FIELDS),
        "choices": [from_foundry_choice(choice) for choice in response.choices],
        "usage": from_foundry_usage(response.usage),
    }
