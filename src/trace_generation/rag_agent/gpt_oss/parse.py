import json
from dataclasses import dataclass
from typing import Any

@dataclass
class ParsedUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ParsedModelOutput:
    content: str
    reasoning: str | None
    tool_calls: list[ParsedToolCall]
    finish_reason: str | None
    usage: ParsedUsage


def _parse_usage(response) -> ParsedUsage:
    usage = getattr(response, "usage", None)

    if usage is None:
        return ParsedUsage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )

    return ParsedUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def parse_chat_completion(response) -> ParsedModelOutput:
    """
    Parse a ChatCompletion response into:
    - visible assistant text
    - optional reasoning text, if the backend exposes it
    - function/tool calls
    - finish reason
    """
    if not response.choices:
        return ParsedModelOutput(
            content="",
            reasoning=None,
            tool_calls=[],
            finish_reason=None,
            usage=ParsedUsage(None, None, None),
        )

    choice = response.choices[0]
    message = choice.message

    content = getattr(message, "content", None) or ""
    reasoning = getattr(message, "reasoning", None)

    parsed_tool_calls: list[ParsedToolCall] = []

    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        if function is None:
            continue

        name = getattr(function, "name", "")
        raw_arguments = getattr(function, "arguments", "") or "{}"

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}

        parsed_tool_calls.append(
            ParsedToolCall(
                id=tool_call.id,
                name=name,
                arguments=arguments,
            )
        )

    usage = _parse_usage(response)

    return ParsedModelOutput(
        content=content,
        reasoning=reasoning,
        tool_calls=parsed_tool_calls,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=usage,
    )
