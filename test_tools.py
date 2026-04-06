#!/usr/bin/env python3
"""Test tool calling via the OpenAI-compatible API with Pydantic schema validation."""
import json
import sys
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

client = OpenAI()
MODEL = client.models.list().data[0].id


# ── Pydantic models for tool call argument validation ────────────────

class GetWeatherArgs(BaseModel):
    location: str
    unit: Optional[Literal["celsius", "fahrenheit"]] = None


class CalculateArgs(BaseModel):
    expression: str


TOOL_ARG_SCHEMAS = {
    "get_weather": GetWeatherArgs,
    "calculate": CalculateArgs,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '2 + 2'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# Simulated tool results
TOOL_RESULTS = {
    "get_weather": json.dumps({"temperature": 18, "unit": "celsius", "condition": "partly cloudy"}),
    "calculate": json.dumps({"result": 42}),
}


def run_test(name, user_message, expected_tools=None):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]
    errors = []

    # Step 1: Send request with tools
    print(f"\n[1] User: {user_message}")
    result = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS,
        max_tokens=256, temperature=0.1,
    )
    choice = result.choices[0]
    assistant_msg = choice.message

    print(f"    Finish reason: {choice.finish_reason}")

    tool_calls = assistant_msg.tool_calls or []

    # Validate: expected tool names were called
    if expected_tools is not None:
        called = sorted(tc.function.name for tc in tool_calls)
        expected_sorted = sorted(expected_tools)
        if called != expected_sorted:
            errors.append(f"Expected tools {expected_sorted}, got {called}")

    if tool_calls:
        print(f"    Tool calls: {len(tool_calls)}")
        for tc in tool_calls:
            fn_name = tc.function.name
            raw_args = tc.function.arguments
            print(f"      -> {fn_name}({raw_args})")

            # Validate: arguments are valid JSON
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError as e:
                errors.append(f"{fn_name}: invalid JSON arguments: {e}")
                continue

            # Validate: arguments match Pydantic schema
            schema = TOOL_ARG_SCHEMAS.get(fn_name)
            if schema:
                try:
                    schema.model_validate(parsed)
                    print(f"         ✓ args valid ({schema.__name__})")
                except ValidationError as e:
                    errors.append(f"{fn_name}: schema validation failed: {e}")
            else:
                errors.append(f"{fn_name}: unknown tool (no schema)")

        # Step 2: Send tool results back
        messages.append(assistant_msg)
        for tc in tool_calls:
            fn_name = tc.function.name
            tool_result = TOOL_RESULTS.get(fn_name, '{"error": "unknown tool"}')
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

        print("\n[2] Sending tool results back...")
        result2 = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=256, temperature=0.1,
        )
        final_content = result2.choices[0].message.content
        print(f"    Assistant: {final_content[:200]}")
    else:
        content = assistant_msg.content or ""
        print(f"    Assistant (no tool call): {content[:200]}")
        if expected_tools:
            errors.append(f"Expected tool calls {expected_tools}, but model answered directly")

    if errors:
        for e in errors:
            print(f"    ✗ {e}")
        return False
    return True


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        ("Weather query", "What's the weather like in San Francisco?", ["get_weather"]),
        ("Math query", "What is 6 times 7?", ["calculate"]),
        ("Multi-step", "What's the weather in Tokyo and also calculate 123 + 456?", ["get_weather", "calculate"]),
    ]

    for name, msg, expected in tests:
        try:
            result = run_test(name, msg, expected_tools=expected)
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Tool calling tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
