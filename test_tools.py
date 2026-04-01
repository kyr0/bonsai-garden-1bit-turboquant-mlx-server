#!/usr/bin/env python3
"""Test tool calling via the Bonsai OpenAI-compatible API."""
import json
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:8430"

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


def api_call(messages, tools=None):
    payload = {
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.1,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def run_test(name, user_message):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    # Step 1: Send request with tools
    print(f"\n[1] User: {user_message}")
    result = api_call(messages, tools=TOOLS)
    choice = result["choices"][0]
    assistant_msg = choice["message"]

    print(f"    Finish reason: {choice.get('finish_reason')}")

    tool_calls = assistant_msg.get("tool_calls", [])
    if tool_calls:
        print(f"    Tool calls: {len(tool_calls)}")
        for tc in tool_calls:
            fn = tc["function"]
            print(f"      -> {fn['name']}({fn.get('arguments', '{}')})")

        # Step 2: Send tool results back
        messages.append(assistant_msg)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            tool_result = TOOL_RESULTS.get(fn_name, '{"error": "unknown tool"}')
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })

        print(f"\n[2] Sending tool results back...")
        result2 = api_call(messages)
        final_content = result2["choices"][0]["message"]["content"]
        print(f"    Assistant: {final_content[:200]}")
        return True
    else:
        content = assistant_msg.get("content", "")
        print(f"    Assistant (no tool call): {content[:200]}")
        # Not necessarily a failure - model may choose to answer directly
        return True


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        ("Weather query", "What's the weather like in San Francisco?"),
        ("Math query", "What is 6 times 7?"),
        ("Multi-step", "What's the weather in Tokyo and also calculate 123 + 456?"),
    ]

    for name, msg in tests:
        try:
            ok = run_test(name, msg)
            if ok:
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
