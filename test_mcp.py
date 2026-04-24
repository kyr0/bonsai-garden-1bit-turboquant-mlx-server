#!/usr/bin/env python3
"""Focused MCP discovery and tool execution checks for the bonsai proxy."""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from proxy import (
    MCPServerConfig,
    MCPToolBinding,
    _execute_mcp_tool_roundtrips,
    _proxy_streaming,
    call_mcp_tool,
    discover_mcp_server_tools,
    merge_openai_tools,
)


load_dotenv()

LIVE_GOOGLE_QUERY = "site:huggingface.co prism-ml Bonsai-8B-mlx-1bit"
EXPECTED_REPO_ID = "prism-ml/Bonsai-8B-mlx-1bit"

mcp = FastMCP(
    "test-mcp",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=["127.0.0.1", "127.0.0.1:*"],
        allowed_origins=["http://127.0.0.1", "http://127.0.0.1:*"],
    ),
)


@mcp.tool()
def calculate(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def make_binding(server_name: str, tool: object, config: MCPServerConfig) -> MCPToolBinding:
    return MCPToolBinding(
        alias=f"mcp__{server_name}__{tool.name}",
        server_name=server_name,
        tool_name=tool.name,
        description=tool.description,
        input_schema=getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None),
        url=config.url,
        timeout=config.timeout,
    )


def load_live_mcp_config() -> tuple[str, MCPServerConfig]:
    config_path = os.environ.get("MCP_CONFIG")
    if not config_path:
        raise AssertionError("MCP_CONFIG is not set")

    payload = json.loads(Path(config_path).expanduser().read_text())
    servers = payload.get("mcpServers") or {}
    if not isinstance(servers, dict):
        raise AssertionError("mcpServers must be a JSON object")

    for server_name, raw_config in servers.items():
        if not isinstance(raw_config, dict):
            continue
        if not raw_config.get("enabled", True):
            continue
        if raw_config.get("transport") != "streamable-http":
            continue
        url = raw_config.get("url")
        if not isinstance(url, str) or not url:
            continue
        return server_name, MCPServerConfig(
            name=server_name,
            url=url,
            transport="streamable-http",
            timeout=float(raw_config.get("timeout", 30)),
        )

    raise AssertionError("No enabled streamable-http MCP servers found in MCP_CONFIG")


def find_google_search_tool(tools: list[object]) -> object:
    for tool in tools:
        if getattr(tool, "name", None) == "google_search":
            return tool
    for tool in tools:
        if "google_search" in getattr(tool, "name", ""):
            return tool
    raise AssertionError("No google_search-like MCP tool found")


async def run_local_proxy_regression() -> None:
    app = mcp.streamable_http_app()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            config = MCPServerConfig(
                name="aidana",
                url="http://127.0.0.1/mcp",
                transport="streamable-http",
                timeout=30.0,
            )

            tools = await discover_mcp_server_tools(config, http_client=client)
            if len(tools) != 1 or tools[0].name != "calculate":
                raise AssertionError("MCP discovery did not return the expected tool")

            binding = make_binding("aidana", tools[0], config)

            merged = merge_openai_tools(
                {"messages": [{"role": "user", "content": "What is 2 + 2?"}]},
                [binding.to_openai_tool()],
            )
            merged_tools = merged.get("tools") or []
            if len(merged_tools) != 1 or merged_tools[0]["function"]["name"] != binding.alias:
                raise AssertionError("MCP tools were not injected into the OpenAI request payload")

            result = await call_mcp_tool(binding, {"a": 2, "b": 2}, http_client=client)
            if result.strip() != "4":
                raise AssertionError(f"MCP tool execution returned an unexpected result: {result!r}")

            class FakeRegistry:
                def __init__(self, alias: str):
                    self.alias = alias

                def knows_all(self, tool_calls):
                    return all((tool_call.get("function") or {}).get("name") == self.alias for tool_call in tool_calls)

                async def call_tool(self, alias: str, arguments: dict[str, int]) -> str:
                    if alias != self.alias or arguments != {"a": 2, "b": 2}:
                        raise AssertionError(f"unexpected MCP tool call: {alias} {arguments}")
                    return "4"

            backend_requests: list[dict] = []

            def backend_handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content.decode("utf-8"))
                backend_requests.append(payload)

                if len(backend_requests) == 1:
                    return httpx.Response(
                        200,
                        json={
                            "id": "chatcmpl-1",
                            "object": "chat.completion",
                            "created": 1,
                            "model": "test-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "tool_calls",
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {
                                                    "name": binding.alias,
                                                    "arguments": json.dumps({"a": 2, "b": 2}),
                                                },
                                            }
                                        ],
                                    },
                                }
                            ],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        },
                    )

                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-2",
                        "object": "chat.completion",
                        "created": 2,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "The result is 4.",
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
                    },
                )

            initial_payload = merge_openai_tools(
                {"messages": [{"role": "user", "content": "What is 2 + 2?"}], "max_tokens": 64},
                [binding.to_openai_tool()],
            )

            async with httpx.AsyncClient(transport=httpx.MockTransport(backend_handler)) as backend_client:
                initial_response = await backend_client.post("http://backend.local/v1/chat/completions", json=initial_payload)
                final_response = await _execute_mcp_tool_roundtrips(
                    client=backend_client,
                    method="POST",
                    url="http://backend.local/v1/chat/completions",
                    headers={},
                    initial_response=initial_response,
                    request_json=initial_payload,
                    mcp_registry=FakeRegistry(binding.alias),
                )

            final_payload = json.loads(final_response.body.decode("utf-8"))
            final_message = final_payload["choices"][0]["message"]["content"]
            if final_message != "The result is 4.":
                raise AssertionError(f"MCP tool round-trip returned an unexpected final answer: {final_message!r}")

            if len(backend_requests) != 2 or backend_requests[1]["messages"][-1]["content"] != "4":
                raise AssertionError("MCP tool results were not fed back into the follow-up chat completion request")

            if final_payload.get("usage", {}).get("total_tokens") != 24:
                raise AssertionError(f"MCP usage aggregation returned an unexpected payload: {final_payload.get('usage')!r}")

            class FakeStreamingManager:
                def __init__(self):
                    self.release_count = 0

                def release_backend(self, backend) -> None:
                    self.release_count += 1

            class FakeStreamingBackend:
                pass

            class MockSSEStream(httpx.AsyncByteStream):
                def __init__(self, events: list[str]):
                    self.events = events

                async def __aiter__(self):
                    for event in self.events:
                        yield event.encode("utf-8")

                async def aclose(self) -> None:
                    return None

            streaming_backend_requests: list[dict] = []

            def streaming_backend_handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content.decode("utf-8"))
                streaming_backend_requests.append(payload)

                if len(streaming_backend_requests) == 1:
                    return httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        stream=MockSSEStream(
                            [
                                ": keepalive 1/1\n\n",
                                "data: "
                                + json.dumps(
                                    {
                                        "id": "chatcmpl-stream-1",
                                        "object": "chat.completion.chunk",
                                        "created": 3,
                                        "model": "test-model",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"role": "assistant"},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                                + "\n\n",
                                "data: "
                                + json.dumps(
                                    {
                                        "id": "chatcmpl-stream-1",
                                        "object": "chat.completion.chunk",
                                        "created": 3,
                                        "model": "test-model",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {
                                                    "tool_calls": [
                                                        {
                                                            "id": "call_stream_1",
                                                            "type": "function",
                                                            "function": {
                                                                "name": binding.alias,
                                                                "arguments": json.dumps({"a": 2, "b": 2}),
                                                            },
                                                        }
                                                    ]
                                                },
                                                "finish_reason": "tool_calls",
                                            }
                                        ],
                                    }
                                )
                                + "\n\n",
                                "data: "
                                + json.dumps(
                                    {
                                        "id": "chatcmpl-stream-1",
                                        "object": "chat.completion",
                                        "created": 3,
                                        "model": "test-model",
                                        "choices": [],
                                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                                    }
                                )
                                + "\n\n",
                                "data: [DONE]\n\n",
                            ]
                        ),
                    )

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=MockSSEStream(
                        [
                            "data: "
                            + json.dumps(
                                {
                                    "id": "chatcmpl-stream-2",
                                    "object": "chat.completion.chunk",
                                    "created": 4,
                                    "model": "test-model",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": "The "},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            + "\n\n",
                            "data: "
                            + json.dumps(
                                {
                                    "id": "chatcmpl-stream-2",
                                    "object": "chat.completion.chunk",
                                    "created": 4,
                                    "model": "test-model",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": "result is 4."},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            + "\n\n",
                            "data: "
                            + json.dumps(
                                {
                                    "id": "chatcmpl-stream-2",
                                    "object": "chat.completion.chunk",
                                    "created": 4,
                                    "model": "test-model",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop",
                                        }
                                    ],
                                }
                            )
                            + "\n\n",
                            "data: "
                            + json.dumps(
                                {
                                    "id": "chatcmpl-stream-2",
                                    "object": "chat.completion",
                                    "created": 4,
                                    "model": "test-model",
                                    "choices": [],
                                    "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
                                }
                            )
                            + "\n\n",
                            "data: [DONE]\n\n",
                        ]
                    ),
                )

            streaming_payload = merge_openai_tools(
                {
                    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                    "max_tokens": 64,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                [binding.to_openai_tool()],
            )

            async with httpx.AsyncClient(transport=httpx.MockTransport(streaming_backend_handler)) as backend_client:
                streaming_manager = FakeStreamingManager()
                stream_response = await _proxy_streaming(
                    client=backend_client,
                    method="POST",
                    url="http://backend.local/v1/chat/completions",
                    headers={},
                    body=b"",
                    backend=FakeStreamingBackend(),
                    mgr=streaming_manager,
                    request_json=streaming_payload,
                    mcp_registry=FakeRegistry(binding.alias),
                )
                stream_chunks: list[str] = []
                async for chunk in stream_response.body_iterator:
                    stream_chunks.append(chunk.decode("utf-8"))

            streamed_body = "".join(stream_chunks)
            streamed_parts: list[str] = []
            for frame in stream_chunks:
                if not frame.startswith("data: "):
                    continue
                data = frame[6:].strip()
                if data == "[DONE]":
                    continue
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content_delta = delta.get("content")
                if isinstance(content_delta, str):
                    streamed_parts.append(content_delta)

            if "".join(streamed_parts) != "The result is 4." or "data: [DONE]" not in streamed_body:
                raise AssertionError(f"Synthetic MCP stream did not contain the expected final answer: {streamed_body!r}")

            if '"content": "The "' not in streamed_body or '"content": "result is 4."' not in streamed_body:
                raise AssertionError(f"Synthetic MCP stream did not preserve incremental streaming frames: {streamed_body!r}")

            if '"usage"' not in streamed_body:
                raise AssertionError(f"Synthetic MCP stream did not include usage data: {streamed_body!r}")

            if len(streaming_backend_requests) != 2:
                raise AssertionError("Streaming MCP proxy path did not perform the expected streamed follow-up round-trip")

            if streaming_backend_requests[0].get("stream") is not True or streaming_backend_requests[1].get("stream") is not True:
                raise AssertionError("Streaming MCP proxy path stopped using upstream stream mode")

            if streaming_manager.release_count != 1:
                raise AssertionError("Streaming MCP proxy path did not release the backend exactly once")

    print("[OK] Local MCP regression passed")
    print("[OK] Local MCP streaming regression passed")


async def run_live_google_search_test() -> None:
    server_name, config = load_live_mcp_config()
    tools = await discover_mcp_server_tools(config)
    if not tools:
        raise AssertionError("Live MCP discovery returned no tools")

    google_tool = find_google_search_tool(tools)
    binding = make_binding(server_name, google_tool, config)

    print(f"[live] discovered {len(tools)} tool(s) from {server_name}")
    print(f"[live] using MCP tool: {google_tool.name}")

    direct_result = await call_mcp_tool(
        binding,
        {"query": LIVE_GOOGLE_QUERY, "topK": 3, "aiSummary": False},
    )
    if EXPECTED_REPO_ID not in direct_result:
        raise AssertionError(
            "Live MCP google_search result did not contain the expected repo id: "
            f"{EXPECTED_REPO_ID!r}"
        )
    print("[OK] Live MCP google_search returned the expected repo id")

    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    if not openai_base_url:
        raise AssertionError("OPENAI_BASE_URL is not set")

    openai_api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    openai_model = os.environ.get("OPENAI_MODEL", "default_model")
    prompt = (
        "Use google_search to search for "
        f"{LIVE_GOOGLE_QUERY} and reply with only the exact model repo id from the top result."
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = await client.post(
            f"{openai_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        payload = response.json()

        streamed_parts: list[str] = []
        saw_done = False
        saw_usage = False

        async with client.stream(
            "POST",
            f"{openai_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as stream_response:
            stream_response.raise_for_status()
            async for line in stream_response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    saw_done = True
                    continue

                chunk = json.loads(data)
                if isinstance(chunk.get("usage"), dict):
                    saw_usage = True

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content_delta = delta.get("content")
                if isinstance(content_delta, str):
                    streamed_parts.append(content_delta)

    message = (((payload.get("choices") or [{}])[0]).get("message") or {})
    content = (message.get("content") or "").strip()
    if EXPECTED_REPO_ID not in content:
        raise AssertionError(
            "Live bonsai server did not return the expected repo id from google_search. "
            f"Got: {content!r}"
        )

    print(f"[OK] Live bonsai server response: {content}")

    streamed_content = "".join(streamed_parts).strip()
    if EXPECTED_REPO_ID not in streamed_content:
        raise AssertionError(
            "Live streamed bonsai response did not return the expected repo id from google_search. "
            f"Got: {streamed_content!r}"
        )

    if not saw_done:
        raise AssertionError("Live streamed bonsai response did not terminate with [DONE]")

    if not saw_usage:
        raise AssertionError("Live streamed bonsai response did not include usage data")

    print(f"[OK] Live bonsai streamed response: {streamed_content}")


async def main() -> int:
    try:
        await run_local_proxy_regression()
        await run_live_google_search_test()
    except Exception as exc:
        print(f"[X] {exc}")
        return 1

    print("[OK] MCP discovery, tool execution, and live Google search integration work")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
