#!/usr/bin/env python3
"""bonsai-proxy: reverse proxy for mlx_lm with auto-unload, auto-scale, and memory watchdog."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("bonsai-proxy")

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")

def get_footprint_kb(pid: int) -> int | None:
    """Physical footprint in KB including Metal/GPU (macOS `footprint` tool)."""
    try:
        out = subprocess.check_output(
            ["footprint", str(pid)], text=True, stderr=subprocess.DEVNULL,
        )
        m = re.search(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)", out)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            return int(val * {"KB": 1, "MB": 1024, "GB": 1048576}[unit])
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return None


def get_total_memory_kb() -> int:
    """Total physical memory in KB via sysctl."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) // 1024
    except (subprocess.CalledProcessError, ValueError):
        return 16 * 1048576  # fallback 16 GB


def get_free_memory_kb() -> int:
    """Free unified memory in KB via os_proc_available_memory (macOS)."""
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("System"))
        lib.os_proc_available_memory.restype = ctypes.c_uint64
        return lib.os_proc_available_memory() // 1024
    except (OSError, AttributeError):
        # Fallback: parse vm_stat
        try:
            out = subprocess.check_output(["vm_stat"], text=True)
            pages = {}
            for line in out.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip().rstrip(".")
                    if v.isdigit():
                        pages[k.strip()] = int(v)
            page_size = 16384  # Apple Silicon
            free = pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
            return (free * page_size) // 1024
        except (subprocess.CalledProcessError, ValueError):
            return 0


def fmt_mb(kb: int) -> str:
    return f"{kb / 1024:.0f} MB"


def _sanitize_tool_name_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "tool"


def _make_mcp_tool_alias(server_name: str, tool_name: str) -> str:
    base = f"mcp__{_sanitize_tool_name_part(server_name)}__{_sanitize_tool_name_part(tool_name)}"
    if len(base) <= 64:
        return base

    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:55]}_{digest}"[:64]


def _normalize_json_schema(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict) and schema:
        return schema
    return {"type": "object", "properties": {}}


def _response_from_httpx(resp: httpx.Response) -> Response:
    headers = dict(resp.headers)
    headers.pop("content-length", None)
    return Response(content=resp.content, status_code=resp.status_code, headers=headers)


def _json_response(status_code: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> Response:
    response_headers = dict(headers or {})
    response_headers.pop("content-length", None)
    response_headers["content-type"] = "application/json"
    return Response(
        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        status_code=status_code,
        headers=response_headers,
        media_type="application/json",
    )


def _sse_frame(payload: dict[str, Any] | str) -> bytes:
    if isinstance(payload, str):
        data = payload
    else:
        data = json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode("utf-8")


def _stream_usage_requested(request_json: dict[str, Any] | None) -> bool:
    if not isinstance(request_json, dict):
        return False
    stream_options = request_json.get("stream_options")
    return isinstance(stream_options, dict) and bool(stream_options.get("include_usage"))


def _append_stream_text(message: dict[str, Any], key: str, value: Any) -> None:
    if not isinstance(value, str):
        return
    current = message.get(key)
    if not isinstance(current, str):
        current = ""
    message[key] = current + value


def _merge_stream_tool_calls(
    existing: list[dict[str, Any]],
    delta_tool_calls: Any,
) -> list[dict[str, Any]]:
    merged = json.loads(json.dumps(existing)) if existing else []
    if not isinstance(delta_tool_calls, list):
        return merged

    for list_index, tool_delta in enumerate(delta_tool_calls):
        if not isinstance(tool_delta, dict):
            continue

        target_index = tool_delta.get("index")
        if not isinstance(target_index, int):
            target_index = list_index

        while len(merged) <= target_index:
            merged.append({})

        target = merged[target_index]
        tool_id = tool_delta.get("id")
        if isinstance(tool_id, str):
            target["id"] = tool_id

        tool_type = tool_delta.get("type")
        if isinstance(tool_type, str):
            target["type"] = tool_type

        function_delta = tool_delta.get("function")
        if not isinstance(function_delta, dict):
            continue

        target_function = target.setdefault("function", {})

        function_name = function_delta.get("name")
        if isinstance(function_name, str):
            existing_name = target_function.get("name")
            if isinstance(existing_name, str) and existing_name and function_name:
                if function_name not in existing_name:
                    target_function["name"] = existing_name + function_name
            else:
                target_function["name"] = function_name

        function_arguments = function_delta.get("arguments")
        if isinstance(function_arguments, str):
            existing_arguments = target_function.get("arguments")
            if isinstance(existing_arguments, str):
                target_function["arguments"] = existing_arguments + function_arguments
            else:
                target_function["arguments"] = function_arguments

    return [tool_call for tool_call in merged if tool_call]


def _collect_stream_choice(
    message: dict[str, Any],
    chunk_payload: dict[str, Any],
) -> str | None:
    choices = chunk_payload.get("choices") or []
    if not choices:
        return None

    choice = choices[0]
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        return choice.get("finish_reason")

    role = delta.get("role")
    if isinstance(role, str):
        message["role"] = role

    _append_stream_text(message, "content", delta.get("content"))

    tool_calls = delta.get("tool_calls")
    if tool_calls:
        message["tool_calls"] = _merge_stream_tool_calls(
            message.get("tool_calls") or [],
            tool_calls,
        )

    return choice.get("finish_reason")


def _normalize_stream_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue

        function = tool_call.get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = ""

        normalized.append(
            {
                "id": tool_call.get("id"),
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )

    return normalized


def _finalize_stream_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    assistant_message = {
        "role": message.get("role", "assistant"),
        "content": message.get("content"),
    }

    if not assistant_message["content"]:
        assistant_message["content"] = None

    tool_calls = _normalize_stream_tool_calls(message.get("tool_calls") or [])
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    return assistant_message


def _make_stream_chunk(
    response_payload: dict[str, Any],
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    chunk = {
        "id": response_payload.get("id", "chatcmpl-proxy"),
        "object": "chat.completion.chunk",
        "created": response_payload.get("created", int(time.time())),
        "model": response_payload.get("model", ""),
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    system_fingerprint = response_payload.get("system_fingerprint")
    if system_fingerprint is not None:
        chunk["system_fingerprint"] = system_fingerprint
    return chunk


def _make_stream_usage_chunk(response_payload: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    chunk = {
        "id": response_payload.get("id", "chatcmpl-proxy"),
        "object": "chat.completion",
        "created": response_payload.get("created", int(time.time())),
        "model": response_payload.get("model", ""),
        "choices": [],
        "usage": usage,
    }
    system_fingerprint = response_payload.get("system_fingerprint")
    if system_fingerprint is not None:
        chunk["system_fingerprint"] = system_fingerprint
    return chunk


def _merge_usage(total: dict[str, Any] | None, usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return total
    if total is None:
        return dict(usage)

    merged = dict(total)
    for key, value in usage.items():
        if isinstance(value, int) and isinstance(merged.get(key), int):
            merged[key] += value
        elif isinstance(value, int) and key not in merged:
            merged[key] = value
        else:
            merged[key] = value
    return merged


def _extract_function_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def merge_openai_tools(payload: dict[str, Any], discovered_tools: list[dict[str, Any]]) -> dict[str, Any]:
    existing_tools = list(payload.get("tools") or [])
    existing_names = {name for name in (_extract_function_name(tool) for tool in existing_tools) if name}

    merged_tools = list(existing_tools)
    for tool in discovered_tools:
        name = _extract_function_name(tool)
        if name and name not in existing_names:
            merged_tools.append(tool)
            existing_names.add(name)

    merged_payload = dict(payload)
    merged_payload["tools"] = merged_tools
    return merged_payload


def _assistant_message_from_choice(choice: dict[str, Any]) -> dict[str, Any]:
    message = dict(choice.get("message") or {})
    tool_calls = message.get("tool_calls") or []
    assistant_message = {
        "role": message.get("role", "assistant"),
        "content": message.get("content"),
    }
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


def _serialize_mcp_tool_result(result: Any) -> str:
    structured_content = getattr(result, "structured_content", None)
    if structured_content is not None:
        return json.dumps(structured_content, ensure_ascii=False)

    chunks: list[str] = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            chunks.append(block.text)
        elif hasattr(block, "model_dump"):
            chunks.append(json.dumps(block.model_dump(mode="json", by_alias=True, exclude_none=True), ensure_ascii=False))
        else:
            chunks.append(str(block))

    if chunks:
        return "\n\n".join(chunks)

    return json.dumps({"ok": not bool(getattr(result, "is_error", False))}, ensure_ascii=False)


def _tool_error_content(tool_name: str, error: Exception) -> str:
    return json.dumps({"error": f"MCP tool '{tool_name}' failed: {error}"}, ensure_ascii=False)


def _summarize_for_log(value: Any, max_chars: int = 400) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = repr(value)

    if len(text) <= max_chars:
        return text

    return f"{text[:max_chars]}... (+{len(text) - max_chars} chars)"


def _load_mcp_client() -> tuple[Any, Any]:
    try:
        mcp_module = importlib.import_module("mcp")
        transport_module = importlib.import_module("mcp.client.streamable_http")
    except ImportError as exc:  # pragma: no cover - dependency is optional unless MCP is configured
        raise RuntimeError("MCP support requires the 'mcp' Python package") from exc

    return mcp_module.ClientSession, transport_module.streamable_http_client


def _get_mcp_tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return _normalize_json_schema(schema)


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    transport: str
    timeout: float = 30.0
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPToolBinding:
    alias: str
    server_name: str
    tool_name: str
    description: str | None
    input_schema: dict[str, Any]
    url: str
    timeout: float
    headers: dict[str, str] = field(default_factory=dict)

    def to_openai_tool(self) -> dict[str, Any]:
        description = self.description or f'Invoke MCP tool "{self.tool_name}" from server "{self.server_name}".'
        if self.description:
            description = f"[MCP server: {self.server_name}, tool: {self.tool_name}] {self.description}"
        return {
            "type": "function",
            "function": {
                "name": self.alias,
                "description": description,
                "parameters": _normalize_json_schema(self.input_schema),
            },
        }


async def discover_mcp_server_tools(
    config: MCPServerConfig,
    http_client: httpx.AsyncClient | None = None,
) -> list[Any]:
    ClientSession, streamable_http_client = _load_mcp_client()

    client = http_client
    created_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=min(config.timeout, 10.0)),
            headers=config.headers,
            follow_redirects=True,
        )
        created_client = True

    try:
        async with streamable_http_client(config.url, http_client=client) as streams:
            read_stream, write_stream = streams[:2]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return list(getattr(result, "tools", []) or [])
    finally:
        if created_client:
            await client.aclose()


async def call_mcp_tool(
    binding: MCPToolBinding,
    arguments: dict[str, Any],
    http_client: httpx.AsyncClient | None = None,
) -> str:
    ClientSession, streamable_http_client = _load_mcp_client()

    client = http_client
    created_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(binding.timeout, connect=min(binding.timeout, 10.0)),
            headers=binding.headers,
            follow_redirects=True,
        )
        created_client = True

    try:
        async with streamable_http_client(binding.url, http_client=client) as streams:
            read_stream, write_stream = streams[:2]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(binding.tool_name, arguments or {})
                return _serialize_mcp_tool_result(result)
    finally:
        if created_client:
            await client.aclose()


class MCPToolRegistry:
    def __init__(self, config_path: str | None, refresh_interval: float = 30.0):
        self.config_path = str(Path(config_path).expanduser()) if config_path else None
        self.refresh_interval = refresh_interval
        self._bindings: dict[str, MCPToolBinding] = {}
        self._openai_tools: list[dict[str, Any]] = []
        self._config_mtime: float | None = None
        self._last_refresh: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.config_path)

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.enabled,
            "config_path": self.config_path,
            "discovered_tools": len(self._bindings),
        }

    def has_tool(self, alias: str) -> bool:
        return alias in self._bindings

    def get_binding(self, alias: str) -> MCPToolBinding | None:
        return self._bindings.get(alias)

    def knows_all(self, tool_calls: list[dict[str, Any]]) -> bool:
        if not tool_calls:
            return False
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            name = function.get("name")
            if not isinstance(name, str) or name not in self._bindings:
                return False
        return True

    async def get_openai_tools(self) -> tuple[list[dict[str, Any]], dict[str, MCPToolBinding]]:
        await self.refresh()
        return list(self._openai_tools), dict(self._bindings)

    async def refresh(self, force: bool = False) -> None:
        if not self.enabled:
            return

        async with self._lock:
            if not self.config_path:
                return

            path = Path(self.config_path)
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                if self._bindings:
                    log.warning("MCP config disappeared: %s", self.config_path)
                self._bindings = {}
                self._openai_tools = []
                self._config_mtime = None
                self._last_refresh = time.monotonic()
                return

            if not force:
                stale = (time.monotonic() - self._last_refresh) >= self.refresh_interval
                config_changed = self._config_mtime != mtime
                if not stale and not config_changed:
                    return

            try:
                configs = self._load_configs(path)
            except Exception as exc:
                log.warning("failed to load MCP config %s: %s", self.config_path, exc)
                self._bindings = {}
                self._openai_tools = []
                self._config_mtime = mtime
                self._last_refresh = time.monotonic()
                return

            discoveries = await asyncio.gather(
                *(discover_mcp_server_tools(config) for config in configs),
                return_exceptions=True,
            )

            bindings: dict[str, MCPToolBinding] = {}
            openai_tools: list[dict[str, Any]] = []

            for config, discovery in zip(configs, discoveries):
                if isinstance(discovery, Exception):
                    log.warning("failed to discover MCP tools from %s (%s): %s", config.name, config.url, discovery)
                    continue

                for tool in discovery:
                    alias = _make_mcp_tool_alias(config.name, tool.name)
                    binding = MCPToolBinding(
                        alias=alias,
                        server_name=config.name,
                        tool_name=tool.name,
                        description=tool.description,
                        input_schema=_get_mcp_tool_input_schema(tool),
                        url=config.url,
                        timeout=config.timeout,
                        headers=dict(config.headers),
                    )
                    bindings[alias] = binding
                    openai_tools.append(binding.to_openai_tool())

            self._bindings = bindings
            self._openai_tools = openai_tools
            self._config_mtime = mtime
            self._last_refresh = time.monotonic()

            if bindings:
                log.info("discovered %d MCP tool(s) from %d configured server(s)", len(bindings), len(configs))


async def _execute_logged_mcp_tool_call(
    mcp_registry: MCPToolRegistry,
    tool_call: dict[str, Any],
) -> str:
    function = tool_call.get("function") or {}
    tool_alias = function.get("name", "unknown_mcp_tool")
    tool_call_id = tool_call.get("id")
    get_binding = getattr(mcp_registry, "get_binding", None)
    binding = get_binding(tool_alias) if callable(get_binding) else None
    server_name = binding.server_name if binding else "unknown"
    actual_tool_name = binding.tool_name if binding else tool_alias
    raw_arguments = function.get("arguments") or "{}"

    try:
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
    except Exception as exc:
        log.warning(
            "MCP tool rejected before execution call_id=%s alias=%s server=%s tool=%s error=%s raw_arguments=%s",
            tool_call_id,
            tool_alias,
            server_name,
            actual_tool_name,
            exc,
            _summarize_for_log(raw_arguments),
        )
        return _tool_error_content(tool_alias, exc)

    log.info(
        "calling MCP tool call_id=%s alias=%s server=%s tool=%s arguments=%s",
        tool_call_id,
        tool_alias,
        server_name,
        actual_tool_name,
        _summarize_for_log(arguments),
    )

    started_at = time.monotonic()
    try:
        content = await mcp_registry.call_tool(tool_alias, arguments)
    except Exception as exc:
        elapsed_ms = (time.monotonic() - started_at) * 1000.0
        log.warning(
            "MCP tool failed call_id=%s alias=%s server=%s tool=%s elapsed_ms=%.1f error=%s arguments=%s",
            tool_call_id,
            tool_alias,
            server_name,
            actual_tool_name,
            elapsed_ms,
            exc,
            _summarize_for_log(arguments),
        )
        return _tool_error_content(tool_alias, exc)

    elapsed_ms = (time.monotonic() - started_at) * 1000.0
    log.info(
        "completed MCP tool call_id=%s alias=%s server=%s tool=%s elapsed_ms=%.1f result=%s",
        tool_call_id,
        tool_alias,
        server_name,
        actual_tool_name,
        elapsed_ms,
        _summarize_for_log(content),
    )
    return content

    async def call_tool(self, alias: str, arguments: dict[str, Any]) -> str:
        binding = self._bindings[alias]
        return await call_mcp_tool(binding, arguments)

    def _load_configs(self, path: Path) -> list[MCPServerConfig]:
        with path.open() as fh:
            payload = json.load(fh)

        servers = payload.get("mcpServers") or {}
        if not isinstance(servers, dict):
            raise ValueError("mcpServers must be a JSON object")

        configs: list[MCPServerConfig] = []
        for name, raw_config in servers.items():
            if not isinstance(raw_config, dict):
                continue

            enabled = bool(raw_config.get("enabled", True))
            transport = str(raw_config.get("transport", ""))
            url = raw_config.get("url")
            if not enabled:
                continue
            if transport != "streamable-http":
                log.warning("skipping MCP server %s: unsupported transport %s", name, transport)
                continue
            if not isinstance(url, str) or not url:
                log.warning("skipping MCP server %s: missing url", name)
                continue

            headers = raw_config.get("headers") or {}
            if not isinstance(headers, dict):
                headers = {}

            timeout = raw_config.get("timeout", 30)
            try:
                timeout_value = float(timeout)
            except (TypeError, ValueError):
                timeout_value = 30.0

            configs.append(
                MCPServerConfig(
                    name=str(name),
                    url=url,
                    transport=transport,
                    timeout=timeout_value,
                    enabled=enabled,
                    headers={str(key): str(value) for key, value in headers.items()},
                )
            )

        return configs


# -- backend instance --------------------------------------------------


@dataclass
class Backend:
    port: int
    process: subprocess.Popen | None = None
    active_connections: int = 0
    spawn_time: float = 0.0
    last_request_time: float = 0.0
    baseline_kb: int = 0
    last_cache_clear: float = 0.0
    _baseline_samples: list[int] = field(default_factory=list)

    @property
    def pid(self) -> int | None:
        if self.process and self.process.poll() is None:
            return self.process.pid
        return None

    @property
    def alive(self) -> bool:
        return self.pid is not None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def footprint_kb(self) -> int | None:
        pid = self.pid
        return get_footprint_kb(pid) if pid else None


# -- backend manager ---------------------------------------------------


class BackendManager:
    def __init__(
        self,
        *,
        model: str,
        draft_model: str | None,
        backend_args: list[str],
        base_port: int,
        max_backends: int,
        max_mem_util: int,
        idle_timeout: int,
        pressure_threshold: int,
        watchdog_interval: int,
        baseline_start: int,
        baseline_end: int,
    ):
        self.model = model
        self.draft_model = draft_model
        self.backend_args = backend_args
        self.base_port = base_port
        self.max_backends = max_backends
        self.max_mem_util = max_mem_util
        self.idle_timeout = idle_timeout
        self.pressure_threshold = pressure_threshold
        self.watchdog_interval = watchdog_interval
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.pressure_cooldown = 30
        self.cache_clear_cooldown = 60

        self.total_memory_kb = get_total_memory_kb()
        self.backends: list[Backend] = []
        self._cold_start_lock = asyncio.Lock()
        self._cold_start_event: asyncio.Event | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._stopping = False

        log.info(
            "system memory: %s, must keep %d%% free = %s reserve",
            fmt_mb(self.total_memory_kb),
            100 - self.max_mem_util,
            fmt_mb(self.total_memory_kb * (100 - self.max_mem_util) // 100),
        )

    # -- lifecycle -------------------------------------------------

    async def start(self) -> None:
        await self._spawn_backend()
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        for b in self.backends:
            self._kill_backend(b)
        self.backends.clear()

    # -- spawn / kill ----------------------------------------------

    async def _spawn_backend(self) -> Backend:
        port = self.base_port + len(self.backends)
        backend = Backend(port=port)

        cmd = [
            VENV_PYTHON, "-m", "mlx_lm.server",
            "--model", self.model,
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        if self.draft_model:
            cmd += ["--draft-model", self.draft_model]
        cmd += self.backend_args

        log_path = os.path.join(SCRIPT_DIR, f"bonsai-backend-{port}.log")
        log_file = open(log_path, "a")

        backend.process = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file,
        )
        backend.spawn_time = time.monotonic()
        backend.last_request_time = time.monotonic()
        self.backends.append(backend)

        log.info("spawned backend pid=%d on :%d (log: %s)", backend.process.pid, port, log_path)

        # Wait for readiness
        ready = await self._wait_for_ready(backend, timeout=120)
        if not ready:
            log.warning("backend :%d failed to become ready - killing", port)
            self._kill_backend(backend)
            self.backends.remove(backend)
            raise RuntimeError(f"Backend on :{port} did not start")

        log.info("backend :%d ready (pid=%d)", port, backend.process.pid)
        return backend

    async def _wait_for_ready(self, backend: Backend, timeout: float = 120) -> bool:
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if not backend.alive:
                    return False
                try:
                    r = await client.get(f"{backend.url}/v1/models", timeout=2)
                    if r.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(2)
        return False

    def _kill_backend(self, backend: Backend) -> None:
        if backend.process and backend.process.poll() is None:
            log.info("killing backend pid=%d :%d", backend.process.pid, backend.port)
            backend.process.terminate()
            try:
                backend.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend.process.kill()
                backend.process.wait(timeout=5)

    # -- routing ---------------------------------------------------

    async def get_backend(self) -> Backend:
        """Return a backend to handle a request. May cold-start or scale up."""
        alive = [b for b in self.backends if b.alive]

        # No backends alive => cold-start
        if not alive:
            return await self._cold_start()

        # All backends busy => try to scale up
        if all(b.active_connections > 0 for b in alive) and len(alive) < self.max_backends:
            scaled = await self._try_scale_up()
            if scaled:
                alive.append(scaled)

        # Least-connections routing
        return min(alive, key=lambda b: b.active_connections)

    async def _cold_start(self) -> Backend:
        """Spawn the first backend, coalescing concurrent requests."""
        async with self._cold_start_lock:
            # Check again under lock - another coroutine may have spawned
            alive = [b for b in self.backends if b.alive]
            if alive:
                return min(alive, key=lambda b: b.active_connections)

            log.info("cold-starting backend (all models unloaded)")
            # Clean up dead backends
            self.backends.clear()
            self._cold_start_event = asyncio.Event()
            try:
                backend = await self._spawn_backend()
                self._cold_start_event.set()
                return backend
            except Exception:
                self._cold_start_event.set()
                raise

    async def _try_scale_up(self) -> Backend | None:
        """Attempt to spawn an additional backend if memory allows."""
        alive = [b for b in self.backends if b.alive]
        if len(alive) >= self.max_backends:
            return None

        if not self._memory_allows_scale_up(alive):
            return None

        # Reuse port of a dead backend slot, or take next
        port = self.base_port + len(self.backends)
        log.info(
            "scaling up: %d/%d backends busy, spawning on :%d",
            len(alive), self.max_backends, port,
        )
        try:
            return await self._spawn_backend()
        except RuntimeError:
            log.warning("scale-up failed")
            return None

    def _memory_allows_scale_up(self, alive: list[Backend]) -> bool:
        """Check whether spawning one more backend would leave enough free memory.

        With max_mem_util=80: after spawning, at least 20% of total memory must
        remain free. We measure actual free unified memory (includes GPU) and
        subtract the estimated cost of a new backend.
        """
        free_kb = get_free_memory_kb()
        reserve_kb = self.total_memory_kb * (100 - self.max_mem_util) // 100

        # Estimate new backend cost from first backend's baseline
        estimate = alive[0].baseline_kb if alive and alive[0].baseline_kb else 0
        if estimate == 0:
            estimate = 2 * 1048576  # conservative 2 GB fallback

        free_after = free_kb - estimate
        allowed = free_after >= reserve_kb

        log.info(
            "memory gate: free=%s - estimate=%s = %s remaining, need %s reserve => %s",
            fmt_mb(free_kb), fmt_mb(estimate), fmt_mb(max(0, free_after)),
            fmt_mb(reserve_kb), "PASS" if allowed else "DENY",
        )
        return allowed

    # -- unload ----------------------------------------------------

    def _unload_all(self) -> None:
        """Kill all backends - proxy stays alive."""
        log.info("unloading all backends (idle timeout)")
        for b in self.backends:
            self._kill_backend(b)
        self.backends.clear()

    def release_backend(self, backend: Backend) -> None:
        """Decrement active connections; clear MLX cache when backend goes idle."""
        backend.active_connections = max(0, backend.active_connections - 1)
        if backend.active_connections == 0 and backend.alive:
            asyncio.create_task(self._clear_backend_cache(backend))

    # -- MLX memory polling ----------------------------------------

    async def _clear_backend_cache(self, backend: Backend) -> bool:
        """Ask the backend to clear its MLX buffer cache. Returns True on success."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(f"{backend.url}/debug/clear_cache", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    log.info(
                        "cache clear :%d  before=%s  after=%s  freed=%s",
                        backend.port,
                        f"{data.get('before_cache_mb', '?'):.0f} MB",
                        f"{data.get('after_cache_mb', '?'):.0f} MB",
                        f"{data.get('freed_mb', '?'):.0f} MB",
                    )
                    return True
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            pass
        return False

    async def _poll_mlx_memory(self, backend: Backend) -> dict | None:
        """Poll the backend's /debug/memory endpoint for MLX memory stats."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{backend.url}/debug/memory", timeout=2)
                if r.status_code == 200:
                    return r.json()
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            pass
        return None

    # -- watchdog loop ---------------------------------------------

    async def _watchdog_loop(self) -> None:
        """Background task: baseline sampling, memory pressure restarts, idle unload, scale-down."""
        try:
            while not self._stopping:
                await asyncio.sleep(self.watchdog_interval)
                await self._watchdog_tick()
        except asyncio.CancelledError:
            return

    async def _watchdog_tick(self) -> None:
        alive = [b for b in self.backends if b.alive]
        if not alive:
            return

        now = time.monotonic()
        total_active = sum(b.active_connections for b in alive)

        for b in alive:
            elapsed = now - b.spawn_time
            fp = b.footprint_kb()
            if fp is None:
                continue

            # -- baseline sampling -----------------------------
            if not b.baseline_kb:
                if self.baseline_start <= elapsed < self.baseline_end:
                    b._baseline_samples.append(fp)
                elif elapsed >= self.baseline_end and b._baseline_samples:
                    b.baseline_kb = sum(b._baseline_samples) // len(b._baseline_samples)
                    log.info(
                        "baseline :%d = %s (%d samples)",
                        b.port, fmt_mb(b.baseline_kb), len(b._baseline_samples),
                    )

            # -- MLX memory diagnostics ------------------------
            mlx_mem = await self._poll_mlx_memory(b)
            if mlx_mem:
                log.info(
                    "mem :%d  footprint=%s  active=%s  peak=%s  cache=%s  pcache=%s(%d)",
                    b.port,
                    fmt_mb(fp),
                    f"{mlx_mem['active_mb']:.0f} MB",
                    f"{mlx_mem['peak_mb']:.0f} MB",
                    f"{mlx_mem['cache_mb']:.0f} MB",
                    f"{mlx_mem['prompt_cache_mb']:.0f} MB",
                    mlx_mem["prompt_cache_entries"],
                )

        # -- idle unload (scale to 0) -------------------------
        if total_active == 0 and alive:
            newest_request = max(b.last_request_time for b in alive)
            idle_secs = now - newest_request
            if idle_secs >= self.idle_timeout:
                self._unload_all()
                return

        # -- scale-down (pool > 1, idle backend) --------------
        if len(alive) > 1 and total_active < len(alive):
            for b in sorted(alive, key=lambda b: b.active_connections):
                if b.active_connections == 0 and (now - b.last_request_time) >= self.idle_timeout:
                    log.info("scaling down: killing idle backend :%d", b.port)
                    self._kill_backend(b)
                    self.backends.remove(b)
                    break  # one at a time

        # -- memory pressure: clear cache, then restart --------
        if total_active == 0:
            for b in alive:
                if not b.baseline_kb:
                    continue
                idle_secs = now - b.last_request_time
                if idle_secs < self.pressure_cooldown:
                    continue
                fp = b.footprint_kb()
                if fp is None:
                    continue
                pressure = ((fp - b.baseline_kb) * 100) // b.baseline_kb
                if pressure < self.pressure_threshold:
                    continue

                # Stage 1: try clearing the MLX buffer cache first
                if (now - b.last_cache_clear) >= self.cache_clear_cooldown:
                    cleared = await self._clear_backend_cache(b)
                    b.last_cache_clear = now
                    if cleared:
                        # Re-check after clear
                        fp2 = b.footprint_kb()
                        if fp2 is not None:
                            pressure2 = ((fp2 - b.baseline_kb) * 100) // b.baseline_kb
                            if pressure2 < self.pressure_threshold:
                                log.info(
                                    "cache clear resolved pressure :%d  %d%% -> %d%%",
                                    b.port, pressure, pressure2,
                                )
                                break
                            log.info(
                                "cache clear insufficient :%d  %d%% -> %d%%, restarting",
                                b.port, pressure, pressure2,
                            )

                # Stage 2: kill and restart
                log.info(
                    "!! backend :%d pressure %d%% >= %d%%, idle %.0fs - restarting",
                    b.port, pressure, self.pressure_threshold, idle_secs,
                )
                self._kill_backend(b)
                self.backends.remove(b)
                await self._spawn_backend()
                break  # one at a time


# -- HTTP client -------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _http_client


# -- proxy routes ------------------------------------------------------


async def health(request: Request) -> JSONResponse:
    mgr: BackendManager = request.app.state.manager
    mcp_registry: MCPToolRegistry = request.app.state.mcp_registry
    alive = sum(1 for b in mgr.backends if b.alive)
    total_active = sum(b.active_connections for b in mgr.backends)
    return JSONResponse({
        "status": "ok",
        "backends_alive": alive,
        "backends_max": mgr.max_backends,
        "active_connections": total_active,
        "mcp": mcp_registry.status(),
    })


async def proxy_request(request: Request) -> Response:
    mgr: BackendManager = request.app.state.manager
    mcp_registry: MCPToolRegistry = request.app.state.mcp_registry
    client = get_client()

    try:
        backend = await mgr.get_backend()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    backend.active_connections += 1
    backend.last_request_time = time.monotonic()

    try:
        url = f"{backend.url}{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"

        body = await request.body()
        request_json: dict[str, Any] | None = None

        # Check if streaming is requested
        is_stream = False
        if body and request.method == "POST":
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    request_json = parsed
                    is_stream = bool(parsed.get("stream", False))
                    if request.url.path in {"/v1/chat/completions", "/chat/completions"}:
                        discovered_tools, _ = await mcp_registry.get_openai_tools()
                        if discovered_tools:
                            request_json = merge_openai_tools(parsed, discovered_tools)
                            body = json.dumps(request_json, ensure_ascii=False).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        if is_stream:
            return await _proxy_streaming(
                client,
                request.method,
                url,
                headers,
                body,
                backend,
                mgr,
                request_json=request_json,
                mcp_registry=mcp_registry if request_json and request.url.path in {"/v1/chat/completions", "/chat/completions"} else None,
            )
        else:
            return await _proxy_regular(
                client,
                request.method,
                url,
                headers,
                body,
                backend,
                mgr,
                request_json=request_json,
                mcp_registry=mcp_registry,
            )

    except Exception as e:
        mgr.release_backend(backend)
        log.exception("proxy error for backend :%d", backend.port)
        return JSONResponse({"error": str(e)}, status_code=502)


async def _proxy_regular(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    body: bytes,
    backend: Backend,
    mgr: BackendManager,
    request_json: dict[str, Any] | None = None,
    mcp_registry: MCPToolRegistry | None = None,
) -> Response:
    try:
        resp = await client.request(method, url, headers=headers, content=body)

        if request_json and mcp_registry and not request_json.get("stream"):
            response = await _execute_mcp_tool_roundtrips(
                client=client,
                method=method,
                url=url,
                headers=headers,
                initial_response=resp,
                request_json=request_json,
                mcp_registry=mcp_registry,
            )
            mgr.release_backend(backend)
            return response

        mgr.release_backend(backend)
        return _response_from_httpx(resp)
    except Exception:
        mgr.release_backend(backend)
        raise


async def _execute_mcp_tool_roundtrips(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    initial_response: httpx.Response,
    request_json: dict[str, Any],
    mcp_registry: MCPToolRegistry,
) -> Response:
    if initial_response.status_code != 200:
        return _response_from_httpx(initial_response)

    try:
        response_payload = initial_response.json()
    except ValueError:
        return _response_from_httpx(initial_response)

    payload = json.loads(json.dumps(request_json))
    response_headers = dict(initial_response.headers)
    total_usage = _merge_usage(None, response_payload.get("usage"))
    max_steps = int(os.environ.get("MCP_MAX_TOOL_STEPS", "8"))

    for _ in range(max_steps):
        choices = response_payload.get("choices") or []
        if not choices:
            return _json_response(initial_response.status_code, response_payload, response_headers)

        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if choice.get("finish_reason") != "tool_calls" or not tool_calls:
            if total_usage is not None:
                response_payload["usage"] = total_usage
            return _json_response(initial_response.status_code, response_payload, response_headers)

        if not mcp_registry.knows_all(tool_calls):
            return _json_response(initial_response.status_code, response_payload, response_headers)

        payload.setdefault("messages", [])
        payload["messages"].append(_assistant_message_from_choice(choice))

        for tool_call in tool_calls:
            content = await _execute_logged_mcp_tool_call(mcp_registry, tool_call)
            payload["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": content,
            })

        followup_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        followup_response = await client.request(method, url, headers=headers, content=followup_body)
        response_headers = dict(followup_response.headers)
        if followup_response.status_code != 200:
            return _response_from_httpx(followup_response)

        try:
            response_payload = followup_response.json()
        except ValueError:
            return _response_from_httpx(followup_response)

        total_usage = _merge_usage(total_usage, response_payload.get("usage"))

    return _json_response(502, {"error": f"MCP tool loop exceeded {max_steps} round-trips"})


async def _stream_mcp_roundtrips(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    initial_response: httpx.Response,
    request_json: dict[str, Any],
    mcp_registry: MCPToolRegistry,
    backend: Backend,
    mgr: BackendManager,
):
    payload = json.loads(json.dumps(request_json))
    current_response = initial_response
    total_usage: dict[str, Any] | None = None
    include_usage = _stream_usage_requested(request_json)
    max_steps = int(os.environ.get("MCP_MAX_TOOL_STEPS", "8"))
    response_meta: dict[str, Any] | None = None

    try:
        for _ in range(max_steps):
            assistant_message: dict[str, Any] = {"role": "assistant"}
            finish_reason: str | None = None
            saw_done = False
            skip_next_blank = False

            try:
                async for line in current_response.aiter_lines():
                    if skip_next_blank and line == "":
                        skip_next_blank = False
                        continue

                    if not line:
                        continue

                    if line.startswith(":"):
                        yield f"{line}\n\n".encode("utf-8")
                        skip_next_blank = True
                        continue

                    if not line.startswith("data: "):
                        yield f"{line}\n".encode("utf-8")
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        saw_done = True
                        break

                    try:
                        chunk_payload = json.loads(data)
                    except json.JSONDecodeError:
                        yield _sse_frame(data)
                        skip_next_blank = True
                        continue

                    response_meta = chunk_payload

                    usage = chunk_payload.get("usage")
                    if isinstance(usage, dict):
                        total_usage = _merge_usage(total_usage, usage)
                        continue

                    chunk_finish_reason = _collect_stream_choice(assistant_message, chunk_payload)
                    if chunk_finish_reason is not None:
                        finish_reason = chunk_finish_reason

                    yield _sse_frame(chunk_payload)
                    skip_next_blank = True
            finally:
                await current_response.aclose()

            tool_calls = _normalize_stream_tool_calls(assistant_message.get("tool_calls") or [])
            if finish_reason == "tool_calls" and tool_calls and mcp_registry.knows_all(tool_calls):
                payload.setdefault("messages", [])
                payload["messages"].append(_finalize_stream_assistant_message(assistant_message))

                for tool_call in tool_calls:
                    content = await _execute_logged_mcp_tool_call(mcp_registry, tool_call)
                    payload["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "content": content,
                        }
                    )

                followup_request = client.build_request(
                    method,
                    url,
                    headers=headers,
                    content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )
                current_response = await client.send(followup_request, stream=True)
                if current_response.status_code != 200:
                    body = await current_response.aread()
                    await current_response.aclose()
                    error_text = body.decode("utf-8", errors="replace") or f"upstream returned {current_response.status_code}"
                    yield _sse_frame({"error": error_text})
                    if include_usage and total_usage and response_meta is not None:
                        yield _sse_frame(_make_stream_usage_chunk(response_meta, total_usage))
                    yield _sse_frame("[DONE]")
                    return
                continue

            if include_usage and total_usage and response_meta is not None:
                yield _sse_frame(_make_stream_usage_chunk(response_meta, total_usage))

            if saw_done:
                yield _sse_frame("[DONE]")
            else:
                yield _sse_frame("[DONE]")
            return

        yield _sse_frame({"error": f"MCP tool loop exceeded {max_steps} round-trips"})
        if include_usage and total_usage and response_meta is not None:
            yield _sse_frame(_make_stream_usage_chunk(response_meta, total_usage))
        yield _sse_frame("[DONE]")
    finally:
        mgr.release_backend(backend)


async def _proxy_streaming(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    body: bytes,
    backend: Backend,
    mgr: BackendManager,
    request_json: dict[str, Any] | None = None,
    mcp_registry: MCPToolRegistry | None = None,
) -> Response:
    if request_json and mcp_registry:
        req = client.build_request(
            method,
            url,
            headers=headers,
            content=json.dumps(request_json, ensure_ascii=False).encode("utf-8"),
        )
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            try:
                content = await resp.aread()
            finally:
                await resp.aclose()
                mgr.release_backend(backend)

            resp_headers = dict(resp.headers)
            resp_headers.pop("content-length", None)
            resp_headers.pop("transfer-encoding", None)
            return Response(
                content=content,
                status_code=resp.status_code,
                headers=resp_headers,
            )

        return StreamingResponse(
            _stream_mcp_roundtrips(
                client=client,
                method=method,
                url=url,
                headers=headers,
                initial_response=resp,
                request_json=request_json,
                mcp_registry=mcp_registry,
                backend=backend,
                mgr=mgr,
            ),
            status_code=200,
            headers={"cache-control": "no-cache"},
            media_type="text/event-stream",
        )

    req = client.build_request(method, url, headers=headers, content=body)
    resp = await client.send(req, stream=True)

    async def stream_body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            mgr.release_backend(backend)

    resp_headers = dict(resp.headers)
    resp_headers.pop("content-length", None)
    resp_headers.pop("transfer-encoding", None)

    return StreamingResponse(
        stream_body(),
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp_headers.get("content-type", "text/event-stream"),
    )


# -- app lifecycle -----------------------------------------------------


@asynccontextmanager
async def lifespan(app: Starlette):
    mgr: BackendManager = app.state.manager
    mcp_registry: MCPToolRegistry = app.state.mcp_registry
    await mcp_registry.refresh(force=True)
    await mgr.start()
    yield
    await mgr.stop()
    client = get_client()
    if client and not client.is_closed:
        await client.aclose()


def create_app(manager: BackendManager, mcp_registry: MCPToolRegistry) -> Starlette:
    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/{path:path}", proxy_request, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.manager = manager
    app.state.mcp_registry = mcp_registry
    return app


# -- CLI ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="bonsai-proxy: reverse proxy for mlx_lm")

    parser.add_argument("--host", default="127.0.0.1", help="Proxy listen host")
    parser.add_argument("--port", type=int, default=8430, help="Proxy listen port")
    parser.add_argument("--base-port", type=int, default=8431, help="First backend port")

    parser.add_argument("--model", required=True, help="Model name/path for backends")
    parser.add_argument("--draft-model", default=None, help="Draft model name/path")

    parser.add_argument("--max-backends", type=int, default=2, help="Max backend instances")
    parser.add_argument("--max-mem-util", type=int, default=80,
                        help="Max system memory utilisation %% (0-100), overrides --max-backends")

    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Seconds idle before unloading backends")
    parser.add_argument("--pressure-threshold", type=int, default=20,
                        help="Memory pressure %% over baseline to trigger restart")
    parser.add_argument("--watchdog-interval", type=int, default=10,
                        help="Watchdog poll interval in seconds")
    parser.add_argument("--baseline-start", type=int, default=10,
                        help="Seconds after spawn to start baseline sampling")
    parser.add_argument("--baseline-end", type=int, default=20,
                        help="Seconds after spawn to end baseline sampling")

    # Remaining args are passed through to mlx_lm.server
    args, backend_extra = parser.parse_known_args()

    logging.basicConfig(
        format="[proxy %(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    log.info("bonsai-proxy starting on %s:%d", args.host, args.port)
    log.info(
        "config: max_backends=%d, max_mem_util=%d%%, idle=%ds, pressure=%d%%",
        args.max_backends, args.max_mem_util, args.idle_timeout, args.pressure_threshold,
    )
    if backend_extra:
        log.info("backend extra args: %s", " ".join(backend_extra))

    manager = BackendManager(
        model=args.model,
        draft_model=args.draft_model,
        backend_args=backend_extra,
        base_port=args.base_port,
        max_backends=args.max_backends,
        max_mem_util=args.max_mem_util,
        idle_timeout=args.idle_timeout,
        pressure_threshold=args.pressure_threshold,
        watchdog_interval=args.watchdog_interval,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )

    mcp_registry = MCPToolRegistry(os.environ.get("MCP_CONFIG"))
    if mcp_registry.enabled:
        log.info("MCP discovery enabled via MCP_CONFIG=%s", mcp_registry.config_path)

    app = create_app(manager, mcp_registry)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
