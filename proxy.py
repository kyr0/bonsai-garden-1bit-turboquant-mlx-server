#!/usr/bin/env python3
"""bonsai-proxy: reverse proxy for mlx_lm with auto-unload, auto-scale, and memory watchdog."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("bonsai-proxy")

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


# -- backend instance --------------------------------------------------


@dataclass
class Backend:
    port: int
    process: subprocess.Popen | None = None
    active_connections: int = 0
    spawn_time: float = 0.0
    last_request_time: float = 0.0
    baseline_kb: int = 0
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

    # -- MLX memory polling ----------------------------------------

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

        # -- memory pressure restart --------------------------
        if total_active == 0:
            for b in alive:
                if not b.baseline_kb:
                    continue
                fp = b.footprint_kb()
                if fp is None:
                    continue
                pressure = ((fp - b.baseline_kb) * 100) // b.baseline_kb
                if pressure >= self.pressure_threshold:
                    idle_secs = now - b.last_request_time
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
    alive = sum(1 for b in mgr.backends if b.alive)
    total_active = sum(b.active_connections for b in mgr.backends)
    return JSONResponse({
        "status": "ok",
        "backends_alive": alive,
        "backends_max": mgr.max_backends,
        "active_connections": total_active,
    })


async def proxy_request(request: Request) -> Response:
    mgr: BackendManager = request.app.state.manager
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

        # Check if streaming is requested
        is_stream = False
        if body and request.method == "POST":
            try:
                import json
                parsed = json.loads(body)
                is_stream = parsed.get("stream", False)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        if is_stream:
            return await _proxy_streaming(client, request.method, url, headers, body, backend)
        else:
            return await _proxy_regular(client, request.method, url, headers, body, backend)

    except Exception as e:
        backend.active_connections = max(0, backend.active_connections - 1)
        log.exception("proxy error for backend :%d", backend.port)
        return JSONResponse({"error": str(e)}, status_code=502)


async def _proxy_regular(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    body: bytes,
    backend: Backend,
) -> Response:
    try:
        resp = await client.request(method, url, headers=headers, content=body)
        backend.active_connections = max(0, backend.active_connections - 1)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception:
        backend.active_connections = max(0, backend.active_connections - 1)
        raise


async def _proxy_streaming(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    body: bytes,
    backend: Backend,
) -> StreamingResponse:
    req = client.build_request(method, url, headers=headers, content=body)
    resp = await client.send(req, stream=True)

    async def stream_body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            backend.active_connections = max(0, backend.active_connections - 1)

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
    await mgr.start()
    yield
    await mgr.stop()
    client = get_client()
    if client and not client.is_closed:
        await client.aclose()


def create_app(manager: BackendManager) -> Starlette:
    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/{path:path}", proxy_request, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.manager = manager
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

    app = create_app(manager)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
