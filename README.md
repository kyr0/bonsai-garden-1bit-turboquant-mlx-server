# Bonsai Garden High Performance MLX Inference Server

This is an OpenAI-compatible LLM inference server running [**Bonsai-8B**](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit) (1-bit quantized, _based on Qwen3-8B_) with 8-bit KV cache quantization on any Apple Silicon machine via [MLX](https://github.com/ml-explore/mlx).

It uses a [PrismML MLX fork](https://github.com/PrismML-Eng/mlx) for 1-bit quantization support. I also fixed quite a bunch of bugs, patched in KV quantization support and implemented custom features. All of this is not yet merged upstream. Feel free to take anything you need.

> A Macbook Air M4 can run the model with good performance (see below) and handle a wide range of tasks, including tool calling and long-context retrieval:

<img src="docs/bonsai_demo_aichat.gif" alt="Bonsai demo in aichat terminal UI" width="600"/>

## Requirements

- macOS on Apple Silicon (tested on Macbook Air M4 24GB, macOS 15.7.3 with Metal SDK 26)
- Xcode Metal Toolchain (`setup` installs it automatically)
- Python 3.12

## Assumptions on Memory Usage and Performance

- Average memory footprint: 1.36 GB (idle) to ~1.58 GB (models + initial KV cache, auto-grow)
- Max. peak @ 65k tokens: up to 3.79 GB (models + KV cache + activations)
- I've implemented aggressive memory management mechanisms with MLX memory pooling and per-backend cache clearing in `_clear_backend_cache()`. This server shouldn't leak memory over time and sustain typical single-user load, returning to 1.36 GB idle footprint and 0 GB footprint after long idle periods (full model weight unloading when unused). First new request has 2-3 seconds extra delay in this case.

=> Fits comfortably within 16 GB RAM with room for OS and other processes; fits densely within 8 GB RAM with ample headroom. macOS  memory management is efficiently using memory compression, so actual memory pressure is about 2x lower than raw numbers suggest.

## Special performance optimizations

This server diverges from mlx-lm baseline by patching in / configuring:

- 1 bit quantization for weights (by PrismML)
- Two patches are applied to mlx-lm in order: `patches/1_rotation.patch` (rotation support + server extensions) and `patches/2_turbo_quant.patch` (TurboQuant KV cache compression). Default KV quantization is **8-bit standard** (`--kv-bits 8 --quantized-kv-start 128`), which keeps the first 128 tokens in full precision and quantizes the rest. For more aggressive compression, you can enable TurboQuant instead (see comment in Makefile): `--turbo-kv-bits 3 --turbo-fp16-layers 2` uses PolarQuant with randomized Hadamard rotation + Lloyd-Max codebook.
- **Memory diagnostics**: the patched server exposes `GET /debug/memory` (MLX active/peak/cache/prompt-cache stats) and `POST /debug/clear_cache` (frees MLX buffer cache, returns before/after). The proxy watchdog polls these every tick and clears the buffer cache automatically when a backend goes idle.
- A **reverse proxy** (`proxy.py`) sits in front of the mlx_lm backends and provides:
  - **Connection-aware routing**: tracks active requests per backend, routes to the least-busy one
  - **Auto-scale**: when all backends are busy and a new request arrives, a new backend is spawned on the next port. If memory allows. The `--max-mem-util` setting (default 80%) is a hard ceiling: after spawning, at least 20% of unified memory (including GPU) must remain free. This overrides `--max-backends` if the machine is memory-constrained.
  - **Auto-unload**: after `--idle-timeout` (default 300s) with zero active requests, all backends are killed and memory is freed. The proxy stays alive and accepts new connections; the next request triggers a cold-start (~2-3s), which is a great compromise for consumer workloads.
  - **Memory watchdog**: a background task samples baseline memory footprint per backend (using macOS `footprint` which includes Metal/GPU unified memory). When pressure is detected and the backend is idle >=30s, it first tries clearing the MLX buffer cache; if that's insufficient, it restarts the backend.
  - **SSE streaming relay**: raw chunk pass-through for ordinary streams, plus server-side MCP tool execution for streamed chat completions with an OpenAI-compatible SSE response.

## Quick start

```sh
make setup   # install uv, venv, deps, download model
make start   # launch server on localhost:8430
make test    # run example queries
make stop    # stop the server
```

## Terminal UI (Chat)

If you want to try the model out without writing code, you can use the built-in terminal UI (you may want to install [brew](https://brew.sh/) first):

```sh
# install aichat via brew
brew install aichat

# run aichat
aichat --session
```

You will be prompted for providing a config. 
Answer like this:

```
> No config file, create a new one? Yes
> API Provider (required): openai-compatible
> Provider Name (required): bonsai
> API Base (required): http://localhost:8430/v1
> API Key (optional): 
? LLMs to include (required):  
> [x] prism-ml/Bonsai-8B-mlx-1bit
```

## Open WebUI

You can run a chat via web-based UI using Open WebUI:

```bash
docker compose up

# Then open http://localhost:3000
```

<img src="docs/openwebui_demo.png" alt="Bonsai demo in Open WebUI" width="600"/>

**Note**: Make sure to set `max_tokens` to ~64k.

## OpenCode

If you like the model to generate code for you, [OpenCode](https://opencode.ai) is a great choice!

```brew
brew install opencode
```

```bash
OPENCODE_CONFIG_CONTENT='{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "bonsai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "bonsai",
      "options": {
        "baseURL": "http://localhost:8430/v1",
        "apiKey": "ignored"
      },
      "models": {
        "prism-ml/Bonsai-8B-mlx-1bit": {
          "name": "prism-ml/Bonsai-8B-mlx-1bit"
        }
      }
    }
  }
}' opencode -m bonsai/prism-ml/Bonsai-8B-mlx-1bit
```



## Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install everything + download model |
| `make start` | Start the OpenAI-compatible server |
| `make stop` | Stop the server |
| `make status` | Check if the server is running |
| `make log` | Tail server logs |
| `make test` | Run example queries (health, chat, streaming) |
| `make bench` | Run 50 varied prompts and report tok/s |
| `make models` | List downloaded models with size and location |
| `make clean` | Remove venv, logs, and caches |

## API

The server exposes the standard OpenAI API at `http://127.0.0.1:8430`:

```sh
curl http://localhost:8430/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128}'
```

Endpoints:
- `GET  /v1/models` - list available models
- `POST /v1/chat/completions` - chat completion (supports `"stream": true`)
- `GET  /health` - proxy health (backends alive, active connections)


## Per Request Hyperparameters

Just send them as part of the JSON body in the `POST /v1/chat/completions` request.

For example (max predictability):

```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 128,
  "temperature": 0.01,
  "seed": 42
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | `512` | Maximum number of tokens to generate |
| `temperature` | `0.5` | Sampling temperature (`0.0` = greedy/deterministic, higher = more "creative") |
| `top_p` | `0.85` | Nucleus sampling cutoff (lower = narrower vocabulary) |
| `seed` | - | Random seed for reproducible output |
| `stream` | `false` | Stream response as SSE events |

## Server Configuration

Edit variables at the top of the `Makefile`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8430` | Proxy listen port |
| `MODEL` | `prism-ml/Bonsai-8B-mlx-1bit` | HuggingFace model ID |
| `DRAFT_MODEL` | _(empty)_ | Draft model for speculative decoding (see below) |
| `MAX_BACKENDS` | `2` | Maximum backend instances |
| `MAX_MEM_UTIL` | `80` | Max memory utilisation % (must keep rest free) |

### MCP Tool Discovery

If you want the proxy to discover MCP tools automatically, point `MCP_CONFIG` at an `mcp.json` file before starting the server:

```sh
export MCP_CONFIG="$HOME/.aidana/mcp.json"
make start
```

Example config:

```json
{
  "mcpServers": {
    "aidana": {
      "transport": "streamable-http",
      "timeout": 30,
      "enabled": true,
      "url": "http://127.0.0.1:3211/mcp"
    }
  }
}
```

Behavior:

- Enabled `streamable-http` MCP servers are discovered from `MCP_CONFIG`.
- Their tools are injected into every `POST /v1/chat/completions` request.
- For both non-streaming and streaming chat-completion requests, if the model emits only discovered MCP tool calls, the proxy executes those tools server-side and continues the conversation until it gets a final assistant answer.
- Streamed MCP responses are returned as SSE events and still end with `data: [DONE]`; if the client requests `stream_options.include_usage`, the proxy also emits a final usage event.
- If a response includes tool calls that do not belong to the discovered MCP registry, the proxy leaves the response unchanged so existing OpenAI-style client-side tool loops keep working.

### Speculative Decoding

You can optionally enable speculative decoding with a draft model:

```sh
make start DRAFT_MODEL=prism-ml/Bonsai-1.7B-mlx-1bit
```

> **Note**: On most Apple Silicon machines, speculative decoding with MLX actually _decreases_ generation speed rather than improving it. The overhead of running the draft model, verifying tokens, and rolling back on misses outweighs the savings from accepted tokens - especially on memory-bandwidth-bound hardware where both models compete for the same unified memory bus. The feature is disabled by default for this reason. It may help on machines with very high memory bandwidth (e.g. M2 Ultra / M4 Max with high-bandwidth unified memory), but benchmark with `make bench` before committing to it.

## Integrations

Both examples use the standard OpenAI function calling API against the running server.

### Python

[test_tools.py](test_tools.py) - zero-dependency (stdlib `urllib` only) integration test.

```sh
make start
python test_tools.py        # or: make test (runs all suites)
```

Covers single tool calls (`get_weather`, `calculate`) and multi-step tool use (two tools in one conversation).

### Node.js / TypeScript

[test_tools.ts](test_tools.ts) - TypeScript integration test, executed via `bun` / `tsx`.

```sh
make start
pnpx/bunx/npx tsx test_tools.ts      # or: make test (runs all suites)
```

Same scenarios as the Python suite: weather query, math query, and multi-step tool calling.

> Both suites are run automatically by `make test`.

## About this model

| Config | Value | Consequence |
|----------|---------|-------------|
| RoPE (Yarn) | `4x` | RoPE scaling factor (`4x` for 16k x 4 context = 64k by default in model config). Max context window is 64k. |
| Thinking/Reasoning | - | This isn't a reasoning model. |
| Architecture | Dense | This model is based on Qwen3-8B (dense). |
| Tool Calling | Supported | The model can call tools via the OpenAI function calling API. Also multi-step tool calls are supported (parallel). |


### What the model does well

Despite being 1-bit quantized, Bonsai-8B handles a wide range of tasks (`make test` passes 18/18 tests + 6/6 tool-calling tests):

- **Concept explanations** - clear, structured answers (e.g. quantum computing with proper use of bold, headings, and analogies)
- **Factual Q&A** - short, accurate responses to direct questions ("The capital of France is Paris.")
- **Creative writing** - haiku, limericks, and freeform poetry with reasonable quality
- **System prompt adherence** - follows system-role instructions correctly
- **Code-related questions** - understands programming topics (Fibonacci, Python)
- **Tool calling** - single and parallel function calls via the OpenAI API; correctly emits `finish_reason: tool_calls` and valid JSON arguments
- **Streaming** - proper SSE streaming with `[DONE]` sentinel
- **Sampling controls** - `temperature`, `top_p`, and `seed` all work as expected; `seed` + low temp produces deterministic output across runs
- **Long context** - needle-in-a-haystack retrieval at ~4.7k prompt tokens and coherent summarization of long multi-topic transcripts

## License

MIT (for my code, for 3rd party code in `./mlx` see their respective licenses)