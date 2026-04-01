# Bonsai

OpenAI-compatible LLM inference server running [**Bonsai-8B**](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit) (1-bit quantized) on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Uses a [PrismML MLX fork](https://github.com/PrismML-Eng/mlx) for 1-bit quantization support.

## Requirements

- macOS on Apple Silicon (tested on Macbook Air M4 24GB, macOS 15.7.3 with Metal SDK 26)
- Xcode Metal Toolchain (`setup` installs it automatically)

## Quick start

```sh
make setup   # install uv, venv, deps, download model
make start   # launch server on localhost:8430
make test    # run example queries
make stop    # stop the server
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
| `PORT` | `8430` | Server port |
| `MODEL` | `prism-ml/Bonsai-8B-mlx-1bit` | HuggingFace model ID |

## About this model

| Config | Value | Consequence |
|----------|---------|-------------|
| RoPE (Yarn) | `4x` | RoPE scaling factor (`4x` for 16k x 4 context = 64k by default in model config). Max context window is 64k. |
| Thinking/Reasoning | - | This isn't a reasoning model. |
| Architecture | Dense | This model is based on Qwen3-8B (dense). |
