#!/usr/bin/env python3
"""bench.py - benchmark streaming and non-streaming generation against the server."""

import argparse
import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8430/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "default_model")
if DEFAULT_MODEL == "default":
    DEFAULT_MODEL = "default_model"

DEFAULT_MAX_TOKENS = 128
DEFAULT_TOTAL = 25
DEFAULT_STREAM_OUTPUT_FILE = "/tmp/bonsai_bench_stream_results.jsonl"
DEFAULT_NONSTREAM_OUTPUT_FILE = "/tmp/bonsai_bench_results.jsonl"

PROMPTS = [
    "What is 2+2?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "What is the capital of France?",
    "Describe the water cycle in three sentences.",
    "List five prime numbers.",
    "Why is the sky blue?",
    "Translate 'hello world' to French.",
    "What causes earthquakes?",
    "Write a one-paragraph story about a robot.",
    "Explain photosynthesis to a five-year-old.",
    "What is the Pythagorean theorem?",
    "Name three programming languages and their main use cases.",
    "How does a combustion engine work?",
    "Write a limerick about a cat.",
    "What is machine learning?",
    "Describe the solar system briefly.",
    "What are the primary colors?",
    "Explain how a blockchain works in simple terms.",
    "What is the speed of light?",
    "Write a short poem about autumn leaves falling gently to the ground on a quiet Sunday morning.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the difference between a stack and a queue in computer science?",
    "How do vaccines work?",
    "What is the Fibonacci sequence?",
    "Explain the concept of supply and demand.",
    "What are the states of matter?",
    "Describe how a neural network learns.",
    "What is the greenhouse effect and why does it matter for our planet?",
    "Write a four-line rhyming verse about programming bugs.",
    "What is DNA and what role does it play in living organisms?",
    "Explain the theory of relativity in plain English so that a high school student could understand it.",
    "How does GPS work?",
    "What are the differences between TCP and UDP?",
    "Describe the process of making bread from scratch.",
    "What is the Turing test?",
    "Explain how compilers work.",
    "What is entropy in thermodynamics?",
    "Write a brief product description for a smart water bottle.",
    "What causes the seasons on Earth?",
    "Explain the concept of recursion with a simple example in Python.",
    "What is the Big Bang theory?",
    "How do airplanes generate lift?",
    "What is the difference between AI, ML, and deep learning?",
    "Describe three sorting algorithms and their time complexities.",
    "What is CRISPR and how is it used in gene editing to modify DNA sequences in living organisms?",
    "Explain how public-key cryptography works and why it is fundamental to secure communication on the internet.",
    "What are the main differences between Python and Rust in terms of memory safety and performance?",
    "Write a detailed step-by-step explanation of how to implement a simple HTTP server from scratch.",
    "Describe the history and evolution of artificial intelligence from the 1950s Dartmouth conference to modern large language models.",
]

def _safe_div(numerator: float | int | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / denominator


def _format_metric(value: float | int | None, *, width: int = 6, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return f"{'n/a':>{width}}{suffix}"
    return f"{value:>{width}.{precision}f}{suffix}"


def _make_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _request_payload(prompt: str, max_tokens: int, model: str, *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def _extract_stream_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _iter_prompts(total: int) -> list[str]:
    return [PROMPTS[index % len(PROMPTS)] for index in range(total)]


def run_stream_benchmark(total: int, max_tokens: int, output_file: str, *, chat_url: str, model: str) -> None:
    print(f"Bonsai Benchmark - streaming - {total} requests, max_tokens={max_tokens}")
    print(f"Server: {chat_url}")
    print(f"Model:  {model}")
    print("=" * 80)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_elapsed = 0.0
    total_decode_time = 0.0
    total_ttft = 0.0
    ttft_count = 0
    success = 0
    fail = 0

    prompts = _iter_prompts(total)
    timeout = httpx.Timeout(180.0, connect=10.0)

    with httpx.Client(timeout=timeout, headers=_make_headers()) as client:
        with open(output_file, "w") as out:
            for index, prompt in enumerate(prompts, start=1):
                text_parts: list[str] = []
                usage: dict[str, Any] | None = None
                first_token_at: float | None = None
                chunk_count = 0
                started_at = time.perf_counter()

                try:
                    with client.stream("POST", chat_url, json=_request_payload(prompt, max_tokens, model, stream=True)) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            if isinstance(line, bytes):
                                line = line.decode("utf-8", "replace")
                            if not line.startswith("data: "):
                                continue

                            data = line[6:]
                            if data == "[DONE]":
                                break

                            chunk = json.loads(data)
                            if isinstance(chunk.get("usage"), dict):
                                usage = chunk["usage"]
                                continue

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue

                            delta = choices[0].get("delta") or {}
                            text = _extract_stream_text(delta.get("content"))
                            if not text:
                                continue

                            if first_token_at is None:
                                first_token_at = time.perf_counter()

                            text_parts.append(text)
                            chunk_count += 1

                    elapsed = time.perf_counter() - started_at
                    ttft = first_token_at - started_at if first_token_at is not None else None
                    decode_time = elapsed - ttft if ttft is not None else None
                    prompt_tokens = usage.get("prompt_tokens") if usage else None
                    completion_tokens = usage.get("completion_tokens") if usage else None
                    turn_tok_s = _safe_div(completion_tokens, elapsed)
                    decode_tok_s = _safe_div(completion_tokens, decode_time)
                    text = "".join(text_parts)

                    print(
                        f"  [{index:2d}/{total}] ttft={_format_metric(ttft, precision=2, suffix='s')}  "
                        f"total={_format_metric(elapsed, precision=2, suffix='s')}  "
                        f"prompt={str(prompt_tokens) if prompt_tokens is not None else 'n/a':>4}  "
                        f"compl={str(completion_tokens) if completion_tokens is not None else 'n/a':>4}  "
                        f"turn={_format_metric(turn_tok_s, precision=1, suffix=' tok/s')}  "
                        f"decode={_format_metric(decode_tok_s, precision=1, suffix=' tok/s')}  "
                        f"chunks={chunk_count:>3}  chars={len(text):>4}"
                    )
                    print(f"           prompt: {prompt}")

                    if prompt_tokens is not None:
                        total_prompt_tokens += prompt_tokens
                    if completion_tokens is not None:
                        total_completion_tokens += completion_tokens
                    total_elapsed += elapsed
                    if decode_time is not None and decode_time > 0:
                        total_decode_time += decode_time
                    if ttft is not None:
                        total_ttft += ttft
                        ttft_count += 1
                    success += 1

                    out.write(
                        json.dumps(
                            {
                                "i": index,
                                "mode": "stream",
                                "prompt": prompt,
                                "ttft_s": round(ttft, 4) if ttft is not None else None,
                                "elapsed_s": round(elapsed, 4),
                                "decode_s": round(decode_time, 4) if decode_time is not None else None,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "turn_tok_s": round(turn_tok_s, 4) if turn_tok_s is not None else None,
                                "decode_tok_s": round(decode_tok_s, 4) if decode_tok_s is not None else None,
                                "chunk_count": chunk_count,
                                "text_chars": len(text),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                except Exception as exc:
                    print(f"  [{index:2d}/{total}] FAIL: {exc}")
                    print(f"           prompt: {prompt}")
                    fail += 1
                    out.write(
                        json.dumps(
                            {
                                "i": index,
                                "mode": "stream",
                                "prompt": prompt,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    avg_ttft = _safe_div(total_ttft, ttft_count)
    avg_elapsed = _safe_div(total_elapsed, success)
    avg_turn_tok_s = _safe_div(total_completion_tokens, total_elapsed)
    avg_decode_tok_s = _safe_div(total_completion_tokens, total_decode_time)

    print()
    print("=" * 80)
    print(f"Results: {success} succeeded, {fail} failed")
    print(f"Total prompt tokens:      {total_prompt_tokens}")
    print(f"Total completion tokens:  {total_completion_tokens}")
    print(f"Total elapsed time:       {total_elapsed:.1f}s")
    print(f"Avg TTFT/request:         {_format_metric(avg_ttft, precision=2, suffix='s', width=0).strip()}")
    print(f"Avg elapsed/request:      {_format_metric(avg_elapsed, precision=2, suffix='s', width=0).strip()}")
    print(f"Avg turn throughput:      {_format_metric(avg_turn_tok_s, precision=1, suffix=' tok/s', width=0).strip()}")
    print(f"Avg decode throughput:    {_format_metric(avg_decode_tok_s, precision=1, suffix=' tok/s', width=0).strip()}")
    print("=" * 80)
    print(f"Raw results: {output_file}")


def run_nonstream_benchmark(total: int, max_tokens: int, output_file: str, *, chat_url: str, model: str) -> None:
    print(f"Bonsai Benchmark - non-streaming - {total} requests, max_tokens={max_tokens}")
    print(f"Server: {chat_url}")
    print(f"Model:  {model}")
    print("=" * 80)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_elapsed = 0.0
    success = 0
    fail = 0

    prompts = _iter_prompts(total)
    timeout = httpx.Timeout(180.0, connect=10.0)

    with httpx.Client(timeout=timeout, headers=_make_headers()) as client:
        with open(output_file, "w") as out:
            for index, prompt in enumerate(prompts, start=1):
                started_at = time.perf_counter()
                try:
                    response = client.post(chat_url, json=_request_payload(prompt, max_tokens, model, stream=False))
                    response.raise_for_status()
                    payload = response.json()
                    elapsed = time.perf_counter() - started_at
                    usage = payload.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    turn_tok_s = _safe_div(completion_tokens, elapsed)
                    content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

                    print(
                        f"  [{index:2d}/{total}] total={_format_metric(elapsed, precision=2, suffix='s')}  "
                        f"prompt={str(prompt_tokens) if prompt_tokens is not None else 'n/a':>4}  "
                        f"compl={str(completion_tokens) if completion_tokens is not None else 'n/a':>4}  "
                        f"turn={_format_metric(turn_tok_s, precision=1, suffix=' tok/s')}  chars={len(content):>4}"
                    )
                    print(f"           prompt: {prompt}")

                    if prompt_tokens is not None:
                        total_prompt_tokens += prompt_tokens
                    if completion_tokens is not None:
                        total_completion_tokens += completion_tokens
                    total_elapsed += elapsed
                    success += 1

                    out.write(
                        json.dumps(
                            {
                                "i": index,
                                "mode": "nonstream",
                                "prompt": prompt,
                                "elapsed_s": round(elapsed, 4),
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "turn_tok_s": round(turn_tok_s, 4) if turn_tok_s is not None else None,
                                "text_chars": len(content),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                except Exception as exc:
                    print(f"  [{index:2d}/{total}] FAIL: {exc}")
                    print(f"           prompt: {prompt}")
                    fail += 1
                    out.write(
                        json.dumps(
                            {
                                "i": index,
                                "mode": "nonstream",
                                "prompt": prompt,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    avg_elapsed = _safe_div(total_elapsed, success)
    avg_turn_tok_s = _safe_div(total_completion_tokens, total_elapsed)

    print()
    print("=" * 80)
    print(f"Results: {success} succeeded, {fail} failed")
    print(f"Total prompt tokens:      {total_prompt_tokens}")
    print(f"Total completion tokens:  {total_completion_tokens}")
    print(f"Total elapsed time:       {total_elapsed:.1f}s")
    print(f"Avg elapsed/request:      {_format_metric(avg_elapsed, precision=2, suffix='s', width=0).strip()}")
    print(f"Avg turn throughput:      {_format_metric(avg_turn_tok_s, precision=1, suffix=' tok/s', width=0).strip()}")
    print("=" * 80)
    print(f"Raw results: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Bonsai server")
    parser.add_argument("--mode", choices=["stream", "nonstream"], default="stream", help="Benchmark streaming or non-streaming responses")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL, for example http://127.0.0.1:8430/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to send in requests")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="Number of requests to send")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens for each request")
    parser.add_argument("--output-file", default=None, help="Where to write JSONL metrics")
    args = parser.parse_args()

    total = max(1, args.total)
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    output_file = args.output_file or (
        DEFAULT_STREAM_OUTPUT_FILE if args.mode == "stream" else DEFAULT_NONSTREAM_OUTPUT_FILE
    )

    if args.mode == "stream":
        run_stream_benchmark(total=total, max_tokens=args.max_tokens, output_file=output_file, chat_url=chat_url, model=args.model)
        return

    run_nonstream_benchmark(total=total, max_tokens=args.max_tokens, output_file=output_file, chat_url=chat_url, model=args.model)


if __name__ == "__main__":
    main()
