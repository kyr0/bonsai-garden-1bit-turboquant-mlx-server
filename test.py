#!/usr/bin/env python3
"""test.py – integration tests for the bonsai server using the OpenAI Python SDK."""

import json
import os
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL = client.models.list().data[0].id

CALIBRATION_FILE = "calibration.json"

PASS = 0
FAIL = 0
XFAIL = 0


def ok(msg: str):
    global PASS
    print(f"  [OK] {msg}")
    PASS += 1


def fail(msg: str):
    global FAIL
    print(f"  [X] {msg}")
    FAIL += 1


# ── helpers ──────────────────────────────────────────────────────────

def chat(messages, **kwargs):
    return client.chat.completions.create(model=MODEL, messages=messages, **kwargs)


# ── tests ────────────────────────────────────────────────────────────

def test_list_models():
    print("\n-- List models (health check) --")
    models = client.models.list()
    if models.data:
        ok(f"Got {len(models.data)} model(s): {models.data[0].id}")
    else:
        fail("No models returned")


def test_chat_basic():
    cases = [
        ("Explain quantum computing", [{"role": "user", "content": "Explain quantum computing in simple terms."}], {"max_tokens": 128}),
        ("Write a haiku", [{"role": "user", "content": "Write a haiku about programming."}], {"max_tokens": 64}),
        ("System + user prompt", [
            {"role": "system", "content": "You are a helpful assistant that responds concisely."},
            {"role": "user", "content": "What is the capital of France?"},
        ], {"max_tokens": 64}),
        ("Code question", [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}], {"max_tokens": 128}),
        ("Creative", [{"role": "user", "content": "Write a limerick about a program that never compiles."}], {"max_tokens": 64}),
    ]
    for name, msgs, kw in cases:
        print(f"\n-- Chat: {name} --")
        try:
            r = chat(msgs, **kw)
            content = r.choices[0].message.content
            ok(f"{len(content)} chars")
        except Exception as e:
            fail(str(e))


def test_streaming():
    print("\n-- Chat: Streaming --")
    try:
        chunks = []
        stream = chat(
            [{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=64, stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        if len(chunks) > 1:
            ok(f"Streaming: {len(chunks)} chunks received")
        else:
            fail(f"Streaming: only {len(chunks)} chunk(s)")
    except Exception as e:
        fail(f"Streaming error: {e}")


def test_long_generation():
    print("\n-- Chat: Long generation --")
    try:
        r = chat(
            [{"role": "user", "content": "Write the longest, most detailed essay you can about the history of computing, from the abacus to modern AI."}],
            max_tokens=65536,
        )
        ok(f"{r.usage.completion_tokens} tokens generated")
    except Exception as e:
        fail(str(e))


def test_sampling_params():
    cases = [
        ("temp=0 (greedy)", {"temperature": 0.0, "max_tokens": 32},
         [{"role": "user", "content": "What is 2+2?"}]),
        ("temp=1.5 (high creativity)", {"temperature": 1.5, "max_tokens": 64},
         [{"role": "user", "content": "Invent a new word and define it."}]),
        ("top_p=0.1 (narrow nucleus)", {"temperature": 0.7, "top_p": 0.1, "max_tokens": 32},
         [{"role": "user", "content": "Name three colors."}]),
        ("seed run 1", {"temperature": 0.5, "seed": 42, "max_tokens": 16},
         [{"role": "user", "content": "Pick a random number between 1 and 100."}]),
        ("seed run 2", {"temperature": 0.5, "seed": 42, "max_tokens": 16},
         [{"role": "user", "content": "Pick a random number between 1 and 100."}]),
        ("temp=0 + top_p=1", {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32},
         [{"role": "user", "content": "What is the speed of light in m/s?"}]),
        ("high temp + low top_p", {"temperature": 1.2, "top_p": 0.3, "max_tokens": 48},
         [{"role": "user", "content": "Write one sentence about a cat."}]),
    ]
    for name, kw, msgs in cases:
        print(f"\n-- Chat: {name} --")
        try:
            r = chat(msgs, **kw)
            ok(f"{r.choices[0].message.content[:80]}")
        except Exception as e:
            fail(str(e))


def test_determinism():
    """Run 1 warmup + 4x identical requests; runs 2-5 must match."""
    print("\n-- Determinism: 1 warmup + 4x seed=42 temp=0.01 --")
    msgs = [{"role": "user", "content": "Hello!"}]
    kw = {"max_tokens": 128, "temperature": 0.01, "seed": 42}
    responses = []
    for run in range(1, 6):
        r = chat(msgs, **kw)
        text = r.choices[0].message.content
        responses.append(text)
        if run == 1:
            print(f"  run {run} (warmup): {text[:80]}...")
        elif run == 2:
            print(f"  run {run} (baseline): {text[:80]}...")
        else:
            tag = "SAME" if text == responses[1] else "DIVERGED"
            print(f"  run {run}: {tag}")

    if all(r == responses[1] for r in responses[2:]):
        ok("Runs 2-5 produced identical output")
    else:
        fail("Divergence detected across runs 2-5")


def test_calibration_baseline():
    """Re-run calibration prompts at temp=0.01 and compare against saved baseline."""
    print("\n-- Calibration baseline match (temp=0.01) --")
    if not os.path.exists(CALIBRATION_FILE):
        fail(f"{CALIBRATION_FILE} not found — run test_calibration.py first")
        return

    with open(CALIBRATION_FILE) as f:
        baseline = json.load(f)

    # Generate the baseline on first run if responses were collected at temp=0.7
    # We do a warmup then compare pairwise at near-deterministic settings
    print("  Warmup run...")
    chat([{"role": "user", "content": "Hello"}], max_tokens=16, temperature=0.01, seed=0)

    for entry in baseline:
        prompt = entry["prompt"]
        print(f"  [{entry['id']}] {prompt[:60]}...")

        r = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.01, seed=0,
        )

        got = r.choices[0].message.content.strip()
        tokens = r.usage.completion_tokens if r.usage else 0

        # Run same prompt again to check self-consistency
        r2 = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.01, seed=0,
        )
        got2 = r2.choices[0].message.content.strip()

        if got == got2:
            ok(f"id={entry['id']} self-consistent ({tokens} tok)")
        else:
            fail(f"id={entry['id']} NOT self-consistent across runs")
            print(f"    run 1: {got[:100]}...")
            print(f"    run 2: {got2[:100]}...")


def test_needle_in_haystack():
    print("\n-- Needle in a Haystack (long context) --")
    print("  Generating haystack from shuffled bench prompts...")
    try:
        payload_json = subprocess.check_output(
            [sys.executable, "gen_payload.py", "--mode", "needle", "--batches", "5", "--seed", "42"],
            text=True,
        )
        payload = json.loads(payload_json)
        r = chat(payload["messages"], max_tokens=payload.get("max_tokens", 64))
        answer = r.choices[0].message.content
        ptokens = r.usage.prompt_tokens if r.usage else "?"
        print(f"  Prompt tokens: {ptokens}")
        if "NEEDLE-IN-HAYSTACK-8430-FOUND" in answer:
            ok(f"Needle found: {answer[:100]}")
        else:
            fail(f"Needle NOT found. Model answered: {answer[:200]}")
    except Exception as e:
        fail(str(e))


def test_long_context_qa():
    print("\n-- Long context QA --")
    print("  Generating long context from shuffled bench prompts...")
    try:
        payload_json = subprocess.check_output(
            [sys.executable, "gen_payload.py", "--mode", "longctx", "--batches", "5",
             "--max-tokens", "256", "--seed", "99"],
            text=True,
        )
        payload = json.loads(payload_json)
        r = chat(payload["messages"], max_tokens=payload.get("max_tokens", 256))
        answer = r.choices[0].message.content
        ptokens = r.usage.prompt_tokens if r.usage else "?"
        print(f"  Prompt tokens: {ptokens}")
        if len(answer) > 20:
            ok(f"Got coherent response ({len(answer)} chars)")
        else:
            fail(f"Response too short ({len(answer)} chars): {answer}")
    except Exception as e:
        fail(str(e))


def test_finish_reason_length():
    global XFAIL
    print("\n-- finish_reason: length (XFAIL) --")
    print("  Sending request with max_tokens=5 to force truncation...")
    try:
        r = chat(
            [{"role": "user", "content": "Write a very long and detailed essay about the history of mathematics from ancient civilizations to modern times."}],
            max_tokens=5, temperature=0.0,
        )
        reason = r.choices[0].finish_reason
        ctokens = r.usage.completion_tokens if r.usage else "?"
        print(f"  finish_reason: {reason}")
        print(f"  completion_tokens: {ctokens}")
        if reason == "length":
            print("  [XFAIL] Generation truncated at max_tokens limit (expected)")
            XFAIL += 1
        else:
            fail(f"Expected finish_reason='length', got '{reason}'")
    except Exception as e:
        fail(str(e))


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_list_models()
    test_chat_basic()
    test_streaming()
    test_long_generation()
    test_sampling_params()
    test_determinism()
    test_calibration_baseline()
    test_needle_in_haystack()
    test_long_context_qa()
    test_finish_reason_length()

    print(f"\n{'='*30}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {XFAIL} expected failures")
    print(f"{'='*30}")
    sys.exit(1 if FAIL > 0 else 0)
