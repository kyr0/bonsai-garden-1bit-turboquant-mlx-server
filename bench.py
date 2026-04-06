#!/usr/bin/env python3
"""bench.py – benchmark generation throughput against the server."""

import json
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL = client.models.list().data[0].id

MAX_TOKENS = 128
OUTPUT_FILE = "/tmp/bonsai_bench_results.jsonl"

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

TOTAL = 25

print(f"Bonsai Benchmark - {TOTAL} requests, max_tokens={MAX_TOKENS}")
print(f"Server: {client.base_url}")
print(f"Model:  {MODEL}")
print("=" * 50)

total_prompt_tokens = 0
total_completion_tokens = 0
total_gen_time = 0.0
success = 0
fail = 0

with open(OUTPUT_FILE, "w") as out:
    for i in range(TOTAL):
        prompt = PROMPTS[i]
        try:
            t0 = time.perf_counter()
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
            elapsed = round(time.perf_counter() - t0, 3)

            ptok = r.usage.prompt_tokens if r.usage else 0
            ctok = r.usage.completion_tokens if r.usage else 0
            tps = round(ctok / elapsed, 1) if elapsed > 0 else 0

            print(f"  [{i+1:2d}/{TOTAL}] e2e={elapsed:5.2f}s  prompt={ptok:<4}  compl={ctok:<4}  e2e={tps:6.1f} tok/s  {prompt[:40]}...")

            total_prompt_tokens += ptok
            total_completion_tokens += ctok
            total_gen_time += elapsed
            success += 1

            out.write(json.dumps({"i": i + 1, "e2e_s": elapsed, "prompt_tokens": ptok, "completion_tokens": ctok, "e2e_tok_s": tps}) + "\n")
        except Exception as e:
            print(f"  [{i+1:2d}/{TOTAL}] FAIL: {e}  {prompt[:40]}...")
            fail += 1

print()
print("=" * 50)
print(f"Results: {success} succeeded, {fail} failed")
print(f"Total prompt tokens:      {total_prompt_tokens}")
print(f"Total completion tokens:  {total_completion_tokens}")
avg_tps = round(total_completion_tokens / total_gen_time, 1) if total_gen_time > 0 else 0
avg_latency = round(total_gen_time / success, 2) if success > 0 else 0
print(f"Total e2e time:           {total_gen_time:.1f}s")
print(f"Avg e2e latency/request:  {avg_latency}s")
print(f"Avg e2e throughput:       {avg_tps} tok/s")
print("=" * 50)
print(f"Raw results: {OUTPUT_FILE}")
