#!/usr/bin/env python3
"""Generate a needle-in-haystack or long-context payload from shuffled prompts.

Usage:
  python gen_payload.py [--mode needle|longctx] [--batches N] [--max-tokens N] [--seed N]

Defaults produce ~65k tokens (80 batches x 50 prompts).
"""
import argparse, json, random, sys


def parse_args():
    p = argparse.ArgumentParser(description="Generate long-context test payloads")
    p.add_argument("--mode", choices=["needle", "longctx"], default="needle",
                    help="needle = needle-in-haystack retrieval, longctx = topic-frequency QA")
    p.add_argument("--batches", type=int, default=80,
                    help="number of times to shuffle+repeat the 50 prompts (~750 tok/batch)")
    p.add_argument("--max-tokens", type=int, default=64,
                    help="max_tokens for the completion response")
    p.add_argument("--seed", type=int, default=42,
                    help="random seed for shuffling")
    return p.parse_args()


prompts = [
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

needle = "The secret code for project Bonsai is: NEEDLE-IN-HAYSTACK-8430-FOUND"


def gen_needle(batches: int, seed: int, max_tokens: int) -> dict:
    random.seed(seed)
    paragraphs = []
    for batch_num in range(batches):
        batch = prompts[:]
        random.shuffle(batch)
        paragraphs.extend(
            f"Section {batch_num+1}, item {j+1}: {p}" for j, p in enumerate(batch)
        )

    mid = len(paragraphs) // 2
    paragraphs.insert(mid, f"Section {batches//2}, item 26: {needle}")

    haystack = "\n".join(paragraphs)
    return {
        "messages": [
            {"role": "system", "content": "Read the entire document carefully, then answer the question."},
            {"role": "user", "content": (
                "DOCUMENT START\n" + haystack + "\nDOCUMENT END\n\n"
                "Question: What is the secret code for project Bonsai? "
                "Reply with ONLY the code, nothing else."
            )},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def gen_longctx(batches: int, seed: int, max_tokens: int) -> dict:
    random.seed(seed)
    paragraphs = []
    for batch_num in range(batches):
        batch = prompts[:]
        random.shuffle(batch)
        for j, p in enumerate(batch):
            paragraphs.append(f"Audience member {batch_num*len(prompts)+j+1}: {p}")

    document = "\n".join(paragraphs)
    return {
        "messages": [
            {"role": "system", "content": "You are analyzing a transcript of questions asked at a technology conference."},
            {"role": "user", "content": (
                "Here is a transcript of all audience questions:\n\n" + document +
                "\n\nBased on this transcript, what were the three most frequently "
                "asked-about topics? List them as a numbered list."
            )},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


if __name__ == "__main__":
    args = parse_args()
    gen = gen_needle if args.mode == "needle" else gen_longctx
    payload = gen(args.batches, args.seed, args.max_tokens)
    print(json.dumps(payload), file=sys.stdout)