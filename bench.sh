#!/usr/bin/env bash
# bench.sh - run 50 generation tasks with varying input and report tok/s
set -euo pipefail
export LC_NUMERIC=C

BASE_URL="${BONSAI_URL:-http://127.0.0.1:8430}"
MAX_TOKENS=128
TOTAL=25
OUTPUT_FILE="/tmp/bonsai_bench_results.jsonl"

# 50 diverse prompts with varying lengths
PROMPTS=(
    "What is 2+2?"
    "Explain quantum computing in simple terms."
    "Write a haiku about the ocean."
    "What is the capital of France?"
    "Describe the water cycle in three sentences."
    "List five prime numbers."
    "Why is the sky blue?"
    "Translate 'hello world' to French."
    "What causes earthquakes?"
    "Write a one-paragraph story about a robot."
    "Explain photosynthesis to a five-year-old."
    "What is the Pythagorean theorem?"
    "Name three programming languages and their main use cases."
    "How does a combustion engine work?"
    "Write a limerick about a cat."
    "What is machine learning?"
    "Describe the solar system briefly."
    "What are the primary colors?"
    "Explain how a blockchain works in simple terms."
    "What is the speed of light?"
    "Write a short poem about autumn leaves falling gently to the ground on a quiet Sunday morning."
    "Summarize the plot of Romeo and Juliet in two sentences."
    "What is the difference between a stack and a queue in computer science?"
    "How do vaccines work?"
    "What is the Fibonacci sequence?"
    "Explain the concept of supply and demand."
    "What are the states of matter?"
    "Describe how a neural network learns."
    "What is the greenhouse effect and why does it matter for our planet?"
    "Write a four-line rhyming verse about programming bugs."
    "What is DNA and what role does it play in living organisms?"
    "Explain the theory of relativity in plain English so that a high school student could understand it."
    "How does GPS work?"
    "What are the differences between TCP and UDP?"
    "Describe the process of making bread from scratch."
    "What is the Turing test?"
    "Explain how compilers work."
    "What is entropy in thermodynamics?"
    "Write a brief product description for a smart water bottle."
    "What causes the seasons on Earth?"
    "Explain the concept of recursion with a simple example in Python."
    "What is the Big Bang theory?"
    "How do airplanes generate lift?"
    "What is the difference between AI, ML, and deep learning?"
    "Describe three sorting algorithms and their time complexities."
    "What is CRISPR and how is it used in gene editing to modify DNA sequences in living organisms?"
    "Explain how public-key cryptography works and why it is fundamental to secure communication on the internet."
    "What are the main differences between Python and Rust in terms of memory safety and performance?"
    "Write a detailed step-by-step explanation of how to implement a simple HTTP server from scratch."
    "Describe the history and evolution of artificial intelligence from the 1950s Dartmouth conference to modern large language models."
)

# Verify server is up
if ! curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
    echo "ERROR: Server not reachable at $BASE_URL. Run 'make start' first."
    exit 1
fi

echo "Bonsai Benchmark - $TOTAL requests, max_tokens=$MAX_TOKENS"
echo "Server: $BASE_URL"
echo "=========================================="

> "$OUTPUT_FILE"

total_prompt_tokens=0
total_completion_tokens=0
total_gen_time=0
success=0
fail=0

for i in $(seq 1 $TOTAL); do
    prompt="${PROMPTS[$((i - 1))]}"
    # Escape prompt for JSON
    json_prompt=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$prompt")

    payload="{\"messages\": [{\"role\": \"user\", \"content\": $json_prompt}], \"max_tokens\": $MAX_TOKENS}"

    t_start=$(python3 -c "import time; print(time.time())")

    HTTP_CODE=$(curl -s -o /tmp/bonsai_bench_resp.json -w "%{http_code}" \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)

    t_end=$(python3 -c "import time; print(time.time())")

    if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
        elapsed=$(python3 -c "print(round($t_end - $t_start, 3))")
        ptok=$(python3 -c "import json; d=json.load(open('/tmp/bonsai_bench_resp.json')); print(d.get('usage',{}).get('prompt_tokens',0))")
        ctok=$(python3 -c "import json; d=json.load(open('/tmp/bonsai_bench_resp.json')); print(d.get('usage',{}).get('completion_tokens',0))")
        e2e_tps=$(python3 -c "e=$elapsed; c=$ctok; print(round(c/e, 1) if e > 0 else 0)")

        printf "  [%2d/%d] e2e=%5.2fs  prompt=%-4s compl=%-4s  e2e=%6.1f tok/s  %.40s...\n" \
            "$i" "$TOTAL" "$elapsed" "$ptok" "$ctok" "$e2e_tps" "$prompt"

        total_prompt_tokens=$((total_prompt_tokens + ptok))
        total_completion_tokens=$((total_completion_tokens + ctok))
        total_gen_time=$(python3 -c "print($total_gen_time + $elapsed)")
        success=$((success + 1))

        echo "{\"i\":$i,\"e2e_s\":$elapsed,\"prompt_tokens\":$ptok,\"completion_tokens\":$ctok,\"e2e_tok_s\":$e2e_tps}" >> "$OUTPUT_FILE"
    else
        printf "  [%2d/%d] FAIL HTTP %s  %.40s...\n" "$i" "$TOTAL" "$HTTP_CODE" "$prompt"
        fail=$((fail + 1))
    fi
done

echo ""
echo "=========================================="
echo "Results: $success succeeded, $fail failed"
echo "Total prompt tokens:     $total_prompt_tokens"
echo "Total completion tokens:  $total_completion_tokens"
avg_tps=$(python3 -c "t=$total_gen_time; c=$total_completion_tokens; print(round(c/t, 1) if t > 0 else 0)")
avg_latency=$(python3 -c "t=$total_gen_time; n=$success; print(round(t/n, 2) if n > 0 else 0)")
echo "Total e2e time:           ${total_gen_time}s"
echo "Avg e2e latency/request:  ${avg_latency}s"
echo "Avg e2e throughput:       ${avg_tps} tok/s"
echo "=========================================="
echo "Raw results: $OUTPUT_FILE"
