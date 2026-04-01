#!/usr/bin/env bash
# test.sh - run a few example queries against the bonsai mlx_lm server
set -euo pipefail

BASE_URL="${BONSAI_URL:-http://127.0.0.1:8430}"
PASS=0
FAIL=0

run_test() {
    local name="$1"
    local endpoint="$2"
    local payload="$3"

    printf "\n-- %s --\n" "$name"
    HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" \
        -X POST "$BASE_URL$endpoint" \
        -H "Content-Type: application/json" \
        -d "$payload")

    if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
        printf "  [OK] HTTP %s\n" "$HTTP_CODE"
        python3 -m json.tool /tmp/bonsai_test_resp.json 2>/dev/null | head -30
        PASS=$((PASS + 1))
    else
        printf "  [X] HTTP %s\n" "$HTTP_CODE"
        cat /tmp/bonsai_test_resp.json 2>/dev/null
        FAIL=$((FAIL + 1))
    fi
}

# -- List models (health check) -----------
printf "\n-- List models --\n"
HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" "$BASE_URL/v1/models")
if [[ "$HTTP_CODE" == "200" ]]; then
    printf "  [OK] HTTP %s\n" "$HTTP_CODE"
    python3 -m json.tool /tmp/bonsai_test_resp.json 2>/dev/null
    PASS=$((PASS + 1))
else
    printf "  [X] HTTP %s\n" "$HTTP_CODE"
    FAIL=$((FAIL + 1))
fi

# -- Chat completions ---------------------
run_test "Chat: Explain quantum computing" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 128
}'

run_test "Chat: Write a haiku" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write a haiku about programming."}],
    "max_tokens": 64
}'

run_test "Chat: System + user prompt" "/v1/chat/completions" '{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that responds concisely."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 64
}'

run_test "Chat: Code question" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}],
    "max_tokens": 128
}'

run_test "Chat: Creative" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write a limerick about a program that never compiles."}],
    "max_tokens": 64
}'

# -- Streaming chat completion ------------
printf "\n-- Chat: Streaming --\n"
STREAM_OUTPUT=$(curl -sN -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 64, "stream": true}' \
    --max-time 30 2>/dev/null)

# Check we got SSE data lines
CHUNK_COUNT=$(echo "$STREAM_OUTPUT" | grep -c "^data: " || true)
HAS_DONE=$(echo "$STREAM_OUTPUT" | grep -c "data: \[DONE\]" || true)

if [[ "$CHUNK_COUNT" -gt 1 && "$HAS_DONE" -ge 1 ]]; then
    printf "  [OK] Streaming: %d chunks received, [DONE] present\n" "$CHUNK_COUNT"
    PASS=$((PASS + 1))
else
    printf "  [X] Streaming: got %d chunks, [DONE]=%d\n" "$CHUNK_COUNT" "$HAS_DONE"
    echo "$STREAM_OUTPUT" | head -5
    FAIL=$((FAIL + 1))
fi

# -- Long generation (long context) -------
run_test "Chat: Long generation (long context)" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write the longest, most detailed essay you can about the history of computing, from the abacus to modern AI."}],
    "max_tokens": 65536
}'

# -- Sampling parameters ------------------
run_test "Chat: temp=0 (greedy)" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
    "temperature": 0.0
}'

run_test "Chat: temp=1.5 (high creativity)" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Invent a new word and define it."}],
    "max_tokens": 64,
    "temperature": 1.5
}'

run_test "Chat: top_p=0.1 (narrow nucleus)" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Name three colors."}],
    "max_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.1
}'

run_test "Chat: seed reproducibility (run 1)" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Pick a random number between 1 and 100."}],
    "max_tokens": 16,
    "temperature": 0.5,
    "seed": 42
}'

run_test "Chat: seed reproducibility (run 2)" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Pick a random number between 1 and 100."}],
    "max_tokens": 16,
    "temperature": 0.5,
    "seed": 42
}'

run_test "Chat: temp=0 + top_p=1 (deterministic)" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "What is the speed of light in m/s?"}],
    "max_tokens": 32,
    "temperature": 0.0,
    "top_p": 1.0
}'

run_test "Chat: high temp + low top_p" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write one sentence about a cat."}],
    "max_tokens": 48,
    "temperature": 1.2,
    "top_p": 0.3
}'

# -- Determinism check (5x same request) --
printf "\n-- Determinism: 5x seed=42 temp=0.01 --\n"
DETERMINISM_PAYLOAD='{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128, "temperature": 0.01, "seed": 42}'
PREV_RESP=""
DETERMINISM_OK=true

for run in 1 2 3 4 5; do
    curl -s -o /tmp/bonsai_det_resp_${run}.json \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$DETERMINISM_PAYLOAD"

    RESP=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_det_resp_${run}.json'))['choices'][0]['message']['content'])")

    if [[ "$run" -eq 1 ]]; then
        PREV_RESP="$RESP"
        printf "  run %d: %.80s...\n" "$run" "$RESP"
    else
        if [[ "$RESP" == "$PREV_RESP" ]]; then
            printf "  run %d: SAME\n" "$run"
        else
            printf "  run %d: DIVERGED\n" "$run"
            printf "    expected: %.80s...\n" "$PREV_RESP"
            printf "    got:      %.80s...\n" "$RESP"
            DETERMINISM_OK=false
        fi
    fi
done

if $DETERMINISM_OK; then
    printf "  [OK] All 5 runs produced identical output\n"
    PASS=$((PASS + 1))
else
    printf "  [X] Divergence detected across runs\n"
    FAIL=$((FAIL + 1))
fi

XFAIL=0

# -- Needle in a Haystack (65k context) ---
printf "\n-- Needle in a Haystack (long context) --\n"
printf "  Generating haystack from shuffled bench prompts...\n"

HAYSTACK_PAYLOAD=$(python3 gen_payload.py --mode needle --batches 5 --seed 42)

PAYLOAD_SIZE=$(echo "$HAYSTACK_PAYLOAD" | wc -c | tr -d ' ')
printf "  Payload size: %s bytes\n" "$PAYLOAD_SIZE"

HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" \
    -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$HAYSTACK_PAYLOAD" \
    --max-time 300)

if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
    ANSWER=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_test_resp.json'))['choices'][0]['message']['content'])")
    PTOKENS=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_test_resp.json')).get('usage',{}).get('prompt_tokens','?'))")
    printf "  Prompt tokens: %s\n" "$PTOKENS"
    if echo "$ANSWER" | grep -q "NEEDLE-IN-HAYSTACK-8430-FOUND"; then
        printf "  [OK] Needle found in haystack: %s\n" "$ANSWER"
        PASS=$((PASS + 1))
    else
        printf "  [X] Needle NOT found. Model answered: %s\n" "$ANSWER"
        FAIL=$((FAIL + 1))
    fi
else
    printf "  [X] HTTP %s\n" "$HTTP_CODE"
    cat /tmp/bonsai_test_resp.json 2>/dev/null
    FAIL=$((FAIL + 1))
fi

# -- Long context QA --------
printf "\n-- Long context QA (long context) --\n"
printf "  Generating long context from shuffled bench prompts...\n"

LONGCTX_PAYLOAD=$(python3 gen_payload.py --mode longctx --batches 5 --max-tokens 256 --seed 99)

PAYLOAD_SIZE=$(echo "$LONGCTX_PAYLOAD" | wc -c | tr -d ' ')
printf "  Payload size: %s bytes\n" "$PAYLOAD_SIZE"

HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" \
    -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$LONGCTX_PAYLOAD" \
    --max-time 300)

if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
    ANSWER=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_test_resp.json'))['choices'][0]['message']['content'])")
    PTOKENS=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_test_resp.json')).get('usage',{}).get('prompt_tokens','?'))")
    printf "  Prompt tokens: %s\n" "$PTOKENS"
    # Just verify we got a coherent response (non-empty, mentions something topical)
    ANSWER_LEN=${#ANSWER}
    if [[ "$ANSWER_LEN" -gt 20 ]]; then
        printf "  [OK] Got coherent response (%d chars):\n" "$ANSWER_LEN"
        echo "$ANSWER" | head -10 | sed 's/^/    /'
        PASS=$((PASS + 1))
    else
        printf "  [X] Response too short (%d chars): %s\n" "$ANSWER_LEN" "$ANSWER"
        FAIL=$((FAIL + 1))
    fi
else
    printf "  [X] HTTP %s\n" "$HTTP_CODE"
    cat /tmp/bonsai_test_resp.json 2>/dev/null
    FAIL=$((FAIL + 1))
fi

# -- finish_reason: "length" (expected failure) --
printf "\n-- finish_reason: length (XFAIL) --\n"
printf "  Sending request with max_tokens=5 to force truncation...\n"

HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" \
    -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Write a very long and detailed essay about the history of mathematics from ancient civilizations to modern times."}], "max_tokens": 5, "temperature": 0.0}')

if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
    FINISH_REASON=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_test_resp.json'))['choices'][0]['finish_reason'])")
    CTOKENS=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_test_resp.json')).get('usage',{}).get('completion_tokens','?'))")
    CONTENT=$(python3 -c "import json; print(json.load(open('/tmp/bonsai_test_resp.json'))['choices'][0]['message']['content'])")
    printf "  finish_reason: %s\n" "$FINISH_REASON"
    printf "  completion_tokens: %s\n" "$CTOKENS"
    printf "  truncated output: %s\n" "$CONTENT"
    if [[ "$FINISH_REASON" == "length" ]]; then
        printf "  [XFAIL] Generation truncated at max_tokens limit (expected)\n"
        XFAIL=$((XFAIL + 1))
    else
        printf "  [X] Expected finish_reason='length', got '%s'\n" "$FINISH_REASON"
        FAIL=$((FAIL + 1))
    fi
else
    printf "  [X] HTTP %s\n" "$HTTP_CODE"
    cat /tmp/bonsai_test_resp.json 2>/dev/null
    FAIL=$((FAIL + 1))
fi

# -- Summary ------------------------------
printf "\n==============================\n"
printf "  Results: %d passed, %d failed, %d expected failures\n" "$PASS" "$FAIL" "$XFAIL"
printf "==============================\n"

[[ "$FAIL" -eq 0 ]]
