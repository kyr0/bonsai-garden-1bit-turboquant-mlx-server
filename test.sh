#!/usr/bin/env bash
# test.sh - run a few example queries against the bonsai server
set -euo pipefail

BASE_URL="${bonsai_URL:-http://127.0.0.1:8430}"
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

# -- Health --------------------------------
printf "\n-- Health check --\n"
HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" "$BASE_URL/health")
if [[ "$HTTP_CODE" == "200" ]]; then
    printf "  [OK] HTTP %s\n" "$HTTP_CODE"
    python3 -m json.tool /tmp/bonsai_test_resp.json 2>/dev/null
    PASS=$((PASS + 1))
else
    printf "  [X] HTTP %s\n" "$HTTP_CODE"
    FAIL=$((FAIL + 1))
fi

# -- List models ---------------------------
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

# -- Plain completions --------------------
run_test "Completion: Continue text" "/v1/completions" '{
    "prompt": "The quick brown fox",
    "max_tokens": 64
}'

run_test "Completion: Code generation" "/v1/completions" '{
    "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "max_tokens": 128
}'

# -- Summary ------------------------------
printf "\n==============================\n"
printf "  Results: %d passed, %d failed\n" "$PASS" "$FAIL"
printf "==============================\n"

[[ "$FAIL" -eq 0 ]]
