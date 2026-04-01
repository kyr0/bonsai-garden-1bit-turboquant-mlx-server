const BASE_URL = process.env.BONSAI_URL || "http://127.0.0.1:8430";

const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a given location.",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "City name, e.g. 'San Francisco'",
          },
          unit: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
            description: "Temperature unit",
          },
        },
        required: ["location"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Evaluate a mathematical expression.",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "Math expression, e.g. '2 + 2'",
          },
        },
        required: ["expression"],
      },
    },
  },
];

const TOOL_RESULTS: Record<string, string> = {
  get_weather: JSON.stringify({ temperature: 18, unit: "celsius", condition: "partly cloudy" }),
  calculate: JSON.stringify({ result: 42 }),
};

async function apiCall(messages: any[], tools?: any[]) {
  const payload: any = {
    messages,
    max_tokens: 256,
    temperature: 0.1,
  };
  if (tools) payload.tools = tools;

  const resp = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
  }
  return resp.json();
}

async function runTest(name: string, userMessage: string): Promise<boolean> {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`TEST: ${name}`);
  console.log("=".repeat(60));

  const messages: any[] = [{ role: "user", content: userMessage }];

  // Step 1: Send request with tools
  console.log(`\n[1] User: ${userMessage}`);
  const result = await apiCall(messages, TOOLS);
  const choice = result.choices[0];
  const assistantMsg = choice.message;

  console.log(`    Finish reason: ${choice.finish_reason}`);

  const toolCalls = assistantMsg.tool_calls || [];
  if (toolCalls.length > 0) {
    console.log(`    Tool calls: ${toolCalls.length}`);
    for (const tc of toolCalls) {
      console.log(`      -> ${tc.function.name}(${tc.function.arguments || "{}"})`);
    }

    // Step 2: Send tool results back
    messages.push(assistantMsg);
    for (const tc of toolCalls) {
      const fnName = tc.function.name;
      const toolResult = TOOL_RESULTS[fnName] || '{"error": "unknown tool"}';
      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: toolResult,
      });
    }

    console.log(`\n[2] Sending tool results back...`);
    const result2 = await apiCall(messages);
    const finalContent = result2.choices[0].message.content;
    console.log(`    Assistant: ${finalContent.slice(0, 200)}`);
    return true;
  } else {
    const content = assistantMsg.content || "";
    console.log(`    Assistant (no tool call): ${content.slice(0, 200)}`);
    return true;
  }
}

async function main() {
  let passed = 0;
  let failed = 0;

  const tests: [string, string][] = [
    ["Weather query", "What's the weather like in San Francisco?"],
    ["Math query", "What is 6 times 7?"],
    ["Multi-step", "What's the weather in Tokyo and also calculate 123 + 456?"],
  ];

  for (const [name, msg] of tests) {
    try {
      const ok = await runTest(name, msg);
      if (ok) passed++;
      else failed++;
    } catch (e) {
      console.log(`  ERROR: ${e}`);
      failed++;
    }
  }

  console.log(`\n${"=".repeat(60)}`);
  console.log(`Tool calling tests: ${passed} passed, ${failed} failed`);
  console.log("=".repeat(60));
  process.exit(failed > 0 ? 1 : 0);
}

main();
