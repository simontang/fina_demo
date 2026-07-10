# Recipe: Creating a Custom Tool

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | Your app code | Register tool via `registerToolLattice()` |
| 2 | Agent config | Reference tool key in `tools: ["my_tool"]` |

## Step 1: Register Tool

```typescript
import { registerToolLattice } from "@axiom-lattice/core";
import { z } from "zod";

registerToolLattice(
  "my_tool",  // key — unique snake_case string, referenced by agents
  {
    // ToolConfig:
    name: "my_tool",              // display name
    description: "What this tool does. Be specific — the LLM reads this.",
    schema: z.object({
      param1: z.string().describe("Description of param1"),
      param2: z.number().optional().describe("Optional number parameter"),
    }),

    // Optional flags:
    returnDirect: false,          // output goes directly to user (no LLM wrapping)
    needUserApprove: false,       // require user confirmation before executing
  },
  // Executor function — separate argument from config:
  async (input: { param1: string; param2?: number }) => {
    const result = `Processed: ${input.param1}`;
    return result;
  }
);
```

## Step 2: Validate Input (Built-in)

The `schema` (Zod) validates input automatically. No separate `validate` function needed — the framework validates against the schema before calling the executor.

## Step 3: Reference in Agent

```typescript
// Tools are referenced by their key string, not by object reference
const agentConfig: ReactAgentConfig = {
  type: AgentType.REACT,
  key: "my-agent",
  tools: ["my_tool"],  // string key matching registerToolLattice first argument
};
```

## Tool Description Best Practices

The LLM reads `description` to decide WHEN to call your tool. Be specific:

```typescript
// BAD — too vague
description: "Processes data",

// GOOD — tells the LLM exactly when to use it
description: "Look up current stock price for a given ticker symbol. " +
  "Use this when the user asks about stock prices, market data, " +
  "or company valuations. Returns price, change, and volume.",
```

## Schema Tips

```typescript
schema: z.object({
  // Use .describe() for every field — the LLM reads these
  ticker: z.string().describe("Stock ticker symbol, e.g. 'AAPL', 'GOOGL'"),

  // Use enums for constrained choices
  period: z.enum(["1d", "5d", "1m", "6m", "1y"]).default("1d")
    .describe("Time period for price history"),

  // Use .optional() with .default() when sensible
  currency: z.string().default("USD")
    .describe("Currency code for price display"),
}),
```

## Testing Your Tool

```typescript
// Test the executor function directly:
const myToolExecutor = async (input: { param1: string }) => {
  return `Result: ${input.param1}`;
};

describe("myTool", () => {
  it("processes valid input", async () => {
    const result = await myToolExecutor({ param1: "hello" });
    expect(result).toContain("hello");
  });
});
```

## Gotchas

- Tool keys must be `snake_case` and unique across all registered tools
- The executor return value is passed to the LLM as a string — return meaningful, readable text
- If the executor throws, the error message is shown to the LLM — throw descriptive errors
- `needUserApprove: true` wraps the executor in an approval gate — user confirms before execution
- Heavy tools should handle their own timeouts to avoid blocking the agent
- **Tools must be registered BEFORE the agent** that references them
