# Build Your First Agent

Step-by-step from zero to a working custom agent.

## Overview

You'll build:
1. A custom tool (weather lookup)
2. A custom agent that uses it
3. Register both and start the gateway
4. Call the agent via API

## Step 1: Project Structure

```
my-app/
  index.ts              # Gateway entry point
  .env
  package.json
```

Your `package.json`:
```json
{
  "name": "my-agent-app",
  "dependencies": {
    "@axiom-lattice/core": "workspace:*",
    "@axiom-lattice/gateway": "workspace:*",
    "@axiom-lattice/protocols": "workspace:*"
  }
}
```

## Step 2: Define and Register a Custom Tool

File: `index.ts` (or `tools/weather.ts`)

```typescript
import { registerToolLattice } from "@axiom-lattice/core";
import { z } from "zod";

// Register tool: key, config, executor
registerToolLattice(
  "get_weather",  // key — agents reference this string
  {
    name: "get_weather",
    description: "Get current weather for a city",
    schema: z.object({
      city: z.string().describe("City name, e.g. 'Beijing'"),
      unit: z.enum(["celsius", "fahrenheit"]).default("celsius"),
    }),
  },
  async ({ city, unit }: { city: string; unit: string }) => {
    // In production, call a real weather API
    const temps: Record<string, number> = {
      beijing: 22,
      shanghai: 25,
      tokyo: 20,
      london: 15,
      "new york": 18,
    };
    const temp = temps[city.toLowerCase()] ?? 18;
    const display = unit === "fahrenheit"
      ? `${temp * 9 / 5 + 32}°F`
      : `${temp}°C`;
    return `Current weather in ${city}: ${display}, partly cloudy`;
  }
);
```

## Step 3: Create and Register an Agent

```typescript
import { registerAgentLattice } from "@axiom-lattice/core";
import { AgentType } from "@axiom-lattice/protocols";
import type { ReactAgentConfig } from "@axiom-lattice/protocols";

const weatherAgentConfig: ReactAgentConfig = {
  type: AgentType.REACT,           // = "react"
  key: "weather-bot",              // unique key
  name: "Weather Bot",
  description: "A weather information assistant",
  modelKey: "azure-gpt-4o",
  prompt: `You are a helpful weather assistant.
When users ask about weather, use the get_weather tool.
Always confirm the city name before calling the tool.
Respond in a friendly, conversational tone.`,
  tools: ["get_weather"],          // tool keys (strings), not objects
};

registerAgentLattice(weatherAgentConfig);
```

## Step 4: Gateway Entry Point

```typescript
// index.ts (complete)
import { LatticeGateway } from "@axiom-lattice/gateway";
import { configureStores, registerToolLattice, registerAgentLattice } from "@axiom-lattice/core";
import { AgentType } from "@axiom-lattice/protocols";
import type { ReactAgentConfig } from "@axiom-lattice/protocols";
import { z } from "zod";

async function main() {
  // 1. Register tools
  registerToolLattice(
    "get_weather",
    {
      name: "get_weather",
      description: "Get current weather for a city",
      schema: z.object({
        city: z.string().describe("City name"),
        unit: z.enum(["celsius", "fahrenheit"]).default("celsius"),
      }),
    },
    async ({ city, unit }) => {
      const temps: Record<string, number> = { beijing: 22, shanghai: 25, tokyo: 20 };
      const temp = temps[city.toLowerCase()] ?? 18;
      return `${city}: ${unit === "fahrenheit" ? temp * 9/5 + 32 + "°F" : temp + "°C"}`;
    }
  );

  // 2. Register agent
  const config: ReactAgentConfig = {
    type: AgentType.REACT,
    key: "weather-bot",
    name: "Weather Bot",
    description: "Weather assistant",
    modelKey: "azure-gpt-4o",
    prompt: "You are a weather assistant. Use get_weather tool.",
    tools: ["get_weather"],
  };
  registerAgentLattice(config);

  // 3. Configure stores (in-memory for dev)
  await configureStores({});

  // 4. Start gateway
  await LatticeGateway.startAsHttpEndpoint({ port: 4001 });
  console.log("Gateway running on http://localhost:4001");
}

main().catch(console.error);
```

## Step 5: Configure LLM

In `.env`:

```bash
# Pick one:
AZURE_OPENAI_API_KEY=sk-...
# or
OPENAI_API_KEY=sk-...
```

## Step 6: Test

```bash
# Start
pnpm tsx index.ts
```

```bash
# Call agent
curl -X POST http://localhost:4001/api/runs \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "weather-bot", "content": "What is the weather in Tokyo?"}'
```

## Step 7: Add a Second Tool

```typescript
// Register tool
registerToolLattice(
  "get_forecast",
  {
    name: "get_forecast",
    description: "Get 3-day weather forecast",
    schema: z.object({ city: z.string() }),
  },
  async ({ city }) => `Forecast for ${city}: Mon sunny, Tue rain, Wed cloudy`
);

// Update agent config — add tool key to the array:
const config: ReactAgentConfig = {
  // ...
  tools: ["get_weather", "get_forecast"],  // add tool keys
};
```

## Next Steps

- [Add more tools](recipes/creating-tool.md)
- [Orchestrate multi-agent workflows](recipes/workflow-dsl.md)
- [Add a custom UI](recipes/ui-customization.md)
- [Deploy to production](deployment.md)
