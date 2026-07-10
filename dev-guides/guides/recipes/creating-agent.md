# Recipe: Creating an Agent

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | Your app code | Define agent config |
| 2 | Your app code | Register tools (optional) |
| 3 | Your app code | Register agent via `registerAgentLattice()` |

## Step 1: Choose Agent Type

| Type | Enum Value | Use When |
|---|---|---|
| `REACT` | `AgentType.REACT` / `"react"` | Single agent with tool-calling loop (most common) |
| `DEEP_AGENT` | `AgentType.DEEP_AGENT` / `"deep_agent"` | Advanced research agent with planning + sub-agents |
| `TEAM` | `AgentType.TEAM` / `"team"` | Multi-agent collaboration with task/mailbox |
| `PROCESSING` | `AgentType.PROCESSING` / `"processing"` | Workflow orchestration with topology enforcement |
| `A2A_REMOTE` | `AgentType.A2A_REMOTE` / `"a2a_remote"` | Bridging an external A2A agent |
| `WORKFLOW` | `AgentType.WORKFLOW` / `"workflow"` | DSL-defined orchestration of multiple agents |

## Step 2: Define Tools (See creating-tool.md)

Tools must be registered BEFORE the agent that references them:
```typescript
registerToolLattice("my_tool", config, executor);
```

## Step 3: Create Agent Config

```typescript
import { AgentType } from "@axiom-lattice/protocols";
import type { ReactAgentConfig } from "@axiom-lattice/protocols";

const myAgentConfig: ReactAgentConfig = {
  type: AgentType.REACT,          // = "react"
  key: "my-agent",                // unique registration key
  name: "My Agent",               // display name
  description: "An agent that helps with X",
  prompt: "You are a helpful assistant that specializes in X...",
  modelKey: "azure-gpt-4o",       // references a registered model lattice key

  // Tool references — string keys, not objects
  tools: ["my_tool1", "my_tool2"],

  // Optional: runtime config injected into tool execution context
  runConfig: {
    databaseKey: "my-db",
  },

  // Optional: input validation schema
  schema: z.object({
    query: z.string(),
  }),

  // Optional: structured output format
  responseFormat: z.object({
    answer: z.string(),
    confidence: z.number(),
  }),

  // Optional: middleware configuration
  middleware: [
    {
      id: "fs-1",
      type: "filesystem",
      name: "Filesystem",
      description: "Filesystem access",
      enabled: true,
      config: { vmIsolation: "agent", modules: ["filesystem"] },
    },
  ],

  // Optional: inherit from parent agent
  extendsAgent: "base-agent-key",
};
```

## Step 4: Register Agent

```typescript
import { registerAgentLattice, registerAgentLatticeWithTenant, registerAgentLattices } from "@axiom-lattice/core";

// Single agent, no tenant
registerAgentLattice(myAgentConfig);

// Single agent, with tenant
registerAgentLatticeWithTenant("my-tenant", myAgentConfig);

// Multiple agents at once
registerAgentLattices("my-tenant", [agent1, agent2, agent3]);
```

## Step 5: Start Gateway

```typescript
import { LatticeGateway } from "@axiom-lattice/gateway";
import { configureStores } from "@axiom-lattice/core";

async function main() {
  // 1. Stores first
  await configureStores({});

  // 2. Register tools and agents
  registerToolLattice("my_tool", toolConfig, toolExecutor);
  registerAgentLattice(myAgentConfig);

  // 3. Start gateway last
  await LatticeGateway.startAsHttpEndpoint({ port: 4001 });
}
```

## Step 6: Test

```bash
curl -X POST http://localhost:4001/api/runs \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "my-agent", "message": "Hello!"}'
```

## Gotchas

- `key` is the unique registration identifier — use in `assistant_id` for API calls
- `modelKey` must match a key registered in `ModelLatticeManager` (e.g., `"azure-gpt-4o"`, `"openai-gpt-4o"`)
- `tools` is `string[]` — tool keys registered via `registerToolLattice(key, ...)`, NOT tool objects
- `type` is lowercase string: `"react"`, `"deep_agent"`, `"team"`, `"processing"`, `"a2a_remote"`, `"workflow"`. Use `AgentType` enum for type safety.
- `WORKFLOW` agents use `workflow: WorkflowDSL` field instead of individual tools
- `TEAM` agents have `maxConcurrency`, `scheduleLatticeKey`, `pollIntervalMs` fields
- `A2A_REMOTE` agents require `agentCardUrl` field
- **New**: `extendsAgent` allows child agents to inherit config from a parent
- **New**: `responseFormat` enables structured (JSON) output responses
- **New**: `runConfig` passes runtime context (e.g., `databaseKey`) to tools
- **Deprecated**: old methods like `registerLattice` on AgentLatticeManager — prefer `registerAgentLatticeWithTenant(tenantId, config)` or `registerAgentLattices(tenantId, configs[])`
