# Recipe: Agent-to-Agent (A2A) Communication

Connect agents across instances using the A2A protocol.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | Gateway config | Expose agents via A2A |
| 2 | Agent config | Configure `A2ARemoteAgentConfig` |
| 3 | `.env` | Set A2A environment variables |

## Step 1: Expose Agent as A2A Endpoint

**Important**: A2A REST routes are NOT registered by default in production. The gateway startup in `packages/gateway/src/index.ts` does NOT call `registerA2ARoutes()`. To enable them, you must call it manually or add it to your gateway startup script:

```typescript
import { registerA2ARoutes } from "@axiom-lattice/gateway";
// After gateway creation:
registerA2ARoutes(app);  // app is the Fastify instance
```

Configure environment:

```bash
# .env
A2A_API_KEYS=key1:tenant1:project1:workspace1
A2A_AGENT_NAME=My Agent
A2A_AGENT_DESCRIPTION=An A2A-compatible agent
```

Test the agent card:
```bash
curl http://localhost:4001/api/a2a/.well-known/agent.json
```

## Step 2: Configure A2A_REMOTE Agent (Client Side)

To call a remote A2A agent from your instance:

```typescript
import { AgentType } from "@axiom-lattice/protocols";
import type { A2ARemoteAgentConfig } from "@axiom-lattice/protocols";
import { registerAgentLattice } from "@axiom-lattice/core";

const remoteAgent: A2ARemoteAgentConfig = {
  type: AgentType.A2A_REMOTE,
  key: "remote-researcher",
  name: "Remote Researcher",
  description: "Research agent on another server",
  prompt: "You are a remote research agent.",
  modelKey: "azure-gpt-4o",

  // A2A-specific fields (flat, NOT nested in a2a: { ... }):
  agentCardUrl: "https://other-server.example.com/api/a2a/.well-known/agent.json",
  apiKey: "key1",           // Must match remote's A2A_API_KEYS
  timeout: 60000,           // 60 second timeout for remote calls
};

registerAgentLattice(remoteAgent);
```

## Step 3: Use in Workflows

```typescript
const workflow: WorkflowAgentConfig = {
  type: AgentType.WORKFLOW,
  key: "orchestrator",
  workflow: {
    version: "1.0",
    steps: [
      {
        type: "agent",
        id: "local_classifier",
        agent: "classifier-agent",
      },
      {
        type: "agent",
        id: "remote_research",
        agent: "remote-researcher",  // A2A_REMOTE agent key
        input: { $ref: "local_classifier.output" },
      },
      {
        type: "agent",
        id: "local_formatter",
        agent: "formatter-agent",
        input: { $ref: "remote_research.output" },
      },
    ],
  },
};
```

## Step 4: A2A CLI Gateway (Alternative)

The `@axiom-lattice/cli-a2a` package provides a standalone A2A gateway for bridging CLI agents:

```bash
# Install
pnpm add @axiom-lattice/cli-a2a

# Run
npx @axiom-lattice/cli-a2a --config a2a-config.json
```

## A2A API Reference

### Remote Server Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/a2a/.well-known/agent.json` | Agent metadata |
| POST | `/api/a2a/tasks/send` | Send task (JSON) |
| GET | `/api/a2a/tasks/:taskId` | Get task status |
| GET | `/api/a2a/tasks/:taskId/stream` | Stream task results (SSE) |
| POST | `/api/a2a/tasks/:taskId/cancel` | Cancel a running task |

## Gotchas

- `A2ARemoteAgentConfig` uses flat field `agentCardUrl` (a URL string), NOT a nested `a2a: { endpoint: "..." }` object
- `A2A_API_KEYS` format: `key1:tenant:project:workspace,key2:tenant:project:workspace`
- The remote server must have matching agent IDs registered
- `A2A_REMOTE` agents can be used in `WORKFLOW` DSL like any other agent
- Timeouts matter — remote agents may take longer than local ones
- For NAT traversal, the CLI A2A gateway supports WebSocket bridging (see `packages/cli-a2a/README.md`)
