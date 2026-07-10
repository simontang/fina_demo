# External Project Setup

Use Axiom Lattice as a dependency in your own project (outside the monorepo).

## Option A: npm Packages (Recommended)

### Step 1: Create Your Project

```bash
mkdir my-axiom-app && cd my-axiom-app
pnpm init
```

### Step 2: Install Dependencies

```json
{
  "name": "my-axiom-app",
  "dependencies": {
    "@axiom-lattice/core": "^2.1.0",
    "@axiom-lattice/gateway": "^2.1.0",
    "@axiom-lattice/protocols": "^2.1.0",
    "@axiom-lattice/pg-stores": "^1.0.0",
    "@axiom-lattice/queue-redis": "^1.0.0",
    "zod": "^3.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "tsx": "^4.0.0"
  }
}
```

```bash
pnpm install
```

### Step 3: Project Structure

```
my-axiom-app/
  src/
    index.ts          # Gateway entry point
    agents/
      my-agent.ts     # Agent configs
    tools/
      my-tool.ts      # Tool definitions
  .env
  package.json
  tsconfig.json
```

### Step 4: Write Your App

```typescript
// src/index.ts
import { LatticeGateway } from "@axiom-lattice/gateway";
import { configureStores, registerAgentLattice, registerToolLattice } from "@axiom-lattice/core";
import { AgentType } from "@axiom-lattice/protocols";
import type { ReactAgentConfig } from "@axiom-lattice/protocols";
import { z } from "zod";

async function main() {
  // 1. Stores
  await configureStores({});

  // 2. Tools
  registerToolLattice(
    "hello",
    { name: "hello", description: "Say hello" },
    async () => "Hello from my app!"
  );

  // 3. Agents
  registerAgentLattice({
    type: AgentType.REACT,
    key: "my-bot",
    name: "My Bot",
    prompt: "You are a helpful assistant.",
    modelKey: "azure-gpt-4o",
    tools: ["hello"],
  } as ReactAgentConfig);

  // 4. Start
  await LatticeGateway.startAsHttpEndpoint({ port: 4001 });
  console.log("Running on http://localhost:4001");
}

main().catch(console.error);
```

### Step 5: Configure & Run

```bash
# .env
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

```bash
pnpm tsx src/index.ts
```

---

## Option B: Monorepo Workspace Reference

If your project lives inside the monorepo (e.g., a new app under `apps/`):

```json
{
  "name": "my-app",
  "dependencies": {
    "@axiom-lattice/core": "workspace:*",
    "@axiom-lattice/gateway": "workspace:*",
    "@axiom-lattice/protocols": "workspace:*"
  }
}
```

The `workspace:*` protocol resolves to the local package. All packages share the same `pnpm install` and build pipeline.

---

## Option C: Frontend Only (React SDK)

For a standalone React frontend that connects to an existing gateway:

```bash
pnpm create vite my-ui --template react-ts
cd my-ui
pnpm add @axiom-lattice/react-sdk @axiom-lattice/protocols
```

```tsx
// src/App.tsx
import { AxiomLatticeProvider, LatticeChatShell } from "@axiom-lattice/react-sdk";

export default function App() {
  return (
    <AxiomLatticeProvider
      config={{
        baseURL: "https://api.example.com",  // your gateway URL
        apiKey: "your-token",
        transport: "sse",
      }}
    >
      <LatticeChatShell
        baseURL="https://api.example.com"
        apiKey="your-token"
        assistantId="my-agent"
        transport="sse"
      />
    </AxiomLatticeProvider>
  );
}
```

---

## Key Imports Cheat Sheet

```typescript
// Core
import { configureStores, registerAgentLattice, registerToolLattice, CustomMiddlewareRegistry, eventBus } from "@axiom-lattice/core";

// Gateway
import { LatticeGateway } from "@axiom-lattice/gateway";

// Types
import { AgentType } from "@axiom-lattice/protocols";
import type { ReactAgentConfig, DeepAgentConfig, WorkflowAgentConfig } from "@axiom-lattice/protocols";

// Stores (optional)
import { createPgStoreConfig } from "@axiom-lattice/pg-stores";

// React (optional)
import { AxiomLatticeProvider, LatticeChatShell, useChat, useAgentState, regsiterElement } from "@axiom-lattice/react-sdk";
```

---

## Gotchas

- All packages must be the **same major version** — mixing versions can cause protocol mismatches
- The gateway **does not auto-start** on import — you must call `LatticeGateway.startAsHttpEndpoint()`
- Environment variables are read at runtime, not bundled — `.env` must be available on the server
- For PostgreSQL, install `@axiom-lattice/pg-stores` and call `configureStores(createPgStoreConfig(url))`
- For Redis queue, install `@axiom-lattice/queue-redis` and set `QUEUE_SERVICE_TYPE=redis`
- The `react-sdk` requires a gateway — it's a client, not a standalone UI
