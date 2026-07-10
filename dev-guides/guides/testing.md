# Testing Guide

How to write and run tests for the Axiom Lattice framework.

---

## Test Framework

- **Runner**: Jest
- **Config**: `jest.config.js` or `jest.config.ts` per package
- **Test location**: `__tests__/` directories or `*.test.ts` files adjacent to source

---

## Running Tests

```bash
# Run all tests across all packages
pnpm test

# Run tests for a specific package
pnpm --filter @axiom-lattice/core test

# Run with watch mode
pnpm --filter @axiom-lattice/core test -- --watch

# Run specific test file
pnpm --filter @axiom-lattice/core test -- path/to/test.test.ts
```

---

## Testing Strategies by Layer

### Layer 0: Core Tests

Test agents, tools, and lattice logic in isolation:

```typescript
// __tests__/weather-tool.test.ts
import { describe, it, expect } from "@jest/globals";

// Test the executor function directly (the 3rd argument to registerToolLattice)
const weatherToolExecutor = async (input: { city: string; unit?: string }) => {
  const temps: Record<string, number> = { tokyo: 20, london: 15 };
  const temp = temps[input.city.toLowerCase()] ?? 18;
  const unit = input.unit || "celsius";
  return `${input.city}: ${temp}°${unit === "fahrenheit" ? "F" : "C"}`;
};

describe("weatherTool", () => {
  it("returns weather for known cities", async () => {
    const result = await weatherToolExecutor({ city: "Tokyo", unit: "celsius" });
    expect(result).toContain("20°C");
  });

  it("defaults to celsius", async () => {
    const result = await weatherToolExecutor({ city: "London" });
    expect(result).toContain("°C");
  });
});
```

**Testing agents** — register agent and test via AgentLatticeManager:

```typescript
import { registerToolLattice, registerAgentLattice, agentLatticeManager } from "@axiom-lattice/core";
import { AgentType } from "@axiom-lattice/protocols";
import type { ReactAgentConfig } from "@axiom-lattice/protocols";

describe("weatherAgent", () => {
  beforeAll(() => {
    // Register tool
    registerToolLattice("get_weather", { name: "get_weather", description: "..." }, myExecutor);
    // Register agent
    registerAgentLattice({
      type: AgentType.REACT,
      key: "test-agent",
      name: "Test Agent",
      prompt: "You are a weather assistant",
      modelKey: "test-model",
      tools: ["get_weather"],
    } as ReactAgentConfig);
  });

  it("is registered and retrievable", () => {
    const config = agentLatticeManager.getAgentConfig("test-agent");
    expect(config?.key).toBe("test-agent");
    expect(config?.tools).toContain("get_weather");
  });
});
});
```

### Layer 1: Gateway Tests

Test HTTP endpoints using Fastify's `inject()` method:

```typescript
import { LatticeGateway } from "@axiom-lattice/gateway";

describe("Gateway API", () => {
  let app: any;

  beforeAll(async () => {
    // LatticeGateway is a plain object, not a builder function.
    // Use the raw Fastify app instance for inject-based testing:
    app = LatticeGateway.app;
    await app.ready();
  });

  afterAll(async () => {
    await app.close();
  });

  it("POST /api/runs creates a run", async () => {
    const response = await app.inject({
      method: "POST",
      url: "/api/runs",
      payload: {
        assistant_id: "test-agent",
        message: "Hello",
      },
    });
    expect(response.statusCode).toBe(200);
    expect(response.json()).toHaveProperty("thread_id");
  });

  it("GET /health returns healthy", async () => {
    const response = await app.inject({
      method: "GET",
      url: "/health",
    });
    expect(response.statusCode).toBe(200);
    expect(response.json().success).toBe(true);
  });
});
```

### Store Tests

Test custom store implementations against the protocol contract:

```typescript
describe("PostgreSQLThreadStore", () => {
  let store: ThreadStore;

  beforeAll(async () => {
    store = new PostgreSQLThreadStore({
      poolConfig: process.env.TEST_DATABASE_URL!,
      autoMigrate: true,
    });
    await store.initialize?.();
  });

  afterAll(async () => {
    await store.dispose?.();
  });

  it("creates and retrieves a thread", async () => {
    // ThreadStore methods require tenantId as FIRST parameter:
    const thread = await store.createThread(
      "test-tenant",
      "test-agent",
      "thread-123",
      { metadata: { title: "Test" } }
    );
    expect(thread.id).toBe("thread-123");

    const retrieved = await store.getThreadById("test-tenant", thread.id);
    expect(retrieved).toBeDefined();
  });
});
```

### Middleware Tests

Test middleware in isolation:

```typescript
import { describe, it, expect } from "@jest/globals";

describe("dateMiddleware", () => {
  it("formats dates correctly", async () => {
    const middleware = createDateMiddleware({ timezone: "Asia/Shanghai" });
    const state = {
      messages: [{ role: "user", content: "What date is it?" }],
      config: {},
    };

    const modified = await middleware(state, async (s) => s);
    expect(modified.messages[0].content).toContain("2024");
  });
});
```

---

## Mock Patterns

### Mocking Tool Executors

```typescript
// Test tool executors in isolation by calling the executor function directly.
// The executor is the 3rd argument to registerToolLattice(key, config, executor).
const mockExecutor = jest.fn().mockResolvedValue("Mocked: Sunny, 22°C");

// Register the tool with mocked executor
registerToolLattice("weather", weatherConfig, mockExecutor);
```

### Mocking Stores

```typescript
// Use InMemory stores for fast tests
import { InMemoryThreadStore } from "@axiom-lattice/core";

const store = new InMemoryThreadStore();
// No database needed — runs entirely in memory
```

### Mocking Environment Variables

```typescript
beforeEach(() => {
  process.env.AZURE_OPENAI_API_KEY = "test-key";
  process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
});

afterEach(() => {
  delete process.env.AZURE_OPENAI_API_KEY;
  delete process.env.DATABASE_URL;
});
```

---

## Integration Tests

For tests that need a real database, use a separate test database:

```bash
# .env.test
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/axiom_lattice_test
```

```typescript
// jest.config.ts
export default {
  testEnvironment: "node",
  setupFiles: ["dotenv/config"],
  setupFilesAfterSetup: ["./jest.setup.ts"], // placed after test framework
};
```

```typescript
// jest.setup.ts — run migrations before integration tests
import { createPgStoreConfig } from "@axiom-lattice/pg-stores";

beforeAll(async () => {
  const stores = createPgStoreConfig(process.env.TEST_DATABASE_URL!);
  // Migrations run automatically via initialize()
  global.__stores__ = stores;
}, 30000);

afterAll(async () => {
  // Clean up test database
  await global.__stores__?.dispose?.();
});
```

---

## Test Naming Conventions

```typescript
// Pattern: describe("<unit>", () => { it("should <behavior> when <condition>", ...) })

describe("ThreadStore", () => {
  describe("createThread", () => {
    it("should create a thread with valid data", ...);
    it("should reject a thread without assistant_id", ...);
    it("should create a unique id for each thread", ...);
  });
});
```

---

## File Location Conventions

```
packages/core/
  src/
    tool_lattice/
      weather.ts
      __tests__/          ← tests in __tests__/ directories
        weather.test.ts
  __tests__/              ← or at package root
    agent.test.ts

packages/pg-stores/
  src/
    __tests__/
      thread-store.test.ts
      workspace-project-store.test.ts
```
