# Recipe: Custom Middleware (External)

Write your own middleware without modifying the framework.

## Overview

`CustomMiddlewareRegistry` is the official plugin system for app developers. Register a factory function at startup, then reference it by key in agent config. No core source modification needed.

## Step 1: Write the Middleware

A middleware is a function `(state, next) => result`:

```typescript
// my-app/middleware/audit-logger.ts
import type { AgentMiddleware } from "langchain";

export function createAuditLogger(config: { level?: string; logFile?: string }): AgentMiddleware {
  const level = config.level || "info";

  return {
    name: "AuditLogger",

    // Runs before each agent step
    beforeAgent: async (state) => {
      console.log(`[${level}] Agent step started, messages:`, state.messages?.length);
      return state;
    },

    // Wraps every LLM call
    wrapModelCall: async (request, handler) => {
      const start = Date.now();
      const result = await handler(request);
      console.log(`[${level}] LLM call took ${Date.now() - start}ms`);
      return result;
    },

    // Runs after each agent step
    afterAgent: async (state) => {
      console.log(`[${level}] Agent step completed`);
      return state;
    },
  };
}
```

## Step 2: Register the Factory

```typescript
// my-app/index.ts — at startup, BEFORE any agent registration
import { CustomMiddlewareRegistry } from "@axiom-lattice/core";
import { createAuditLogger } from "./middleware/audit-logger";

CustomMiddlewareRegistry.register("audit-logger", createAuditLogger);
```

The factory receives the `config` from the agent's middleware config (minus the `key` field).

## Step 3: Use in Agent Config

```typescript
registerAgentLattice({
  type: AgentType.REACT,
  key: "my-agent",
  name: "My Agent",
  prompt: "...",
  modelKey: "azure-gpt-4o",
  tools: ["..."],
  middleware: [
    {
      id: "audit-1",
      type: "custom",
      name: "Audit Logger",
      description: "Logs agent activity",
      enabled: true,
      config: {
        key: "audit-logger",   // matches the registered factory key
        level: "debug",        // passed to factory as config
      },
    },
  ],
});
```

## Complete Examples

### Permission Check Middleware

```typescript
import type { AgentMiddleware } from "langchain";

export function createPermissionCheck(config: { allowedTools: string[] }): AgentMiddleware {
  return {
    name: "PermissionCheck",
    wrapModelCall: async (request, handler) => {
      // Intercept tool calls before execution
      return handler(request);
    },
    afterModel: async (state) => {
      // Check tool calls in the last AI message
      const lastMsg = state.messages?.[state.messages.length - 1];
      if (lastMsg?.tool_calls) {
        for (const tc of lastMsg.tool_calls) {
          if (!config.allowedTools.includes(tc.name)) {
            console.warn(`Blocked tool: ${tc.name}`);
            // Inject an error message
            state.messages.push({
              role: "tool",
              content: `Tool "${tc.name}" is not allowed.`,
              tool_call_id: tc.id,
            });
          }
        }
      }
      return state;
    },
  };
}

// Register:
CustomMiddlewareRegistry.register("permission-check", createPermissionCheck);
```

### Request/Response Logging Middleware

```typescript
export function createRequestLogger(config: { logBody?: boolean }): AgentMiddleware {
  return {
    name: "RequestLogger",

    wrapModelCall: async (request, handler) => {
      const messages = request.messages?.length || 0;
      console.log(`[LLM] Sending ${messages} messages`);

      if (config.logBody) {
        console.log("[LLM] Last message:", request.messages?.[messages - 1]?.content?.slice(0, 200));
      }

      const result = await handler(request);
      console.log(`[LLM] Response length: ${result.content?.length || 0}`);

      return result;
    },
  };
}

CustomMiddlewareRegistry.register("request-logger", createRequestLogger);
```

### Rate Limiting Middleware

```typescript
export function createRateLimiter(config: { maxCallsPerMinute: number }): AgentMiddleware {
  let callCount = 0;
  let windowStart = Date.now();

  return {
    name: "RateLimiter",

    wrapModelCall: async (request, handler) => {
      const now = Date.now();
      if (now - windowStart > 60000) {
        callCount = 0;
        windowStart = now;
      }

      if (callCount >= config.maxCallsPerMinute) {
        throw new Error(`Rate limit exceeded: ${config.maxCallsPerMinute} calls/min`);
      }

      callCount++;
      return handler(request);
    },
  };
}

CustomMiddlewareRegistry.register("rate-limiter", createRateLimiter);
```

## Middleware Lifecycle Hooks

| Hook | When it runs | Use for |
|---|---|---|
| `beforeAgent(state)` | Before each agent step | Initialize state, inject context |
| `wrapModelCall(request, handler)` | Wraps every LLM call | Logging, rate limiting, prompt modification |
| `afterModel(state)` | After LLM generates a response | Check tool calls, modify output |
| `afterAgent(state)` | After each agent step | Cleanup, persistence |

## Gotchas

- **Register BEFORE agent registration** — factories must exist when agents are built
- **Duplicate keys overwrite** — last `register()` wins
- **Unregistered keys are silently skipped** — check `CustomMiddlewareRegistry.has(key)` if unsure
- **`config` passed to factory excludes the `key` field** — the framework strips it before calling
- **Middleware runs in the order listed** in `middleware[]` array
- **Return `state` from hooks** — the modified state flows to the next middleware/tool
- **For heavy async work**, prefer `beforeAgent`/`afterAgent` over `wrapModelCall` (which blocks the LLM)
