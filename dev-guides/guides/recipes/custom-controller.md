# Recipe: Custom API Controller

Expose custom REST endpoints on the gateway for your frontend to consume.

## Overview

`LatticeGateway.app` is a raw Fastify instance. Add routes directly before calling `startAsHttpEndpoint()`. This is how you expose custom stores, trigger custom logic, or build any backend API your frontend needs.

---

## Step 1: Add a Simple Route

```typescript
import { LatticeGateway } from "@axiom-lattice/gateway";

// Add routes BEFORE starting:
LatticeGateway.app.get("/api/custom/ping", async (request, reply) => {
  return { success: true, data: { message: "pong" } };
});

await LatticeGateway.startAsHttpEndpoint({ port: 4001 });
```

All Fastify methods are available: `.get()`, `.post()`, `.put()`, `.delete()`, `.patch()`.

---

## Step 2: Access Custom Stores from Routes

```typescript
import { LatticeGateway } from "@axiom-lattice/gateway";
import { getStoreLattice } from "@axiom-lattice/core";
import type { AuditLogStore } from "./stores/AuditLogStore";

// GET /api/custom/audit-logs?limit=20
LatticeGateway.app.get("/api/custom/audit-logs", async (request, reply) => {
  const tenantId = request.headers["x-tenant-id"] as string;
  const limit = (request.query as any).limit || 100;

  const store = getStoreLattice("default", "auditLog" as any).store as AuditLogStore;
  const logs = await store.queryLogs(tenantId, limit);

  return { success: true, data: logs };
});

// POST /api/custom/reports/generate
LatticeGateway.app.post("/api/custom/reports/generate", async (request, reply) => {
  const tenantId = request.headers["x-tenant-id"] as string; // from auto-injected header
  const { reportType, params } = request.body as any;
  // custom logic...
  return { success: true, data: { reportId: "rpt-123" } };
});
```

---

## Step 3: Organize with a Route Registrar

For larger apps, extract routes into a separate file:

```typescript
// my-app/routes.ts
import type { FastifyInstance } from "fastify";
import { getStoreLattice } from "@axiom-lattice/core";
import type { AuditLogStore } from "./stores/AuditLogStore";

export function registerMyRoutes(app: FastifyInstance): void {
  app.get("/api/custom/audit-logs", async (request, reply) => {
    const tenantId = request.headers["x-tenant-id"] as string;
    const store = getStoreLattice("default", "auditLog" as any).store as AuditLogStore;
    const logs = await store.queryLogs(tenantId);
    return { success: true, data: logs };
  });

  app.post("/api/custom/audit-logs", async (request, reply) => {
    const tenantId = request.headers["x-tenant-id"] as string; // auto-injected by client
    const { userId, action } = request.body as any;
    const store = getStoreLattice("default", "auditLog" as any).store as AuditLogStore;
    await store.logEvent({ tenantId, userId, action });
    return { success: true };
  });
}
```

```typescript
// index.ts — main entry point
import { LatticeGateway } from "@axiom-lattice/gateway";
import { registerMyRoutes } from "./routes";

registerMyRoutes(LatticeGateway.app);
await LatticeGateway.startAsHttpEndpoint({ port: 4001 });
```

---

## Step 4: Consume from Frontend

`useApi()` returns `{ get, post, put, del, isLoading, error }`. All methods auto-attach auth headers (`Authorization`, `x-tenant-id`, `x-workspace-id`).

```tsx
import { useApi } from "@axiom-lattice/react-sdk";
import { useState, useEffect } from "react";

function AuditLogViewer() {
  const { get, post, isLoading } = useApi();
  const [logs, setLogs] = useState<any[]>([]);

  // GET — typed response
  const fetchLogs = async () => {
    const result = await get<{ success: boolean; data: any[] }>(
      "/api/custom/audit-logs"
    );
    if (result.success) setLogs(result.data);
  };

  // POST — with body (no tenantId needed, it's in headers)
  const addLog = async () => {
    const result = await post<{ success: boolean }>(
      "/api/custom/audit-logs",
      { userId: "u1", action: "manual_log" }
    );
    if (result.success) fetchLogs();
  };

  useEffect(() => { fetchLogs(); }, []);

  return (
    <div>
      <button onClick={addLog} disabled={isLoading}>Add Log</button>
      {logs.map(log => <div key={log.id}>{log.action}</div>)}
    </div>
  );
}
```

### Methods

| Method | Signature |
|---|---|
| `get<T>(url, options?)` | `Promise<T>`. Relative URLs resolve against `baseURL`. |
| `post<T>(url, body?, options?)` | `Promise<T>`. Body is JSON-serialized. |
| `put<T>(url, body?, options?)` | `Promise<T>` |
| `del<T>(url, options?)` | `Promise<T>` |

### Per-request headers

```tsx
const result = await get<{ data: any }>("/api/custom/data", {
  headers: { "x-custom-header": "value" }, // merge into auto-headers
});
```

### Auth handling

Auth headers are automatic — no manual token passing:
- `Authorization: Bearer <token>` from `AxiomLatticeProvider` config
- `x-tenant-id` from current tenant context
- `401` triggers `onUnauthorized` callback or global `lattice:unauthorized` event

---

## Step 5: Full-Stack Wiring (Store → Route → Component)

```
┌─────────────────────────────────────────────────────┐
│ index.ts                                            │
│   configureStores({}, { customStores: { auditLog }})│ → store registered
│   registerMyRoutes(LatticeGateway.app)              │ → routes on Fastify
│   LatticeGateway.startAsHttpEndpoint()              │ → server running
└─────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│ routes.ts                                           │
│   GET /api/custom/audit-logs → store.queryLogs()    │ → reads store
│   POST /api/custom/audit-logs → store.logEvent()    │ → writes store
└─────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│ AuditLogViewer.tsx (React)                          │
│   useApi() → GET /api/custom/audit-logs             │ → fetches data
│   renders log list                                  │ → displays data
└─────────────────────────────────────────────────────┘
```

---

## Request Context (Auth, Tenant)

Standard headers are available on all routes:

| Header | Access |
|---|---|
| `x-tenant-id` | `request.headers["x-tenant-id"]` |
| `x-user-id` | `request.headers["x-user-id"]` |
| `x-workspace-id` | `request.headers["x-workspace-id"]` |
| `x-project-id` | `request.headers["x-project-id"]` |
| `Authorization` | `request.headers["authorization"]` |

For parsed auth, the gateway adds a `user` property:
```typescript
app.get("/api/custom/profile", async (request, reply) => {
  const user = (request as any).user;  // { id, email, tenantId, ... }
  if (!user) return reply.status(401).send({ error: "Unauthorized" });
  return { success: true, data: { userId: user.id } };
});
```

---

## Gotchas

- **Add routes BEFORE `startAsHttpEndpoint()`** — after the server starts, adding routes is possible but may cause hot-reload issues
- **Store access is always runtime** — call `getStoreLattice()` inside the handler, never at module load time
- **Cast custom store types** — `getStoreLattice("default", "customKey" as any).store as MyStore`
- **Fastify APIs are fully available** — you can add hooks, plugins, schema validation, etc.
- **Response convention**: follow the framework's `{ success: true, data: ... }` format for consistency
- For SSE streaming endpoints, use Fastify's reply.raw for raw HTTP access
