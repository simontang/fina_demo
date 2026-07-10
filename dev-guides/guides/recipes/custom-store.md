# Recipe: Custom Store (Database Backend)

Implement a custom store to use a different database (MongoDB, MySQL, DynamoDB, etc.).

## Overview

All stores follow the same pattern:
1. **Read the protocol** in `packages/protocols/src/`
2. **Implement the interface** for your database
3. **Register** via `configureStores()`
4. **Optional**: implement `initialize()` and `dispose()` for lifecycle

**Critical**: ALL store methods take `tenantId` as their FIRST parameter. This is mandatory for multi-tenant isolation.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `packages/protocols/src/ThreadStoreProtocol.ts` | Read the real interface |
| 2 | `your-stores/MongoThreadStore.ts` | Implement the interface |
| 3 | Gateway startup | Register via `configureStores()` |

## Step 1: Read the Protocol

File: `packages/protocols/src/ThreadStoreProtocol.ts`

```typescript
export interface ThreadStore {
  // ALL methods require tenantId as first parameter

  getThreadsByAssistantId(
    tenantId: string,        // REQUIRED first param
    assistantId: string,
    metadataFilter?: Record<string, string>
  ): Promise<Thread[]>;

  getThreadById(
    tenantId: string,        // REQUIRED first param
    threadId: string
  ): Promise<Thread | undefined>;

  createThread(
    tenantId: string,        // REQUIRED first param
    assistantId: string,
    threadId: string,
    data: CreateThreadRequest  // { metadata?: Record<string, any> }
  ): Promise<Thread>;

  updateThread(
    tenantId: string,        // REQUIRED first param
    threadId: string,
    updates: Partial<CreateThreadRequest>
  ): Promise<Thread | null>;

  deleteThread(
    tenantId: string,        // REQUIRED first param
    threadId: string
  ): Promise<boolean>;

  hasThread(
    tenantId: string,        // REQUIRED first param
    threadId: string
  ): Promise<boolean>;

  // Lifecycle (auto-detected by configureStores)
  initialize?(): Promise<void>;
  dispose?(): Promise<void>;
}

interface Thread {
  id: string;
  tenantId: string;
  assistantId: string;
  metadata?: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

interface CreateThreadRequest {
  metadata?: Record<string, any>;
}
```

## Step 2: Implement for Your Database

```typescript
import type { Thread, ThreadStore, CreateThreadRequest } from "@axiom-lattice/protocols";
import { MongoClient, Db, Collection } from "mongodb";

export class MongoThreadStore implements ThreadStore {
  private db!: Db;
  private collection!: Collection;
  private client!: MongoClient;

  constructor(private connectionString: string, private dbName: string) {}

  async initialize(): Promise<void> {
    this.client = new MongoClient(this.connectionString);
    await this.client.connect();
    this.db = this.client.db(this.dbName);
    this.collection = this.db.collection("threads");
    await this.collection.createIndex({ tenantId: 1, assistantId: 1 });
  }

  async dispose(): Promise<void> { await this.client.close(); }

  async getThreadsByAssistantId(
    tenantId: string,
    assistantId: string,
    metadataFilter?: Record<string, string>
  ): Promise<Thread[]> {
    const filter: any = { tenantId, assistantId };
    if (metadataFilter) {
      for (const [k, v] of Object.entries(metadataFilter)) {
        filter[`metadata.${k}`] = v;
      }
    }
    return this.collection.find(filter).toArray() as unknown as Thread[];
  }

  async getThreadById(tenantId: string, threadId: string): Promise<Thread | undefined> {
    const doc = await this.collection.findOne({ tenantId, id: threadId });
    return doc as unknown as Thread | undefined;
  }

  async createThread(
    tenantId: string,
    assistantId: string,
    threadId: string,
    data: CreateThreadRequest
  ): Promise<Thread> {
    const thread: Thread = {
      id: threadId,
      tenantId,
      assistantId,
      metadata: data.metadata,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    await this.collection.insertOne(thread as any);
    return thread;
  }

  async updateThread(
    tenantId: string,
    threadId: string,
    updates: Partial<CreateThreadRequest>
  ): Promise<Thread | null> {
    const result = await this.collection.findOneAndUpdate(
      { tenantId, id: threadId },
      { $set: { ...updates, updatedAt: new Date() } },
      { returnDocument: "after" }
    );
    return result as unknown as Thread | null;
  }

  async deleteThread(tenantId: string, threadId: string): Promise<boolean> {
    const result = await this.collection.deleteOne({ tenantId, id: threadId });
    return result.deletedCount > 0;
  }

  async hasThread(tenantId: string, threadId: string): Promise<boolean> {
    const count = await this.collection.countDocuments({ tenantId, id: threadId });
    return count > 0;
  }
}
```

## Step 3: Register

```typescript
import { configureStores } from "@axiom-lattice/core";
await configureStores({
  thread: new MongoThreadStore(process.env.MONGO_URL!, "axiom_lattice"),
  // ... other stores
});
```

---

## App-Specific Custom Store (Not Replacing Built-in)

For your own app's data (audit logs, settings, analytics, etc.), register under a custom key:

### Define the store

```typescript
// my-app/stores/AuditLogStore.ts
export class AuditLogStore {
  private db: Database;

  constructor(private connectionString: string) {}

  async initialize(): Promise<void> {
    this.db = await connect(this.connectionString);
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        user_id TEXT,
        action TEXT NOT NULL,
        detail TEXT,
        created_at TEXT NOT NULL
      )
    `);
  }

  async dispose(): Promise<void> { await this.db.close(); }

  async logEvent(event: { tenantId: string; userId?: string; action: string; detail?: string }) {
    await this.db.run(
      "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?)",
      [crypto.randomUUID(), event.tenantId, event.userId, event.action, event.detail, new Date().toISOString()]
    );
  }

  async queryLogs(tenantId: string, limit = 100) {
    return this.db.all("SELECT * FROM audit_log WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?", [tenantId, limit]);
  }
}
```

### Register under a custom key

```typescript
import { configureStores } from "@axiom-lattice/core";

// Custom keys go in the second (options) argument:
await configureStores(
  {},  // no built-in store replacements (use defaults)
  {
    customStores: {
      auditLog: new AuditLogStore(process.env.DATABASE_URL!),
    },
    autoDisposeStores: true,
  }
);
```

### Access from middleware / tools at runtime

```typescript
import { getStoreLattice } from "@axiom-lattice/core";

// Inside a tool executor or middleware:
function getAuditStore(): AuditLogStore {
  return getStoreLattice("default", "auditLog" as any).store as AuditLogStore;
}

// Use it:
await getAuditStore().logEvent({
  tenantId: "my-tenant",
  userId: "user-123",
  action: "agent_completed",
});
```

### Expose via custom API endpoint

```typescript
import { LatticeGateway } from "@axiom-lattice/gateway";
import { getStoreLattice } from "@axiom-lattice/core";

// Add a custom route BEFORE starting the gateway:
LatticeGateway.app.get("/api/custom/audit-logs", async (request, reply) => {
  const tenantId = request.headers["x-tenant-id"] as string;
  const store = getStoreLattice("default", "auditLog" as any).store as AuditLogStore;
  const logs = await store.queryLogs(tenantId);
  return { success: true, data: logs };
});

await LatticeGateway.startAsHttpEndpoint({ port: 4001 });
```

See [custom-controller.md](custom-controller.md) for more on API endpoints.

## Gotchas

- **For built-in stores**: `tenantId` is ALWAYS the first parameter — non-negotiable for multi-tenant isolation
- **For custom stores**: design your own interface — no protocol constraints
- `configureStores(stores, options)` — built-in keys go in first arg, `customStores` in second
- `getStoreLattice("default", "key").store` — retrieve at runtime, cast as needed
- `configureStores({})` returns `Promise<() => Promise<void>>` (dispose function)
- See [migration.md](migration.md) for adding database migrations
- See [custom-controller.md](custom-controller.md) for exposing stores via API
- See [custom-middleware.md](custom-middleware.md) for using stores in middleware
