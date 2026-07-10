# Recipe: Database Migrations

Create and run database migrations for custom stores or extending existing PG stores.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `your-store/migrations.ts` | Define migration |
| 2 | Store class | Register and run migrations |

## Migration System Overview

The migration manager at `packages/pg-stores/src/migrations/migration.ts`:

- Uses PostgreSQL **advisory locks** for concurrent safety
- Tracks migrations in the `lattice_schema_migrations` table
- Runs migrations in **version order**
- Supports `up()` (forward) and optional `down()` (rollback)

### Version Numbering Scheme

| Range | Purpose |
|---|---|
| 1-99 | Core table changes |
| 100+ | Feature modules |

---

## Step 1: Define a Migration

```typescript
// your-store/migrations.ts
import type { Migration } from "@axiom-lattice/pg-stores";
import type { PoolClient } from "pg";

export const myStoreV1: Migration = {
  version: 200,
  name: "create_my_custom_table",
  up: async (client: PoolClient) => {
    await client.query(`
      CREATE TABLE IF NOT EXISTS my_custom_data (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )
    `);
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_my_data_tenant ON my_custom_data(tenant_id)
    `);
  },
  down: async (client: PoolClient) => {
    await client.query("DROP TABLE IF EXISTS my_custom_data");
  },
};
```

## Step 2: Register and Run Migrations

```typescript
// your-store/MyCustomPgStore.ts
import { Pool } from "pg";
import { MigrationManager } from "@axiom-lattice/pg-stores";
import type { Migration } from "@axiom-lattice/pg-stores";

export class MyCustomStore {
  private pool: Pool;

  constructor(connectionString: string) {
    this.pool = new Pool({ connectionString });
  }

  async initialize(): Promise<void> {
    // MigrationManager constructor requires a Pool
    const mgr = new MigrationManager(this.pool);
    mgr.register(myStoreV1);

    // migrate() takes no arguments — uses this.pool from constructor
    await mgr.migrate();
  }

  async dispose(): Promise<void> {
    await this.pool.end();
  }
}
```

## Step 3: Standalone Migration Script

```typescript
// scripts/migrate.ts
import { Pool } from "pg";
import { MigrationManager } from "@axiom-lattice/pg-stores";
import { myStoreV1 } from "./migrations";

async function run() {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL! });
  const mgr = new MigrationManager(pool);
  mgr.register(myStoreV1);
  await mgr.migrate();
  console.log("Done");
  await pool.end();
}

run().catch(console.error);
```

## Migration Best Practices

### Idempotent Migrations

```typescript
// GOOD
await client.query("CREATE TABLE IF NOT EXISTS ...");
await client.query("ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...");

// BAD — fails on re-run
await client.query("CREATE TABLE ...");
```

### One Migration Per Change

```typescript
{ version: 200, name: "create_users_table", ... }
{ version: 201, name: "add_email_index", ... }
{ version: 202, name: "add_last_login", ... }
```

## Checking Migration Status

```sql
-- The migration table is lattice_schema_migrations
SELECT version, name, applied_at
FROM lattice_schema_migrations
ORDER BY version;
```

## Gotchas

- **MigrationManager constructor**: `new MigrationManager(pool)` — pool is required
- **migrate()**: takes no arguments — uses `this.pool`
- **Table name**: `lattice_schema_migrations` (column: `applied_at`, not `executed_at`)
- **Advisory locks**: prevent concurrent migration runs
- **Never change an executed migration** — create a new one with higher version
- **`down()` is optional** — omit if rollback never needed
