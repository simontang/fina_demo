# Recipe: Custom Checkpoint Saver

Replace the conversation checkpoint storage for agent state persistence.

## Overview

Checkpoints store agent conversation state (messages, intermediate steps) for resuming. The framework uses LangGraph's `BaseCheckpointSaver` interface. The default is in-memory (`MemorySaver`), PostgreSQL is the production replacement.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `your-saver/MyCheckpointSaver.ts` | Implement `BaseCheckpointSaver` |
| 2 | Gateway startup | Register via `configureStores()` |

## Step 1: Implement BaseCheckpointSaver

This is a LangGraph interface from `@langchain/langgraph-checkpoint`. Implement the required methods:

```typescript
// your-saver/MongoCheckpointSaver.ts
import { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";
import type { CheckpointMetadata } from "@langchain/langgraph-checkpoint";
import { MongoClient, Collection } from "mongodb";

export class MongoCheckpointSaver extends BaseCheckpointSaver {
  private client: MongoClient;
  private collection: Collection;

  constructor(private connectionString: string) {
    super();
  }

  async initialize(): Promise<void> {
    this.client = new MongoClient(this.connectionString);
    await this.client.connect();
    const db = this.client.db("axiom_lattice");
    this.collection = db.collection("checkpoints");
    await this.collection.createIndex(
      { thread_id: 1, checkpoint_id: 1 },
      { unique: true }
    );
  }

  async dispose(): Promise<void> {
    await this.client.close();
  }

  async getTuple(config: {
    configurable?: { thread_id?: string; checkpoint_id?: string };
  }): Promise<any | undefined> {
    const threadId = config.configurable?.thread_id;
    const checkpointId = config.configurable?.checkpoint_id;

    const filter: any = { thread_id: threadId };
    if (checkpointId) filter.checkpoint_id = checkpointId;

    const doc = checkpointId
      ? await this.collection.findOne(filter)
      : await this.collection.findOne(filter, { sort: { timestamp: -1 } });

    if (!doc) return undefined;

    return {
      config: doc.config,
      checkpoint: doc.checkpoint,
      metadata: doc.metadata,
      parentConfig: doc.parent_config,
      pendingWrites: doc.pending_writes || [],
    };
  }

  async put(
    config: any,
    checkpoint: any,
    metadata: CheckpointMetadata
  ): Promise<any> {
    const doc = {
      thread_id: config.configurable?.thread_id,
      checkpoint_id: checkpoint.id,
      config,
      checkpoint,
      metadata,
      parent_config: metadata.parentConfig || null,
      pending_writes: [],
      timestamp: new Date(),
    };

    await this.collection.updateOne(
      { thread_id: doc.thread_id, checkpoint_id: doc.checkpoint_id },
      { $set: doc },
      { upsert: true }
    );

    return config;
  }

  async putWrites(
    config: any,
    writes: any[],
    taskId: string
  ): Promise<void> {
    await this.collection.updateOne(
      {
        thread_id: config.configurable?.thread_id,
        checkpoint_id: config.configurable?.checkpoint_id,
      },
      {
        $push: { pending_writes: { writes, task_id: taskId } },
      }
    );
  }
}
```

## Step 2: Register via configureStores()

```typescript
// In gateway startup
import { configureStores } from "@axiom-lattice/core";
import { MongoCheckpointSaver } from "./your-saver/MongoCheckpointSaver";

await configureStores({
  checkpoint: new MongoCheckpointSaver(process.env.MONGO_URL!),
  // Other stores can be different backends
  thread: new PostgreSQLThreadStore(dbPool),
});
```

The `checkpoint` key is a special key in `configureStores()` — it gets registered via `MemoryLatticeManager.registerCheckpointSaver()`.

## Existing Implementations (for reference)

| Implementation | File | Notes |
|---|---|---|
| In-memory (default) | `packages/core/src/memory_lattice/DefaultMemorySaver.ts` | Simple, no persistence |
| PostgreSQL | `packages/core/src/util/PGMemory.ts` | Production-ready |

---

## Gotchas

- `getTuple` must return `undefined` (not null) when no checkpoint exists
- `put` must handle upserts — same config may be written multiple times
- `putWrites` is called for pending writes that haven't been applied yet
- Checkpoints can grow large with long conversations — implement TTL/pruning if needed
- Thread safety: concurrent writes to the same thread/checkpoint must be handled (use DB-level locking or upsert)
