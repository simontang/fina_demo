# Recipe: Custom Schedule Backend

Replace the scheduling system's storage or execution backend.

## Overview

The schedule system has two parts:
1. **ScheduleStorage** — persists task definitions
2. **ScheduleClient** — executes tasks (cron, one-shot, recovery)

You can replace either or both.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `your-schedule/MyScheduleStorage.ts` | Implement `ScheduleStorage` |
| 2 | Gateway startup | Register via `configureStores()` |

## Step 1: Implement ScheduleStorage

File: `packages/protocols/src/ScheduleLatticeProtocol.ts`

```typescript
import type { ScheduleStorage, ScheduledTaskDefinition, ScheduledTaskStatus, ScheduleExecutionType } from "@axiom-lattice/protocols";

export class RedisScheduleStorage implements ScheduleStorage {
  private redis: Redis;

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl);
  }

  async save(task: ScheduledTaskDefinition): Promise<void> {
    await this.redis.set(`schedule:${task.taskId}`, JSON.stringify(task));
  }

  async get(taskId: string): Promise<ScheduledTaskDefinition | null> {
    const data = await this.redis.get(`schedule:${taskId}`);
    return data ? JSON.parse(data) : null;
  }

  async update(taskId: string, updates: Partial<ScheduledTaskDefinition>): Promise<void> {
    const task = await this.get(taskId);
    if (task) await this.save({ ...task, ...updates, updatedAt: Date.now() });
  }

  async delete(taskId: string): Promise<void> {
    await this.redis.del(`schedule:${taskId}`);
  }

  async getActiveTasks(): Promise<ScheduledTaskDefinition[]> {
    // Active = PENDING or PAUSED
    const keys = await this.redis.keys("schedule:*");
    const tasks = await Promise.all(keys.map(k => this.redis.get(k)));
    return tasks
      .filter((t): t is string => t !== null)
      .map(t => JSON.parse(t))
      .filter(t => t.status === "pending" || t.status === "paused");
  }

  async getTasksByType(taskType: string): Promise<ScheduledTaskDefinition[]> {
    const all = await this.getActiveTasks();
    return all.filter(t => t.taskType === taskType);
  }

  async getTasksByStatus(status: ScheduledTaskStatus): Promise<ScheduledTaskDefinition[]> {
    const all = await this.getActiveTasks();
    return all.filter(t => t.status === status);
  }

  async getTasksByExecutionType(type: ScheduleExecutionType): Promise<ScheduledTaskDefinition[]> {
    const all = await this.getActiveTasks();
    return all.filter(t => t.executionType === type);
  }

  async getTasksByAssistantId(assistantId: string): Promise<ScheduledTaskDefinition[]> {
    const all = await this.getActiveTasks();
    return all.filter(t => t.assistantId === assistantId);
  }

  async getTasksByThreadId(threadId: string): Promise<ScheduledTaskDefinition[]> {
    const all = await this.getActiveTasks();
    return all.filter(t => t.threadId === threadId);
  }

  async getAllTasks(filters?: {
    tenantId?: string;
    status?: ScheduledTaskStatus;
    executionType?: ScheduleExecutionType;
    taskType?: string;
    assistantId?: string;
    threadId?: string;
  }): Promise<ScheduledTaskDefinition[]> {
    let tasks = await this.getActiveTasks();
    if (filters?.tenantId) tasks = tasks.filter(t => t.tenantId === filters.tenantId);
    if (filters?.status) tasks = tasks.filter(t => t.status === filters.status);
    if (filters?.taskType) tasks = tasks.filter(t => t.taskType === filters.taskType);
    // ... etc
    return tasks;
  }

  async countTasks(filters?: Record<string, unknown>): Promise<number> {
    const tasks = await this.getAllTasks(filters);
    return tasks.length;
  }

  async deleteOldTasks(olderThanMs: number): Promise<number> {
    const now = Date.now();
    const all = await this.getActiveTasks();
    let count = 0;
    for (const task of all) {
      if (task.createdAt < now - olderThanMs) {
        await this.delete(task.taskId);
        count++;
      }
    }
    return count;
  }
}
```

## Step 2: ScheduledTaskDefinition

```typescript
interface ScheduledTaskDefinition {
  taskId: string;
  taskType: string;          // Maps to registered handler
  payload: Record<string, any>;
  tenantId: string;
  assistantId?: string;
  threadId?: string;
  executionType: "once" | "cron";
  executeAt?: number;        // For ONCE type
  cronExpression?: string;   // For CRON type
  status: "pending" | "running" | "completed" | "failed" | "cancelled" | "paused";
  runCount: number;
  maxRuns?: number;
  maxRetries: number;
  retryCount: number;
  createdAt: number;
  updatedAt: number;
}
```

## Step 3: Register

```typescript
import { configureStores } from "@axiom-lattice/core";
await configureStores({
  schedule: new RedisScheduleStorage(process.env.REDIS_URL!),
});
```

## Built-in Implementations

| Implementation | File |
|---|---|
| Memory | `packages/core/src/schedule_lattice/MemoryScheduleStorage.ts` |
| PostgreSQL | `packages/pg-stores/src/stores/PostgreSQLScheduleStorage.ts` |

## Gotchas

- `ScheduleStorage` methods return `void` (not booleans) for `save`, `update`, `delete`
- Type is `ScheduledTaskDefinition` (not `ScheduledTask`)
- `getTasksByType` (by registered handler type), NOT `getTasksByHandler`
- There is NO `getTasksByTenant` method — use `getAllTasks({ tenantId })` instead
- `scheduleOnce(taskId, taskType, payload, options)` takes 4 args, returns `Promise<boolean>`
- `scheduleCron(taskId, taskType, payload, options)` same pattern, options has `cronExpression`
