# Recipe: Agent Lifecycle Events

Subscribe to agent lifecycle events for monitoring, logging, and orchestration.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | Your app code | Subscribe to events |
| 2 | Gateway setup | Configure event handlers |

## Event Types

| Event | Emitted When | Data |
|---|---|---|
| `message:started` | Agent begins processing a message | `{ threadId, messageId, timestamp }` |
| `message:completed` | Agent finishes successfully | `{ threadId, messageId, response, timestamp }` |
| `message:interrupted` | Agent is paused/interrupted | `{ threadId, messageId, state, timestamp }` |
| `message:failed` | Agent execution fails | `{ threadId, messageId, error, timestamp }` |
| `thread:busy` | Thread starts processing | `{ threadId, timestamp }` |
| `thread:idle` | Thread finishes all pending messages | `{ threadId, timestamp }` |
| `queue:pending` | Message queued for processing | `{ threadId, messageId, queueMode }` |
| `reply:ready` | Agent response ready for delivery | `{ threadId, messageId, reply }` |

## Step 1: Subscribe via Agent Instance

Agent events are namespaced as `{eventName}:{tenantId}:{threadId}` internally. The `Agent` class provides convenience methods:

```typescript
// Get an agent instance from agentInstanceManager (for a running thread),
// NOT from agentLatticeManager (which manages configs, not instances).
import { agentInstanceManager } from "@axiom-lattice/core";

const agent = agentInstanceManager.getAgent(threadId);
// or: agentInstanceManager.getAgentByThreadId(assistantId, threadId, tenantId?)

// Subscribe (returns void — store the callback to unsubscribe later)
const handler = (event) => {
  console.log(`Thread ${event.threadId} completed`);
};
agent.subscribe("message:completed", handler);

// One-time subscription
agent.subscribeOnce("message:failed", (event) => {
  console.error(`Failed: ${event.error?.message}`);
});

// Unsubscribe — requires the same callback reference
agent.unsubscribe("message:completed", handler);
```

## Step 2: Subscribe via Global Event Bus

```typescript
import eventBus from "@axiom-lattice/core";  // default export

// Global subscription — fires for ALL agents/threads
eventBus.subscribe("thread:busy", (event) => {
  metrics.increment("active_threads");
});

eventBus.subscribe("thread:idle", (event) => {
  metrics.decrement("active_threads");
});

// Publish custom events
eventBus.publish("custom:hook", { key: "value" });
```

## Step 3: Common Use Cases

### Send Notification on Completion

```typescript
agent.subscribe("message:completed", async (event) => {
  await sendSlackNotification({
    channel: "#agent-logs",
    text: `Agent completed thread ${event.threadId}`,
  });
});
```

### Track Metrics

```typescript
const startTimes = new Map<string, number>();

eventBus.subscribe("message:started", (event) => {
  startTimes.set(event.messageId, Date.now());
});

eventBus.subscribe("message:completed", (event) => {
  const start = startTimes.get(event.messageId);
  if (start) {
    const duration = Date.now() - start;
    metrics.histogram("agent_message_duration_ms", duration, {
      thread_id: event.threadId,
    });
    startTimes.delete(event.messageId);
  }
});

eventBus.subscribe("message:failed", (event) => {
  metrics.increment("agent_message_failures", {
    error: event.error?.name || "unknown",
  });
});
```

### Implement Retry Logic

```typescript
const MAX_RETRIES = 3;
const retries = new Map<string, number>();

eventBus.subscribe("message:failed", async (event) => {
  const count = (retries.get(event.messageId) || 0) + 1;
  retries.set(event.messageId, count);

  if (count <= MAX_RETRIES) {
    console.log(`Retrying message ${event.messageId} (attempt ${count})`);
    // Re-queue the message
    await messageQueue.addMessageAtHead(event.threadId, event.originalMessage);
  } else {
    console.error(`Giving up on message ${event.messageId} after ${MAX_RETRIES} retries`);
    retries.delete(event.messageId);
    // Notify user of permanent failure
  }
});
```

### Chain Agents Together

```typescript
// Events emitted by Agent are auto-namespaced as: {eventName}:{tenantId}:{threadId}
// Use agent-level subscribe for thread-specific chains, eventBus for global.
```

## Event Namespacing

Agent events are automatically namespaced as `{eventName}:{tenantId}:{threadId}` by the Agent class. Use agent-level methods for scoped subscriptions, or subscribe to the full event name on the eventBus:

```typescript
// Via agent (auto-namespaced):
agent.subscribe("message:completed", handler);
// Internally subscribes to: "message:completed:{tenantId}:{threadId}"

// Via eventBus (manual namespace — for listening to a specific agent/thread):
eventBus.subscribe("message:completed:my-tenant:thread-123", handler);
```

## Gotchas

- Events are emitted synchronously during agent execution — don't do heavy work in handlers
- Use async handlers for I/O but be aware they run in parallel with agent execution
- The `eventBus` is a singleton — all agents share it
- `subscribeOnce` auto-unsubscribes after first invocation
- Always unsubscribe when an agent is disposed to prevent memory leaks
- For production, prefer the event bus over agent-level subscriptions for cross-cutting concerns
