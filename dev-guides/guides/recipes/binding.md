# Recipe: Managing Channel Bindings

Bindings map external channel senders to specific agents.

## Overview

A binding tells the system: "when sender X on channel Y sends a message, route it to agent Z."

## API Endpoints (all under `/api/channel-bindings`)

```bash
# Create binding
curl -X POST http://localhost:4001/api/channel-bindings \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "slack",
    "channelInstallationId": "inst_abc",
    "senderId": "U12345",
    "agentId": "support-bot",
    "tenantId": "my-company",
    "threadMode": "per_conversation"
  }'

# List bindings for a tenant
curl "http://localhost:4001/api/channel-bindings?tenantId=my-company"

# Get specific binding
curl http://localhost:4001/api/channel-bindings/:id

# Update binding
curl -X PUT http://localhost:4001/api/channel-bindings/:id \
  -H "Content-Type: application/json" \
  -d '{"agentId": "new-agent"}'

# Delete binding
curl -X DELETE http://localhost:4001/api/channel-bindings/:id
```

## Programmatic Binding Management

```typescript
import type { BindingRegistry, CreateBindingInput } from "@axiom-lattice/protocols";

// Resolve binding for a sender
const binding = await bindings.resolve({
  channel: "slack",
  senderId: "U12345",
  channelInstallationId: "inst_abc",  // required
  tenantId: "my-company",
});

// Create binding
await bindings.create({
  channel: "slack",
  channelInstallationId: "inst_abc",
  tenantId: "my-company",
  senderId: "U12345",
  agentId: "support-bot",
  threadMode: "per_conversation",  // "fixed" | "per_conversation"
  workspaceId: "ws-1",
  projectId: "proj-1",
  senderDisplayName: "Alice",
});

// List bindings
const all = await bindings.list({ tenantId: "my-company" });

// Update binding
await bindings.update("binding-id", { agentId: "new-agent" });

// Delete binding
await bindings.delete("binding-id");

// Import/export
const exported = await bindings.export({ tenantId: "my-company" });
await bindings.import(exported);
```

## Binding Interface

```typescript
interface Binding {
  id: string;
  channel: string;
  channelInstallationId: string;  // required
  tenantId: string;
  senderId: string;
  agentId: string;
  threadId?: string;
  workspaceId?: string;
  projectId?: string;
  threadMode: "fixed" | "per_conversation";
  senderDisplayName?: string;
  senderMetadata?: Record<string, unknown>;
  enabled: boolean;
  createdAt: Date;
  updatedAt: Date;
}
```

## Gotchas

- All API paths use `/api/channel-bindings` (NOT `/api/bindings`)
- `resolve()` requires `channelInstallationId` — not optional
- `create()` requires `channel`, `channelInstallationId`, `tenantId`, `senderId`, `agentId`
- `threadMode` is `"fixed"` (always same thread) or `"per_conversation"` (new thread per conversation)
- Bindings can be created/updated/deleted at runtime — no restart needed
