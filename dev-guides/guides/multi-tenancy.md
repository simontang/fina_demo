# Multi-Tenancy Guide

Axiom Lattice supports multi-tenant isolation at the agent, store, and sandbox levels.

---

## Overview

The multi-tenancy model has three entities:

```
Tenant
  ├── Workspace(s)
  │     └── Project(s)
  │           └── Agent(s)  ← agents are scoped to a project
  └── User(s) ← via UserTenantLink
```

- **Tenant**: Top-level isolation unit (e.g., a company or team)
- **Workspace**: Group of related projects within a tenant
- **Project**: A specific application or use case
- **User**: Authenticated user, can belong to multiple tenants via links
- **Agent**: Lives in a project within a workspace within a tenant

---

## Configuration

```bash
# .env
AUTH_REQUIRED=true
JWT_SECRET=your-strong-secret
ALLOW_TENANT_REGISTRATION=true   # Allow tenants to self-register
AUTO_APPROVE_USERS=true          # Auto-approve new user registrations
TENANT_ID=default                # Default tenant for system-level operations
```

---

## Tenant Management API

```bash
# Create tenant
curl -X POST http://localhost:4001/api/tenants \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt>" \
  -d '{"name": "my-company", "displayName": "My Company"}'

# List tenants (admin)
curl http://localhost:4001/api/tenants \
  -H "Authorization: Bearer <jwt>"

# Get tenant by ID
curl http://localhost:4001/api/tenants/:tenantId \
  -H "Authorization: Bearer <jwt>"
```

---

## User-Tenant Linking

Users are managed at a flat API namespace, not nested under tenants:

```bash
# Create user (assigns to tenant via body)
curl -X POST http://localhost:4001/api/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt>" \
  -d '{"email": "user@example.com", "tenantId": "my-company"}'

# List users
curl http://localhost:4001/api/users \
  -H "Authorization: Bearer <jwt>"

# Get specific user
curl http://localhost:4001/api/users/:userId \
  -H "Authorization: Bearer <jwt>"
```

---

## Tenant Isolation Levels

### Store-Level Isolation

All store methods receive `tenantId` as their FIRST positional parameter (not as an options object):

```typescript
// tenantId is always the first argument:
await threadStore.getThreadsByAssistantId("my-company", assistantId);
await threadStore.getThreadById("my-company", threadId);
await threadStore.createThread("my-company", assistantId, threadId, { metadata: {} });
```

The `configureStores()` system registers stores under the `"default"` lattice key:

### Agent-Level Isolation

Agents are configured with tenant/project/workspace context:

```typescript
const agentConfig = {
  type: "REACT",
  name: "support-bot",
  tenantId: "my-company",
  workspaceId: "support-workspace",
  projectId: "helpdesk",
  // ...
};
```

### Sandbox Isolation

Sandbox isolation uses `vmIsolation` levels:

```typescript
// Levels:
// "global"   — one sandbox shared across all tenants
// "agent"    — one sandbox per agent
// "project"  — one sandbox per project (recommended for multi-tenant)
```

Set in agent config:
```typescript
const agentConfig = {
  // ...
  vmIsolation: "project",  // Each project gets its own sandbox
};
```

### Filesystem Isolation

When using filesystem tools, working directories are scoped by tenant:

```typescript
// The framework auto-scopes paths:
// /workspace/{tenantId}/{workspaceId}/{projectId}/
```

---

## Middleware with Tenant Context

Custom middleware receives tenant context via `runConfig`:

```typescript
// In your middleware:
async function myMiddleware(state, next) {
  const tenantId = state.runConfig?.tenantId;
  // Use tenantId for scoped operations
  return next(state);
}
```

---

## Auth Flow

```
1. User registers  →  POST /api/auth/register
2. Admin approves  →  (or AUTO_APPROVE_USERS=true)
3. User logs in    →  POST /api/auth/login  →  receives JWT
4. JWT contains    →  { userId, tenantId, ... }
5. All API calls   →  Authorization: Bearer <jwt>
6. Gateway extracts → tenantId from JWT for store queries
```

---

## Implementing Custom Multi-Tenant Logic

### Custom Tenant-Aware Store

```typescript
class MyTenantAwareStore implements ThreadStore {
  constructor(private db: Database) {}

  async getThreadsByAssistantId(
    assistantId: string,
    context?: { tenantId?: string }
  ): Promise<Thread[]> {
    const tenantId = context?.tenantId ?? "default";
    return this.db.query(
      "SELECT * FROM threads WHERE assistant_id = $1 AND tenant_id = $2",
      [assistantId, tenantId]
    );
  }
}
```

### Custom Tenant Middleware

```typescript
// gateway/src/middleware/tenant-context.ts
export async function tenantContextMiddleware(request, reply) {
  const token = request.headers.authorization?.replace("Bearer ", "");
  if (token) {
    // Auth system uses base64-encoded JSON tokens (not signed JWT)
    const decoded = JSON.parse(atob(token));
    request.tenantId = decoded.tenantId;
    request.userId = decoded.userId;
  }
}
```

---

## Store Reference

All stores that support tenant isolation:

| Store | Tenant Field |
|---|---|
| ThreadStore | `context.tenantId` |
| AssistantStore | Implicit via workspace/project |
| UserStore | Global (users span tenants via links) |
| TenantStore | N/A (manages tenants themselves) |
| UserTenantLinkStore | Manages user-tenant mappings |
| WorkspaceStore | Scoped to tenant |
| ProjectStore | Scoped to workspace |
| WorkflowTrackingStore | `tenantId` parameter |
| ChannelInstallationStore | `tenantId` parameter |
| BindingRegistry | Scoped to tenant/workspace/project |

---

## Gotchas

1. **Default tenant** — if no `tenantId` is provided, the system defaults to `"default"`. Make sure to explicitly set tenants in production.
2. **UserTenantLink is required** — a user cannot access a tenant's resources without a link record.
3. **Sandbox isolation levels matter** — using `"global"` in multi-tenant means tenants share sandbox state. Use `"project"` for proper isolation.
4. **Encryption keys** — `LATTICE_ENCRYPTION_KEY` encrypts sensitive data. If you change it, existing encrypted data cannot be decrypted.
