# Troubleshooting Guide

Common errors, their causes, and solutions.

---

## Build & Setup

### `Cannot find module '@axiom-lattice/...'`

**Cause**: Packages not built.

```bash
pnpm turbo build
```

---

### `TypeScript error: Cannot find module '...' or its corresponding type declarations`

**Cause**: TypeScript project references out of sync.

```bash
pnpm turbo build --force
```

---

### `pnpm: command not found`

```bash
npm install -g pnpm
corepack enable  # If using Node.js 16.9+
```

---

## LLM / Model

### `Error: API key not found for provider 'azure'`

**Cause**: Missing `AZURE_OPENAI_API_KEY` environment variable.

```bash
# Set in .env or export directly
export AZURE_OPENAI_API_KEY=sk-...
```

---

### `Error: Connection refused` when calling LLM

**Cause**: Wrong endpoint or network issue.

- Verify `AZURE_OPENAI_ENDPOINT` is correct
- Check that the deployment exists in your Azure portal
- Verify network/firewall allows outbound HTTPS

---

### `Error: Model not found` or `Deployment not found`

**Cause**: `AZURE_OPENAI_DEPLOYMENT_NAME` doesn't match Azure deployment.

Check your Azure OpenAI Studio → Deployments for the exact deployment name.

---

### Agent keeps looping without making progress

**Cause**: Agent may be stuck in a tool-calling loop.

```typescript
// Check agent middleware and tool configurations.
// For REACT agents, ensure tools return clear results.
// For WORKFLOW agents, ensure the DSL has proper end conditions.
// No built-in maxIterations or earlyStop config fields exist at the agent level.
```

---

## Database

### `Error: connect ECONNREFUSED 127.0.0.1:5432`

**Cause**: PostgreSQL not running.

```bash
# macOS
brew services start postgresql

# Linux
sudo systemctl start postgresql

# Docker
docker run -d -p 5432:5432 -e POSTGRES_DB=axiom_lattice postgres:16
```

---

### `Error: relation "threads" does not exist`

**Cause**: Migrations haven't run.

Migrations run automatically when PG stores are initialized via `createPgStoreConfig()` + `configureStores()`. If they didn't run, check that `DATABASE_URL` is configured correctly and PG stores are being used (not in-memory defaults).

Check migration status:
```bash
psql $DATABASE_URL -c "SELECT version, name, applied_at FROM lattice_schema_migrations ORDER BY version;"
```

---

### `Error: duplicate key value violates unique constraint`

**Cause**: Data already exists with the same ID.

This is usually safe to ignore — it means a concurrent request already created the record. If it's unexpected, check your ID generation logic.

---

### `FATAL: sorry, too many clients already`

**Cause**: Too many PostgreSQL connections.

Solutions:
1. Use PgBouncer as connection pooler
2. Set `max_connections` higher in `postgresql.conf`
3. Reduce concurrent agent runs

---

## Redis / Queue

### `Error: connect ECONNREFUSED 127.0.0.1:6379`

**Cause**: Redis not running.

```bash
# macOS
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:7-alpine
```

---

### Messages not being processed

**Cause**: Queue service type mismatch or Redis not connected.

Check:
1. `QUEUE_SERVICE_TYPE` is set to `redis` (not `memory`)
2. `REDIS_URL` is correct
3. Redis is reachable

---

## SSE / Streaming

### SSE connection drops immediately

**Cause**: Proxy buffering enabled.

If behind Nginx, ensure:
```nginx
location /api/ {
    proxy_buffering off;
    proxy_cache off;
}
```

---

### Frontend stops receiving messages but agent is still running

**Cause**: SSE connection timeout or network issue.

The agent continues executing even if the SSE connection drops. Reconnect by sending a new request to the same thread. To check status: `GET /api/assistants/:id/threads/:threadId`.

To abort a running agent:
```bash
POST /api/assistants/:id/threads/:threadId/abort
```

---

## Auth

### `401 Unauthorized`

**Cause**: Missing or invalid JWT.

```bash
# 1. Register
curl -X POST http://localhost:4001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# 2. Login to get token
curl -X POST http://localhost:4001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# 3. Use token
curl http://localhost:4001/api/... \
  -H "Authorization: Bearer <token-from-login>"
```

---

### `403 Forbidden`

**Cause**: User not approved or not linked to tenant.

Check:
1. `AUTO_APPROVE_USERS=true` in dev
2. User has a `UserTenantLink` record for the target tenant
3. Admin has approved the user (if `AUTO_APPROVE_USERS=false`)

---

## Sandbox

### `Error: Sandbox provider not configured`

**Cause**: Missing sandbox env vars.

Set `SANDBOX_PROVIDER_TYPE` and the corresponding provider credentials:
- `microsandbox-remote` → `MICROSANDBOX_SERVICE_BASE_URL`, `MICROSANDBOX_API_KEY`
- `e2b` → `E2B_API_KEY`
- `daytona` → `DAYTONA_API_KEY`, `DAYTONA_API_URL`

---

### Sandbox times out

**Cause**: Sandbox idle timeout too short.

Increase the appropriate timeout for your provider:
```bash
# microsandbox
MICROSANDBOX_IDLE_TIMEOUT_SEC=1800   # 30 minutes

# Check provider-specific timeout options
```

---

## Performance

### High memory usage

**Cause**: Too many concurrent agent runs.

Solutions:
1. Limit concurrent agent runs via gateway config
2. Use Redis queue to serialize execution
3. Increase Node.js heap: `NODE_OPTIONS="--max-old-space-size=4096"`

---

### Slow agent responses

**Cause**: LLM latency or too many tool calls.

Check:
1. Tool execution time — optimize slow tools
2. Use streaming (`"streaming": true` on `POST /api/runs`) for progressive UX
3. Check LLM provider latency separately

---

## Debugging

### Enable verbose logging

```bash
# Enable Node inspector
node --inspect packages/gateway/dist/index.js
```

### Inspect agent state

```typescript
// Subscribe to agent events
agent.subscribe("message:completed", (event) => {
  console.log("Agent completed:", JSON.stringify(event, null, 2));
});

agent.subscribe("message:failed", (event) => {
  console.error("Agent failed:", event.error);
});
```

### Check database directly

```sql
-- List recent threads
SELECT id, assistant_id, title, created_at
FROM threads
ORDER BY created_at DESC
LIMIT 10;

-- Check queue
SELECT thread_id, status, message_count
FROM thread_message_queues
WHERE status = 'pending';
```
