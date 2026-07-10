# Quickstart Guide

Get Axiom Lattice running in 5 minutes.

## Prerequisites

- Node.js >= 18
- pnpm >= 10 (`npm i -g pnpm`)
- PostgreSQL (optional — defaults to in-memory stores for development)

## Step 1: Install & Build

```bash
git clone <repo-url> my-agent-app
cd my-agent-app
pnpm install
pnpm turbo build
```

## Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys. Minimal required:

```bash
# LLM Provider (pick one — Azure OpenAI example)
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Or use standard OpenAI
# OPENAI_API_KEY=sk-...

# Database (optional — in-memory stores used if not set)
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
```

See [configuration.md](configuration.md) for the complete env var reference.

## Step 3: Start the Gateway

```bash
pnpm --filter @axiom-lattice/gateway dev
```

By default, the gateway starts on **port 4001** with:
- REST API: `http://localhost:4001/api/`
- SSE streaming: Set `"streaming": true` in request body
- Health check: `GET http://localhost:4001/health`

## Step 4: Start the Web UI (optional)

```bash
pnpm --filter web dev
```

Opens on `http://localhost:4000`.

## Step 5: Send Your First Message

```bash
# Non-streaming run
curl -X POST http://localhost:4001/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "default",
    "message": "Hello! What can you do?"
  }'
```

For streaming responses, add `"streaming": true`:

```bash
curl -N -X POST http://localhost:4001/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "default",
    "message": "Write a haiku about programming",
    "streaming": true
  }'
```

## Step 6: Build Your Own Agent

See [first-agent.md](first-agent.md) for a step-by-step tutorial to build your first custom agent.

## Development Mode

All packages watch for changes:
```bash
# In separate terminals:
pnpm --filter @axiom-lattice/core dev --watch
pnpm --filter @axiom-lattice/gateway dev
pnpm --filter web dev
```

## Common Issues

| Problem | Solution |
|---|---|
| `Cannot find module` | Run `pnpm turbo build` to build all packages |
| LLM errors | Check API key in `.env`, verify endpoint URL |
| PostgreSQL errors | Set `DATABASE_URL` or omit for in-memory stores |
| Port in use | Set `PORT=4001` for gateway or `PORT=4000` for web |

## Next Steps

- [Build your first agent](first-agent.md)
- [Add a custom tool](recipes/creating-tool.md)
- [Understand the architecture](../ARCHITECTURE.md)
