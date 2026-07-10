# Configuration Reference

Every environment variable in the Axiom Lattice framework, organized by category.

---

## LLM Providers

### Azure OpenAI

| Variable | Required | Default |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Yes (if using Azure) | — |
| `AZURE_OPENAI_ENDPOINT` | Yes (if using Azure) | — |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Yes | `gpt-4` |
| `AZURE_OPENAI_API_VERSION` | Yes | `2024-08-01-preview` |

### OpenAI

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Yes (if using OpenAI) | — |

### DeepSeek

| Variable | Required | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | Yes (if using DeepSeek) | — |

### SiliconCloud

| Variable | Required | Default |
|---|---|---|
| `SILICONCLOUD_API_KEY` | Yes (if using SiliconCloud) | — |

### Volcengine

| Variable | Required | Default |
|---|---|---|
| `VOLCENGINE_API_KEY` | Yes (if using Volcengine) | — |

### Custom LLM Endpoint

*Use `OPENAI_API_KEY` + `LLM_BASE_URL` for OpenAI-compatible APIs (Ollama, vLLM, etc.) — this is an application-level pattern, not a framework env var.*

---

## Database

| Variable | Required | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | Required for PG stores | `postgresql://localhost:5432/axiom_lattice` | PostgreSQL connection string |
| `TEST_DATABASE_URL` | Optional (tests) | — | Separate DB for running tests |

---

## Redis / Queue

| Variable | Required | Default | Notes |
|---|---|---|---|
| `QUEUE_SERVICE_TYPE` | Optional | `memory` | `memory` or `redis` |
| `REDIS_URL` | Required for Redis | `redis://localhost:6379` | Redis connection string |
| `REDIS_PASSWORD` | Optional | — | Redis auth password |
| `QUEUE_NAME` | Optional | `tasks` | Queue name in Redis |

---

## Gateway Server

| Variable | Required | Default | Notes |
|---|---|---|---|
| `PORT` | Optional | `4001` | HTTP server port |
| `BODY_LIMIT` | Optional | `52428800` (50 MB) | Max request body size in bytes |
| `NODE_ENV` | Optional | `development` | `production` enables optimized logging |

---

## Auth & Security

| Variable | Required | Default | Notes |
|---|---|---|---|
| `AUTH_REQUIRED` | Optional | `false` | Set `true` to enable JWT auth |
| `JWT_SECRET` | Required in production | `your-secret-key-change-in-production` | JWT signing secret |
| `AUTO_APPROVE_USERS` | Optional | `true` | Auto-approve new registrations |
| `ALLOW_TENANT_REGISTRATION` | Optional | `true` | Allow self-service tenant creation |
| `LATTICE_ENCRYPTION_KEY` | Required in production | Hardcoded fallback | 32-byte hex key for sensitive data encryption |

---

## Multi-Tenancy

| Variable | Required | Default | Notes |
|---|---|---|---|
| `TENANT_ID` | Optional | `default` | Default tenant for agents |

---

## Channels

### Lark / Feishu

| Variable | Required | Default | Notes |
|---|---|---|---|
| `LARK_ENABLED` | Optional | `true` | Enable Lark channel |
| `LARK_APP_ID` | Required for Lark | — | Lark app ID |
| `LARK_APP_SECRET` | Required for Lark | — | Lark app secret |
| `LARK_VERIFICATION_TOKEN` | Optional | — | Event verification token |
| `LARK_ENCRYPT_KEY` | Optional | — | Message encryption key |
| `LARK_TENANT_ID` | Optional | `default` | Target tenant |
| `LARK_ASSISTANT_ID` | Optional | `default_agent` | Target assistant |
| `LARK_WORKSPACE_ID` | Optional | — | Target workspace |
| `LARK_PROJECT_ID` | Optional | — | Target project |
| `LARK_MAPPING_MODE` | Optional | `hybrid` | `user`, `group`, or `hybrid` |

---

## Sandbox

### Common

| Variable | Required | Default | Notes |
|---|---|---|---|
| `SANDBOX_PROVIDER_TYPE` | Optional | `microsandbox-remote` | `microsandbox-remote`, `remote`, `e2b`, or `daytona` |
| `SANDBOX_BASE_URL` | Optional | — | Generic sandbox base URL |
| `AGENT_INFRA_SANDBOX_BASE_URL` | Optional | `http://localhost:8080` | Browser tool sandbox URL |

### Microsandbox

| Variable | Required | Default | Notes |
|---|---|---|---|
| `MICROSANDBOX_SERVICE_BASE_URL` | Optional | — | Microsandbox service URL |
| `MICROSANDBOX_API_KEY` | Optional | — | API key |
| `MICROSANDBOX_IMAGE` | Optional | `kioko12520/sandbox:0.1.0` | Docker image |
| `MICROSANDBOX_CPUS` | Optional | `1` | CPU allocation |
| `MICROSANDBOX_MEMORY` | Optional | `512` | Memory in MiB |
| `MICROSANDBOX_IDLE_TIMEOUT_SEC` | Optional | `600` | Auto-stop timeout |
| `MICROSANDBOX_VOLUME_QUOTA_MIB` | Optional | `2048` | Per-volume storage limit |

### E2B

| Variable | Required | Default | Notes |
|---|---|---|---|
| `E2B_API_KEY` | Required for E2B | — | E2B API key |
| `E2B_TEMPLATE` | Optional | — | E2B sandbox template |
| `E2B_TIMEOUT_MS` | Optional | — | Sandbox timeout |

### Daytona

| Variable | Required | Default | Notes |
|---|---|---|---|
| `DAYTONA_API_KEY` | Required for Daytona | — | Daytona API key |
| `DAYTONA_API_URL` | Optional | — | API endpoint URL |
| `DAYTONA_TARGET` | Optional | — | Deployment target |
| `DAYTONA_TIMEOUT` | Optional | — | Sandbox timeout |
| `DAYTONA_VOLUME_NAME` | Optional | — | Named volume |

---

## A2A Protocol

| Variable | Required | Default | Notes |
|---|---|---|---|
| `A2A_API_KEYS` | Optional | — | Format: `key1:tenant:project:workspace,key2:...` |
| `A2A_DEFAULT_AGENT_ID` | Optional | — | Default A2A agent |
| `A2A_DEFAULT_TENANT_ID` | Optional | `a2a-default-tenant` | Default tenant |
| `A2A_DEFAULT_PROJECT_ID` | Optional | — | Default project |
| `A2A_DEFAULT_WORKSPACE_ID` | Optional | — | Default workspace |
| `A2A_AGENT_NAME` | Optional | `Axiom Lattice Agent` | Agent card name |
| `A2A_AGENT_DESCRIPTION` | Optional | — | Agent card description |
| `A2A_ORGANIZATION` | Optional | `Axiom Lattice` | Organization name |
| `A2A_ORGANIZATION_URL` | Optional | `https://axiom-lattice.ai` | Organization URL |
| `A2A_VERSION` | Optional | `1.0.0` | Agent card version |

---

## External Services

| Variable | Required | Default | Notes |
|---|---|---|---|
| `TAVILY_API_KEY` | Optional | — | Internet search API key |
| `SUPABASE_URL` | Required if using Supabase | — | Supabase project URL |
| `SUPABASE_KEY` | Required if using Supabase | — | Supabase service key |

---

## .env.example Template

```bash
# LLM Provider (choose one)
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview
# OPENAI_API_KEY=sk-...
# DEEPSEEK_API_KEY=sk-...
# LLM_BASE_URL=https://llm.alphafina.cn/v1

# Database (optional for dev — uses in-memory stores if unset)
DATABASE_URL=postgresql://user:pass@localhost:5432/axiom_lattice
USE_PG_STORES=false

# Redis Queue (optional — uses in-memory queue if unset)
QUEUE_SERVICE_TYPE=memory
# REDIS_URL=redis://localhost:6379
# REDIS_PASSWORD=

# Gateway
PORT=4001
NODE_ENV=development

# Auth
AUTH_REQUIRED=false
JWT_SECRET=change-me-in-production
LATTICE_ENCRYPTION_KEY=change-me-in-production

# Multi-tenancy
TENANT_ID=default

# Sandbox
SANDBOX_PROVIDER_TYPE=microsandbox-remote
MICROSANDBOX_SERVICE_BASE_URL=http://localhost:8080

# Tools
TAVILY_API_KEY=tvly-...
```
