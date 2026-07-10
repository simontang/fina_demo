# Deployment Guide

Production deployment checklist and reference configurations.

---

## Architecture Overview

```
                    ┌──────────────┐
                    │   Nginx /    │
                    │   LB         │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
     │  Gateway    │ │ Gateway  │ │  Web App    │
     │  (Fastify)  │ │ (Fastify)│ │  (Next.js)  │
     └──────┬──────┘ └────┬─────┘ └─────────────┘
            │              │
     ┌──────▼──────┐ ┌────▼─────┐
     │ PostgreSQL  │ │  Redis   │
     └─────────────┘ └──────────┘
```

---

## Step 1: Build for Production

```bash
pnpm turbo build
```

---

## Step 2: Configure Environment

```bash
# Required
NODE_ENV=production
JWT_SECRET=<random-64-char-string>
LATTICE_ENCRYPTION_KEY=<random-32-byte-hex>
DATABASE_URL=postgresql://user:pass@host:5432/axiom_lattice

# LLM
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Queue
QUEUE_SERVICE_TYPE=redis
REDIS_URL=redis://redis:6379

# Auth (note: default auth uses base64-encoded tokens, not signed JWT. 
# JWT_SECRET exists as an example env var but is not used for cryptographic signing.)
AUTH_REQUIRED=true
ALLOW_TENANT_REGISTRATION=false
AUTO_APPROVE_USERS=false
```

---

## Step 3: Database Setup

```sql
CREATE DATABASE axiom_lattice;
CREATE USER axiom_user WITH PASSWORD 'strong-password';
GRANT ALL PRIVILEGES ON DATABASE axiom_lattice TO axiom_user;
```

Migrations run automatically when PG stores are initialized via `createPgStoreConfig()` + `configureStores()`.

Check migration status:
```sql
SELECT version, name, applied_at
FROM lattice_schema_migrations
ORDER BY version;
```

---

## Step 4: Docker Compose

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: axiom_lattice
      POSTGRES_USER: axiom_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U axiom_user -d axiom_lattice"]

  redis:
    image: redis:7-alpine

  gateway:
    build:
      context: .
      dockerfile: Dockerfile    # Create your own Dockerfile per §5
    ports:
      - "4001:4001"
    environment:
      NODE_ENV: production
      PORT: "4001"
      DATABASE_URL: postgresql://axiom_user:${DB_PASSWORD}@postgres:5432/axiom_lattice
      REDIS_URL: redis://redis:6379
      QUEUE_SERVICE_TYPE: redis
      JWT_SECRET: ${JWT_SECRET}
      LATTICE_ENCRYPTION_KEY: ${LATTICE_ENCRYPTION_KEY}
      AZURE_OPENAI_API_KEY: ${AZURE_OPENAI_API_KEY}
      AZURE_OPENAI_ENDPOINT: ${AZURE_OPENAI_ENDPOINT}
      AZURE_OPENAI_DEPLOYMENT_NAME: ${AZURE_OPENAI_DEPLOYMENT_NAME}
      AZURE_OPENAI_API_VERSION: ${AZURE_OPENAI_API_VERSION}
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  pgdata:
```

---

## Step 5: Dockerfile

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./
COPY packages/ ./packages/
RUN corepack enable && pnpm install --frozen-lockfile
COPY . .
RUN pnpm turbo build --filter=@axiom-lattice/gateway

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/packages ./packages
EXPOSE 4001
# Note: The gateway does NOT auto-start on import.
# You need an entrypoint script that calls:
#   import { LatticeGateway } from "@axiom-lattice/gateway";
#   await LatticeGateway.startAsHttpEndpoint({ port: 4001 });
CMD ["node", "entrypoint.js"]
```

---

## Step 6: Nginx Reverse Proxy

```nginx
upstream gateway { server 127.0.0.1:4001; }

server {
    listen 443 ssl;

    # SSE streaming: MUST disable buffering
    location /api/ {
        proxy_pass http://gateway;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Health check
    location /health {
        proxy_pass http://gateway;
    }
}
```

---

## Step 7: Production Checklist

- [ ] `NODE_ENV=production`
- [ ] `JWT_SECRET` is strong random string (not default)
- [ ] `LATTICE_ENCRYPTION_KEY` is strong random (not default)
- [ ] `AUTH_REQUIRED=true`
- [ ] PostgreSQL with proper credentials
- [ ] Redis with password if exposed
- [ ] Nginx with HTTPS, `proxy_buffering off` on SSE endpoints
- [ ] Database backups configured
- [ ] Health check monitored: `GET /health` → `{ success: true, data: { status: "healthy" } }`
- [ ] Log aggregation (pino outputs structured JSON)
- [ ] Memory/CPU monitoring

---

## Scaling Notes

- **Gateway instances** scale horizontally behind Nginx (stateless)
- **PostgreSQL** is single source of truth — use PgBouncer for connection pooling
- **Redis** handles message queue — use Sentinel/Cluster for HA
- **Agent execution** is CPU/memory-intensive — each concurrent run may open multiple LLM connections
