FROM node:20-bookworm-slim AS deps

WORKDIR /app/agent

# Install dependencies first (better layer cache).
COPY package.json pnpm-lock.yaml ./

RUN corepack enable \
  && pnpm install --frozen-lockfile


FROM deps AS builder

COPY . .

RUN pnpm build


FROM node:20-bookworm-slim AS runner

WORKDIR /app/agent

ENV NODE_ENV=production

# Runtime needs the built output + node_modules.
COPY --from=deps /app/agent/node_modules ./node_modules
COPY --from=builder /app/agent/dist ./dist
COPY package.json ./

# Default upload location used by the agent is repo-root `uploads/` (resolved from dist).
RUN mkdir -p /app/uploads
RUN mkdir -p /app/lattice_store

# Copy skills directory to lattice_store
COPY --from=builder /app/agent/skills /app/lattice_store/skills



EXPOSE 5702

CMD ["node", "dist/index.js"]

