# This Dockerfile uses nginx to serve the static files.
FROM refinedev/node:18 AS base

FROM base as deps

COPY package.json yarn.lock* package-lock.json* pnpm-lock.yaml* .npmrc* ./

RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then yarn global add pnpm && pnpm i --frozen-lockfile; \
  else echo "Lockfile not found." && exit 1; \
  fi

FROM base as builder

ENV NODE_ENV production
# Set API URL for production build (can be overridden via build arg)
# Use relative path /api so it works with nginx reverse proxy
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}

COPY --from=deps /app/refine/node_modules ./node_modules

COPY . .

RUN npm run build

FROM nginx:alpine as runner

# Copy custom nginx config for SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files to nginx serve directory
COPY --from=builder /app/refine/dist /usr/share/nginx/html

EXPOSE 3201

CMD ["nginx", "-g", "daemon off;"]
