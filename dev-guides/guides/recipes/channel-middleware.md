# Recipe: Channel Message Middleware

Add message processing middleware to the channel pipeline.

## Overview

Message middleware runs in the channel pipeline, between receiving an inbound message and routing it to an agent. Use it for message transformation, filtering, enrichment, or logging.

## Step 1: Middleware Signature

File: `packages/protocols/src/ChannelAdapterProtocol.ts`

```typescript
type MessageMiddleware = (
  ctx: MessageContext,
  next: () => Promise<void>
) => Promise<void>;

interface MessageContext {
  inboundMessage: InboundMessage;
  binding?: Binding;
  result?: any;
  error?: Error;
  metadata: Record<string, unknown>;
}
```

## Step 2: Write Middleware

### Example 1: Message Enrichment

```typescript
// middleware/enrich-user-info.ts
import type { MessageMiddleware } from "@axiom-lattice/protocols";

export const enrichUserInfo: MessageMiddleware = async (ctx, next) => {
  // Look up user profile before routing to agent
  try {
    const userProfile = await fetchUserProfile(ctx.inboundMessage.sender.id);
    ctx.inboundMessage.content.metadata = {
      ...ctx.inboundMessage.content.metadata,
      userName: userProfile.name,
      userRole: userProfile.role,
      preferredLanguage: userProfile.language,
    };
  } catch {
    // Gracefully continue without enrichment
  }

  await next();
};
```

### Example 2: Content Filtering

```typescript
// middleware/content-filter.ts
import type { MessageMiddleware } from "@axiom-lattice/protocols";

const BANNED_PATTERNS = [/spam_pattern_1/i, /spam_pattern_2/i];

export const contentFilter: MessageMiddleware = async (ctx, next) => {
  const message = ctx.inboundMessage.content.text;

  const isSpam = BANNED_PATTERNS.some((pattern) => pattern.test(message));
  if (isSpam) {
    // Drop the message — don't call next()
    ctx.result = { filtered: true, reason: "spam" };
    return;
  }

  await next();
};
```

### Example 3: Request Logging

```typescript
// middleware/request-logger.ts
import type { MessageMiddleware } from "@axiom-lattice/protocols";

export const requestLogger: MessageMiddleware = async (ctx, next) => {
  const start = Date.now();
  const messageId = ctx.inboundMessage.content.metadata?.messageId || "unknown";

  console.log(`[INBOUND] ${messageId} from ${ctx.inboundMessage.sender.id}`);

  try {
    await next();
    const duration = Date.now() - start;
    console.log(`[SUCCESS] ${messageId} — ${duration}ms`);
  } catch (error) {
    const duration = Date.now() - start;
    console.error(`[FAILED] ${messageId} — ${duration}ms:`, error);
    throw error;
  }
};
```

### Example 4: Command Prefix Parsing

```typescript
// middleware/command-parser.ts
import type { MessageMiddleware } from "@axiom-lattice/protocols";

export const commandParser: MessageMiddleware = async (ctx, next) => {
  const content = ctx.inboundMessage.content.text;

  const commandMatch = content.match(/^\/(\w+)\s*(.*)?/);
  if (commandMatch) {
    ctx.inboundMessage.content.metadata = {
      ...ctx.inboundMessage.content.metadata,
      command: commandMatch[1],
      commandArgs: commandMatch[2]?.trim(),
    };
  }

  await next();
};
```

## Step 3: Register Middleware in Pipeline

```typescript
// In gateway startup — register middleware on the channel adapter or router
import { MessageRouter } from "@axiom-lattice/gateway";
import { enrichUserInfo, contentFilter, requestLogger, commandParser } from "./middleware";

const router = new MessageRouter({ /* config */ });

// Middleware runs in order
router.use(requestLogger);       // 1. Log every request
router.use(contentFilter);       // 2. Filter spam (may short-circuit)
router.use(commandParser);       // 3. Parse commands
router.use(enrichUserInfo);      // 4. Enrich with user data

// After middleware, the message is routed to the bound agent
```

## Middleware Execution Order

```
InboundMessage received
  │
  ▼
Middleware 1: requestLogger  ──► next()
  │
  ▼
Middleware 2: contentFilter  ──► next()  (or short-circuit)
  │
  ▼
Middleware 3: commandParser  ──► next()
  │
  ▼
Middleware 4: enrichUserInfo ──► next()
  │
  ▼
MessageRouter → BindingRegistry.resolve() → Agent.invoke()
  │
  ▼
ChannelAdapter.sendReply() ← Agent response
```

## Gotchas

- Middleware runs **before** the agent — use it to modify `InboundMessage` before routing
- If middleware does NOT call `next()`, the pipeline stops — use this for filtering/spam detection
- If middleware throws, the error propagates and the message is not processed
- `ctx.metadata` is a shared object — all middleware in the pipeline sees the same metadata
- Middleware runs **per message**, not per channel — keep it lightweight
- For async operations in middleware (API calls), handle errors gracefully to avoid dropping legitimate messages
