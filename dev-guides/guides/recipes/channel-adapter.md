# Recipe: Building a Channel Adapter

Add a new messaging channel (Slack, Teams, WhatsApp, Discord, etc.).

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `your-channel/MyChannelAdapter.ts` | Implement `ChannelAdapter` |
| 2 | Gateway setup | Register adapter via `ChannelAdapterRegistry` |
| 3 | Webhook endpoint | Configure channel's webhook URL |

## Step 1: Understand the Protocol

File: `packages/protocols/src/ChannelAdapterProtocol.ts`

```typescript
export interface ChannelAdapter<TConfig = unknown> {
  readonly channel: string;
  readonly configSchema: z.ZodSchema<TConfig>;
  receive(rawPayload: unknown, installation: ChannelInstallation): Promise<InboundMessage | null>;
  sendReply(replyTarget: ReplyTarget, message: OutboundMessage, installation: ChannelInstallation): Promise<void>;
}

// Real InboundMessage shape:
interface InboundMessage {
  channel: string;
  channelInstallationId: string;
  tenantId?: string;
  sender: { id: string; displayName?: string };
  content: {
    text: string;
    attachments?: Attachment[];
    metadata?: Record<string, unknown>;
  };
  conversation?: { id: string; type: "direct" | "group" };
  replyTarget?: ReplyTarget;
}

// Real OutboundMessage shape:
interface OutboundMessage {
  text: string;
  attachments?: Attachment[];
  metadata?: Record<string, unknown>;
}

// ReplyTarget — set this in InboundMessage.replyTarget for auto-reply:
interface ReplyTarget {
  adapterChannel: string;
  channelInstallationId: string;
  rawTarget: Record<string, unknown>;
}
```

## Step 2: Implement ChannelAdapter

Example — Slack channel adapter:

```typescript
import { z } from "zod";
import type {
  ChannelAdapter, InboundMessage, OutboundMessage,
  ChannelInstallation, ReplyTarget,
} from "@axiom-lattice/protocols";
import { WebClient } from "@slack/web-api";

interface SlackConfig {
  botToken: string;
  signingSecret: string;
}

const slackConfigSchema = z.object({
  botToken: z.string().min(1),
  signingSecret: z.string().min(1),
});

export const slackAdapter: ChannelAdapter<SlackConfig> = {
  channel: "slack",

  configSchema: slackConfigSchema,

  async receive(rawPayload: unknown, installation: ChannelInstallation): Promise<InboundMessage | null> {
    const payload = rawPayload as any;

    // Handle Slack URL verification
    if (payload.type === "url_verification") return null;

    // Handle events
    if (payload.type === "event_callback") {
      const ev = payload.event;
      if (ev.type === "message" && !ev.bot_id) {
        return {
          channel: "slack",
          channelInstallationId: installation.id,
          tenantId: installation.tenantId,
          sender: { id: ev.user, displayName: ev.user },
          content: { text: ev.text },
          conversation: { id: ev.channel, type: ev.channel_type === "im" ? "direct" : "group" },
          replyTarget: {
            adapterChannel: "slack",
            channelInstallationId: installation.id,
            rawTarget: { channel: ev.channel, threadTs: ev.thread_ts },
          },
        };
      }
    }

    return null;
  },

  async sendReply(
    replyTarget: ReplyTarget,
    message: OutboundMessage,
    installation: ChannelInstallation,
  ): Promise<void> {
    const config = installation.config as SlackConfig;
    const client = new WebClient(config.botToken);
    const target = replyTarget.rawTarget as { channel: string; threadTs?: string };

    await client.chat.postMessage({
      channel: target.channel,
      thread_ts: target.threadTs,
      text: message.text,
    });
  },
};
```

## Step 3: Register Adapter

```typescript
import { ChannelAdapterRegistry } from "@axiom-lattice/gateway";
import { slackAdapter } from "./your-channel/SlackChannelAdapter";

// ChannelAdapterRegistry is instantiated with `new`, NOT a singleton
const registry = new ChannelAdapterRegistry();
registry.register(slackAdapter);
```

In the gateway, the registry is created inside `start()` and passes to route registration automatically. Externally, create and register before gateway starts.

## Step 4: Set Up Webhook Endpoint

Configure your channel platform to send webhooks to:

```
POST https://your-domain.com/api/channels/inbound
```

Webhook payload format:

```json
{
  "channel": "slack",
  "installation_id": "inst_abc123",
  "payload": { /* raw payload from the platform */ }
}
```

## Step 5: Configure Channel Installation

```bash
curl -X POST http://localhost:4001/api/channel-installations \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: my-tenant" \
  -d '{
    "channel": "slack",
    "config": {
      "botToken": "xoxb-...",
      "signingSecret": "abc123..."
    }
  }'
```

## Step 6: Configure Binding (Sender → Agent)

```bash
curl -X POST http://localhost:4001/api/channel-bindings \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "slack",
    "sender_id": "U12345",
    "assistant_id": "support-bot",
    "tenant_id": "my-tenant"
  }'
```

## Message Flow

```
External Platform
  │ POST webhook
  ▼
/api/channels/inbound
  │ ChannelAdapterRegistry.get("slack")
  ▼
ChannelAdapter.receive(rawPayload)
  │ Normalizes to InboundMessage (includes replyTarget)
  ▼
MessageRouter.dispatch(inboundMessage)
  │ BindingRegistry.resolve(senderId)
  ▼
Agent processes message
  │ Emits 'reply:ready' event
  ▼
ChannelAdapter.sendReply(replyTarget, outboundMessage, installation)
  │ Posts back to channel
  ▼
External Platform
```

## Gotchas

- `receive()` returns `null` for events that shouldn't trigger processing (e.g., health checks)
- Set `replyTarget` in the returned `InboundMessage` for automatic reply handling
- The `OutboundMessage` has `text` (string), NOT `content` — use `message.text`
- The `InboundMessage.content` has `text` field, NOT `contentType` — content is always text with optional attachments
- `ChannelAdapterRegistry` uses `new ChannelAdapterRegistry()`, NOT `.getInstance()` — it's not a singleton
- Use `rawTarget` in `ReplyTarget` to pass channel-specific data needed for sending the reply
- See [channel-middleware.md](channel-middleware.md) for adding message processing middleware
- See [binding.md](binding.md) for managing sender-to-agent bindings
