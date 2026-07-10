# Recipe: Custom Queue Backend

Replace the message queue with RabbitMQ, SQS, Kafka, or any other backend.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `your-queue/MyQueueClient.ts` | Implement `QueueClient` |
| 2 | Gateway startup | Register queue via `QueueLatticeManager` |

## Step 1: Implement QueueClient

File: `packages/protocols/src/QueueLatticeProtocol.ts`

```typescript
export interface QueueClient {
  push: (item: any) => Promise<QueueResult<number>>;
  pop: () => Promise<QueueResult<any>>;
  createQueue?: () => Promise<{ success: boolean; queue_name?: string; error?: any }>;
}

export interface QueueResult<T = any> {
  data: T | null;
  error: any | null;
}
```

Example — RabbitMQ:

```typescript
import type { QueueClient, QueueResult } from "@axiom-lattice/protocols";
import amqp from "amqplib";

export class RabbitMQQueueClient implements QueueClient {
  private connection: amqp.Connection | null = null;
  private channel: amqp.Channel | null = null;

  constructor(private queueName: string, private url: string) {}

  async createQueue(): Promise<{ success: boolean; queue_name?: string; error?: any }> {
    try {
      this.connection = await amqp.connect(this.url);
      this.channel = await this.connection.createChannel();
      await this.channel!.assertQueue(this.queueName, { durable: true });
      return { success: true, queue_name: this.queueName };
    } catch (error) {
      return { success: false, error };
    }
  }

  async push(item: any): Promise<QueueResult<number>> {
    try {
      if (!this.channel) await this.createQueue();
      const buffer = Buffer.from(JSON.stringify(item));
      this.channel!.sendToQueue(this.queueName, buffer, { persistent: true });
      return { data: 1, error: null };
    } catch (error) {
      return { data: null, error };
    }
  }

  async pop(): Promise<QueueResult<any>> {
    try {
      if (!this.channel) await this.createQueue();
      const msg = await this.channel!.get(this.queueName, { noAck: false });
      if (!msg) return { data: null, error: null };
      const item = JSON.parse(msg.content.toString());
      this.channel!.ack(msg);
      return { data: item, error: null };
    } catch (error) {
      return { data: null, error };
    }
  }
}
```

## Step 2: Register Queue Client

`QueueLatticeManager.registerLattice()` takes 3 arguments: `(key, config, client?)`:

```typescript
import { queueLatticeManager } from "@axiom-lattice/core";
import { QueueType } from "@axiom-lattice/protocols";
import { RabbitMQQueueClient } from "./your-queue/RabbitMQQueueClient";

const rabbitClient = new RabbitMQQueueClient("agent_tasks", "amqp://localhost");

// registerLattice(key, QueueConfig, QueueClient?)
queueLatticeManager.registerLattice(
  "default",                    // key
  {
    name: "rabbitmq",
    description: "RabbitMQ task queue",
    type: QueueType.REDIS,      // or use custom type
    queueName: "agent_tasks",
  },
  rabbitClient                  // QueueClient — separate argument
);
```

## Step 3: Configure Gateway

```bash
# .env
QUEUE_SERVICE_TYPE=redis   # Tells gateway which queue service to use
REDIS_URL=amqp://localhost # or in this case, your custom URL
```

## Alternative Backends

| Backend | Library | Notes |
|---|---|---|
| **RabbitMQ** | `amqplib` | Handle connection recovery |
| **AWS SQS** | `@aws-sdk/client-sqs` | `pop()` = `ReceiveMessage` + `DeleteMessage` |
| **Google Pub/Sub** | `@google-cloud/pubsub` | `pop()` = `pull()` |
| **Kafka** | `kafkajs` | Consumer-based pop pattern |

## Gotchas

- `registerLattice(key, config, client)` — config and client are SEPARATE arguments, NOT `{ item: client }`
- `push()` returns `{ data: queueDepth, error: null }`
- `pop()` returns `{ data: null, error: null }` for empty queue (not an error)
- `createQueue()` is optional — only implement if backend needs explicit setup
- Handle connection failures gracefully with reconnection logic
