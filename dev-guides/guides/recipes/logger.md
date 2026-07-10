# Recipe: Custom Logger

Implement a custom logger (e.g., structured logging to Datadog, Splunk, CloudWatch).

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `your-logger/MyLoggerClient.ts` | Implement `LoggerClient` |
| 2 | Gateway startup | Register logger |

## Step 1: Real LoggerClient Interface

File: `packages/protocols/src/LoggerLatticeProtocol.ts`

```typescript
export interface LoggerClient {
  info: (msg: string, obj?: object) => void;
  error: (msg: string, obj?: object | Error) => void;
  warn: (msg: string, obj?: object) => void;
  debug: (msg: string, obj?: object) => void;
  updateContext?: (context: Partial<LoggerContext>) => void;
  child?: (options: Partial<LoggerConfig>) => LoggerClient;
}
```

## Step 2: Implement LoggerClient

```typescript
import type { LoggerClient, LoggerConfig, LoggerContext } from "@axiom-lattice/protocols";

export class DatadogLogger implements LoggerClient {
  private context: Partial<LoggerContext> = {};

  constructor(private config: Partial<LoggerConfig>) {}

  info(msg: string, obj?: object): void {
    console.log(JSON.stringify({ level: "info", msg, ...obj }));
  }

  error(msg: string, obj?: object | Error): void {
    const errObj = obj instanceof Error ? { error: obj.message, stack: obj.stack } : obj;
    console.error(JSON.stringify({ level: "error", msg, ...errObj }));
  }

  warn(msg: string, obj?: object): void {
    console.warn(JSON.stringify({ level: "warn", msg, ...obj }));
  }

  debug(msg: string, obj?: object): void {
    console.debug(JSON.stringify({ level: "debug", msg, ...obj }));
  }

  updateContext(context: Partial<LoggerContext>): void {
    this.context = { ...this.context, ...context };
  }

  child(options: Partial<LoggerConfig>): LoggerClient {
    return new DatadogLogger({ ...this.config, ...options });
  }
}
```

## Step 3: Register Logger

`registerLattice(key, config, client)` — client is a SEPARATE third argument:

```typescript
import { loggerLatticeManager } from "@axiom-lattice/core";
import { LoggerType } from "@axiom-lattice/protocols";

const logger = new DatadogLogger({ serviceName: "axiom-gateway" });

loggerLatticeManager.registerLattice(
  "default",                    // key
  {                             // LoggerConfig
    name: "datadog",
    description: "Datadog logger",
    type: LoggerType.PINO,      // or custom type
    serviceName: "axiom/gateway",
  },
  logger                        // LoggerClient — separate 3rd argument
);
```

## Built-in Implementations

| File | Notes |
|---|---|
| `packages/core/src/logger_lattice/PinoLoggerClient.ts` | Default. Pino-based structured JSON. |
| `packages/core/src/logger_lattice/ConsoleLoggerClient.ts` | Simple console-based for development. |

## Gotchas

- `registerLattice(key, config, client)` — client is 3rd arg, NOT inside config as `item`
- `info(msg, obj?)` — second param is `object`, not `Record<string, unknown>`
- `error(msg, obj?)` — second param can be `Error`
- `updateContext(context)` — takes `Partial<LoggerContext>`, not `Record<string, unknown>`
- `child(options)` — takes `Partial<LoggerConfig>`, not `Record<string, unknown>`
- The gateway initializes its own logger from `DEFAULT_LOGGER_CONFIG` (not from a `LOGGER_CONFIG.md` file)
