# Recipe: Sandbox Provider

Add a new sandbox execution backend.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `your-sandbox/MyProvider.ts` | Implement `SandboxProvider` |
| 2 | `your-sandbox/MySandboxInstance.ts` | Implement `SandboxInstance` |
| 3 | `packages/core/src/sandbox_lattice/SandboxProviderFactory.ts` | Add case to factory |

## Overview

The sandbox system provides isolated execution environments. The framework supports Daytona, E2B, and microsandbox out of the box.

## Step 1: Real Interfaces

### SandboxProvider

File: `packages/core/src/sandbox_lattice/contracts/SandboxProvider.ts`

```typescript
export interface SandboxProvider {
  createSandbox(name: string, config: RunSandboxConfig): Promise<SandboxInstance>;
  getSandbox(name: string): Promise<SandboxInstance>;
  stopSandbox(name: string): Promise<void>;
  deleteSandbox(name: string): Promise<void>;
  listSandboxes(): Promise<SandboxInstance[]>;
  createVolumeFsClient?(volumeName: string, pathPrefix?: string): VolumeFsClient;
}

// RunSandboxConfig
interface RunSandboxConfig {
  assistant_id: string;
  thread_id: string;
  tenantId?: string;
  workspaceId?: string;
  projectId?: string;
  vmIsolation?: "global" | "agent" | "project";
  volumes?: Record<string, SandboxVolumeDefinition>;
}
```

### SandboxInstance

File: `packages/core/src/sandbox_lattice/contracts/SandboxInstance.ts`

```typescript
export interface SandboxInstance {
  readonly name: string;
  start(): Promise<void>;
  stop(): Promise<void>;
  kill(): Promise<void>;
  getStatus(): Promise<"running" | "stopped" | "unknown">;
  readonly file: SandboxFileService;
  readonly shell: SandboxShellService;
}

export interface SandboxFileService {
  readFile(file: string): Promise<{ content: string }>;
  writeFile(file: string, content: string): Promise<void>;
  listPath(path: string, options?: { recursive?: boolean }): Promise<{ files: SandboxFileInfo[] }>;
  findFiles(path: string, glob: string): Promise<{ files: string[] }>;
  searchInFile(file: string, regex: string): Promise<{ matches: string[]; line_numbers: number[] }>;
  strReplaceEditor(params: {
    command: "str_replace";
    path: string;
    old_str: string;
    new_str: string;
    replace_mode: "FIRST" | "ALL";
  }): Promise<void>;
  uploadFile(params: { file: string; data: Buffer; encoding?: string }): Promise<void>;
  downloadFile(params: { file: string }): Promise<Buffer>;
  deletePath(path: string): Promise<void>;
  createDirectory(path: string): Promise<void>;
}

export interface SandboxShellService {
  execCommand(params: {
    command: string;
    exec_dir?: string;
    timeout?: number;
  }): Promise<{ output: string; exit_code: number }>;
}
```

## Step 2: Implement Provider (Docker Example)

```typescript
import type { SandboxProvider, SandboxInstance, RunSandboxConfig, VolumeFsClient } from "@axiom-lattice/core";

export class DockerSandboxProvider implements SandboxProvider {
  // Note: createSandbox(name, config) — config is RunSandboxConfig, not ad-hoc options
  async createSandbox(name: string, config: RunSandboxConfig): Promise<SandboxInstance> {
    return new DockerSandboxInstance(name, config);
  }

  async getSandbox(name: string): Promise<SandboxInstance> {
    return DockerSandboxInstance.fromExisting(name);
  }

  async stopSandbox(name: string): Promise<void> { /* ... */ }
  async deleteSandbox(name: string): Promise<void> { /* ... */ }

  // Returns SandboxInstance[], not string[]
  async listSandboxes(): Promise<SandboxInstance[]> { return []; }

  async createVolumeFsClient?(volumeName: string, pathPrefix?: string): VolumeFsClient {
    return new DockerVolumeFsClient(volumeName, pathPrefix);
  }
}
```

## Step 3: Implement SandboxInstance

```typescript
import type { SandboxInstance, SandboxFileService, SandboxShellService } from "@axiom-lattice/core";

export class DockerSandboxInstance implements SandboxInstance {
  readonly name: string;
  readonly file: SandboxFileService;
  readonly shell: SandboxShellService;

  constructor(name: string, private config: RunSandboxConfig) {
    this.name = name;

    this.shell = {
      // Returns { output: string; exit_code: number }
      execCommand: async (params) => {
        const { command, exec_dir, timeout } = params;
        // Execute command in docker container
        const output = await dockerExec(this.name, command, exec_dir);
        return { output, exit_code: 0 };
      },
    };

    this.file = {
      readFile: async (file) => ({ content: await dockerReadFile(this.name, file) }),
      writeFile: async (file, content) => { /* ... */ },
      listPath: async (path, opts) => ({ files: [] }),
      findFiles: async (path, glob) => ({ files: [] }),
      searchInFile: async (file, regex) => ({ matches: [], line_numbers: [] }),
      strReplaceEditor: async (params) => { /* ... */ },
      uploadFile: async (params) => { /* ... */ },
      downloadFile: async (params) => Buffer.from(""),
      deletePath: async (path) => { /* ... */ },
      createDirectory: async (path) => { /* ... */ },
    };
  }

  async start(): Promise<void> { /* docker start */ }
  async stop(): Promise<void> { /* docker stop */ }
  async kill(): Promise<void> { /* docker rm -f */ }

  // Returns union type, not generic string
  async getStatus(): Promise<"running" | "stopped" | "unknown"> {
    return "running";
  }
}
```

## Step 4: Register in Factory

File: `packages/core/src/sandbox_lattice/SandboxProviderFactory.ts`

The factory is a **standalone function** `createSandboxProvider(config: CreateSandboxProviderConfig)`, NOT a class with static method:

```typescript
export function createSandboxProvider(config: CreateSandboxProviderConfig): SandboxProvider {
  switch (config.type) {
    case "microsandbox-remote": return new MicrosandboxRemoteProvider(config);
    case "remote": return new RemoteSandboxProvider(config);
    case "e2b": return new E2BProvider(config);
    case "daytona": return new DaytonaProvider(config);
    case "docker": return new DockerSandboxProvider(config);  // Add your case
    default: throw new Error(`Unknown type: ${config.type}`);
  }
}
```

## Gotchas

- **Factory is `createSandboxProvider()` standalone function**, NOT `SandboxProviderFactory.createSandboxProvider()`
- **`createSandbox(name, config)`**: config is `RunSandboxConfig` (with `assistant_id`, `thread_id`, etc.), NOT ad-hoc `{ image, env, timeout }`
- **`listSandboxes()` returns `SandboxInstance[]`**, not `string[]`
- **`getStatus()` returns `"running" | "stopped" | "unknown"`**, not generic `string`
- **`SandboxShellService.execCommand()` returns `{ output, exit_code }`**, NOT `{ stdout, stderr, exitCode }`
- **`name` is `readonly`** on SandboxInstance
- **`SandboxFileService` has 10 methods** — implement all of them
- **`createVolumeFsClient(volumeName, pathPrefix?)`** has optional second parameter
- **SandboxProviderFactory has a hardcoded switch** — you MUST modify core source to add a new type
- **E2B class is `E2BProvider`**, Daytona class is `DaytonaProvider`
