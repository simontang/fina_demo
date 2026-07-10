# Recipe: Connecting an MCP Server

Model Context Protocol (MCP) servers provide external tools and resources to agents.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | Gateway startup config | Register MCP servers |
| 2 | `.env` | Store connection configs |
| 3 | `McpServerConfigStore` | Persist connection configs |

## Step 1: Configure MCP Server

```typescript
// In gateway startup
import { McpLatticeManager } from "@axiom-lattice/core";

const mcpMgr = McpLatticeManager.getInstance();

await mcpMgr.registerServers([
  {
    name: "filesystem",
    connection: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    },
  },
  {
    name: "github",
    connection: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-github"],
      env: {
        GITHUB_PERSONAL_ACCESS_TOKEN: process.env.GITHUB_TOKEN!,
      },
    },
  },
  {
    name: "postgres",
    connection: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-postgres", process.env.DATABASE_URL!],
    },
  },
]);
```

## Step 2: Transport Types

| Transport | Use When |
|---|---|
| `stdio` | Local MCP server (subprocess) |
| `sse` | Remote MCP server with SSE |
| `streamable_http` | Remote MCP server with HTTP streaming |

```typescript
// stdio (local process)
{
  name: "my-server",
  connection: {
    transport: "stdio",
    command: "node",
    args: ["./my-mcp-server.js"],
  },
}

// streamable_http (remote)
{
  name: "remote-server",
  connection: {
    transport: "streamable_http",
    url: "https://my-mcp-server.example.com/mcp",
    headers: {
      Authorization: `Bearer ${process.env.MCP_TOKEN}`,
    },
  },
}
```

## Step 3: Persist Configs via Store

MCP server configs are stored in `McpServerConfigStore`:

```bash
# Create config via API
curl -X POST http://localhost:4001/api/mcp-servers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "weather",
    "transport": "streamable_http",
    "url": "https://weather-mcp.example.com"
  }'

# List configs
curl http://localhost:4001/api/mcp-servers

# Delete config
curl -X DELETE http://localhost:4001/api/mcp-servers/:id
```

## Step 4: Use MCP Tools in Agents

Once registered, MCP tools are automatically available to agents. The MCP client discovers tools from the server at connection time and registers them in the tool lattice:

```typescript
// The agent automatically has access to all MCP tools:
const config: ReactAgentConfig = {
  type: AgentType.REACT,
  key: "dev-assistant",
  name: "Dev Assistant",
  modelKey: "azure-gpt-4o",
  prompt: "You have access to filesystem and GitHub tools via MCP.",
  // No explicit tool list needed — MCP tools are auto-discovered
};
registerAgentLattice(config);
```

## Gotchas

- MCP servers are discovered and connected at startup — they must be running/accessible
- `stdio` servers are spawned as child processes by the gateway
- MCP tool names may conflict with built-in tools — check for name collisions
- The `McpClient` currently uses LangChain's `@langchain/mcp-adapters` `MultiServerMCPClient`
- To use a non-LangChain MCP transport, implement `McpClient` interface from `packages/protocols/src/McpLatticeProtocol.ts`
