import "dotenv/config";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const url = process.env.MCP_SERVER_URL ?? "http://127.0.0.1:5702/open/mcp";
const key = process.env.MCP_API_KEY;
if (!key) {
  console.error("MCP_API_KEY is required");
  process.exit(1);
}

const transport = new StreamableHTTPClientTransport(new URL(url), {
  requestInit: { headers: { Authorization: `Bearer ${key}` } },
});
const client = new Client({ name: "el-ai-gateway-probe", version: "0.1.0" }, { capabilities: {} });
await client.connect(transport);

const tools = await client.listTools();
console.log("tools:", JSON.stringify(tools, null, 2));

if (process.env.MCP_PROBE_ACTION === "create_task") {
  const created = await client.callTool({
    name: "task_manage_task",
    arguments: { action: "create", title: "el-ai-gateway probe", status: "in_progress" },
  });
  console.log("create_task:", JSON.stringify(created, null, 2));
}

await client.close();
