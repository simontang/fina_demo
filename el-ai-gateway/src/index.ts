import "dotenv/config";
import { loadConfig } from "./config";
import { createAuthenticator } from "./auth";
import { createPlatformFilesClient } from "./upstream/platformFiles";
import { createAgentRunsClient } from "./upstream/agentRuns";
import { createMcpClient } from "./upstream/mcp";
import { createTaskToolClient } from "./upstream/taskTools";
import { createBoTools } from "./upstream/boTools";
import { buildServer } from "./server";

async function main(): Promise<void> {
  const config = loadConfig();
  const mcp = createMcpClient(config);
  const app = buildServer({
    config,
    authenticator: createAuthenticator(config),
    platformFiles: createPlatformFilesClient(config),
    agentRuns: createAgentRunsClient(config),
    taskTools: createTaskToolClient(mcp),
    boTools: createBoTools(mcp),
  });
  await app.listen({ port: config.port, host: "0.0.0.0" });
  app.log.info(`el-ai-gateway listening on :${config.port}`);
}

main().catch((error) => {
  console.error("Failed to start el-ai-gateway:", error);
  process.exit(1);
});
