import "dotenv/config";
import { loadConfig } from "./config";
import { createAuthenticator } from "./auth";
import { createPlatformFilesClient } from "./upstream/platformFiles";
import { createA2AClient } from "./upstream/a2a";
import { createMcpClient } from "./upstream/mcp";
import { createTaskToolClient } from "./upstream/taskTools";
import { buildServer } from "./server";

async function main(): Promise<void> {
  const config = loadConfig();
  const mcp = createMcpClient(config);
  const app = buildServer({
    config,
    authenticator: createAuthenticator(config),
    platformFiles: createPlatformFilesClient(config),
    a2a: createA2AClient(config),
    taskTools: createTaskToolClient(mcp),
  });
  await app.listen({ port: config.port, host: "0.0.0.0" });
  app.log.info(`el-ai-gateway listening on :${config.port}`);
}

main().catch((error) => {
  console.error("Failed to start el-ai-gateway:", error);
  process.exit(1);
});
