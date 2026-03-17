import dotenv from "dotenv";
import path from "path";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgreSQLAssistantStore, PostgreSQLThreadStore, PostgreSQLDatabaseConfigStore, PostgreSQLWorkspaceStore, PostgreSQLProjectStore, PostgreSQLUserStore, PostgreSQLTenantStore, PostgreSQLUserTenantLinkStore, PostgreSQLMetricsServerConfigStore, PostgreSQLMcpServerConfigStore } from "@axiom-lattice/pg-stores";

dotenv.config({ path: path.resolve(__dirname, "../.env") });
import { startServer } from "./gateway";
import {
  registerCheckpointSaver,
  registerModelLattice,
  MemoryLatticeManager,
  skillLatticeManager,
  registerStoreLattice,
  storeLatticeManager,
  FileSystemSkillStore,
  sandboxLatticeManager,
  sqlDatabaseManager,
  metricsServerManager,
  SandboxSkillStore,
} from "@axiom-lattice/core";
const fs = require("fs");

import "./agents";

// 加载环境变量

// registerModelLattice("default", {
//   model: "kimi-k2-0711-preview",
//   provider: "openai",
//   streaming: true,
//   apiKeyEnvName: "KIMI_API_KEY",
//   baseURL: "https://api.moonshot.cn/v1",
// });

// registerModelLattice("default", {
//   model: "deepseek-chat",
//   provider: "deepseek",
//   streaming: true,
// });

// registerModelLattice("default", {
//   model: process.env.VOLCENGINE_MODEL || "kimi-k2-250905",
//   provider: "volcengine",
//   streaming: true,
//   apiKeyEnvName: "VOLCENGINE_API_KEY2",
//   baseURL: process.env.VOLCENGINE_API_URL || "https://ark.cn-beijing.volces.com/api/v3",
//   maxTokens: 32768,
// });

registerModelLattice("default",
  {
    model: "qwen3.5-plus",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    // enableThinking: true,
  }
);



// registerModelLattice("default", {
//   model: "qwen-plus",
//   provider: "openai",
//   streaming: true,
//   apiKeyEnvName: "DASHSCOPE_API_KEY",
//   baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
// });

if (process.env.NODE_ENV === "production") {
  const globalMemory = PostgresSaver.fromConnString(process.env.DATABASE_URL!);
  globalMemory.setup();
  MemoryLatticeManager.getInstance().removeCheckpointSaver("default");
  registerCheckpointSaver("default", globalMemory);


  // Create and initialize PostgreSQL ThreadStore
  const threadStore = new PostgreSQLThreadStore({
    poolConfig: process.env.DATABASE_URL || "",
  });

  // Initialize (runs migrations automatically)
  //threadStore.initialize();
  storeLatticeManager.removeLattice("default", "thread");
  registerStoreLattice("default", "thread", threadStore);

}


// Create and initialize AssistantStore with connection string
const assistantStore = new PostgreSQLAssistantStore({
  poolConfig: process.env.DATABASE_URL || "",
});

// Ensure initialization (migrations run automatically)
// assistantStore.initialize();

// Register to StoreLatticeManager
storeLatticeManager.removeLattice("default", "assistant");

registerStoreLattice("default", "assistant", assistantStore);



// Initialize and register PostgreSQL DatabaseConfigStore
// This stores database connection configurations with encryption
const databaseConfigStore = new PostgreSQLDatabaseConfigStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});

// Register databaseConfigStore to replace the default in-memory store
storeLatticeManager.removeLattice("default", "database");
registerStoreLattice("default", "database", databaseConfigStore);
sqlDatabaseManager.loadAllConfigsFromStore(databaseConfigStore)
console.log("PostgreSQL DatabaseConfigStore initialized with auto-migration");



// Initialize and register PostgreSQL MetricsServerConfigStore
// This stores metrics server configurations with apiKey and password encryption
const metricsConfigStore = new PostgreSQLMetricsServerConfigStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});

// Register metricsConfigStore to replace the default in-memory store
storeLatticeManager.removeLattice("default", "metrics");
registerStoreLattice("default", "metrics", metricsConfigStore);
metricsServerManager.loadAllConfigsFromStore(metricsConfigStore)
console.log("PostgreSQL MetricsServerConfigStore initialized with auto-migration");



// Initialize and register PostgreSQL McpServerConfigStore
// This stores MCP server configurations with env encryption
const mcpConfigStore = new PostgreSQLMcpServerConfigStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});

// Register mcpConfigStore to replace the default in-memory store
storeLatticeManager.removeLattice("default", "mcp");
registerStoreLattice("default", "mcp", mcpConfigStore);
console.log("PostgreSQL McpServerConfigStore initialized with auto-migration");


// Initialize and register PostgreSQL WorkspaceStore
const workspaceStore = new PostgreSQLWorkspaceStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});

// Register workspaceStore to replace the default in-memory store
storeLatticeManager.removeLattice("default", "workspace");
registerStoreLattice("default", "workspace", workspaceStore);
console.log("PostgreSQL WorkspaceStore initialized with auto-migration");

// Initialize and register PostgreSQL ProjectStore
const projectStore = new PostgreSQLProjectStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});

// Register projectStore to replace the default in-memory store
storeLatticeManager.removeLattice("default", "project");
registerStoreLattice("default", "project", projectStore);
console.log("PostgreSQL ProjectStore initialized with auto-migration");

// Initialize and register PostgreSQL UserStore (for authentication)
const userStore = new PostgreSQLUserStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});
storeLatticeManager.removeLattice("default", "user");
registerStoreLattice("default", "user", userStore);
console.log("PostgreSQL UserStore initialized with auto-migration");

// Initialize and register PostgreSQL TenantStore (for multi-tenancy)
const tenantStore = new PostgreSQLTenantStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});
storeLatticeManager.removeLattice("default", "tenant");
registerStoreLattice("default", "tenant", tenantStore);
console.log("PostgreSQL TenantStore initialized with auto-migration");

// Initialize and register PostgreSQL UserTenantLinkStore (for user-tenant relationships)
const userTenantLinkStore = new PostgreSQLUserTenantLinkStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});
storeLatticeManager.removeLattice("default", "userTenantLink");
registerStoreLattice("default", "userTenantLink", userTenantLinkStore);
console.log("PostgreSQL UserTenantLinkStore initialized with auto-migration");

// Auth configuration
const AUTH_CONFIG = {
  autoApproveUsers: process.env.AUTO_APPROVE_USERS !== "false",
  allowTenantRegistration: process.env.ALLOW_TENANT_REGISTRATION !== "false",
  jwtSecret: process.env.JWT_SECRET || "your-secret-key-change-in-production",
  tokenExpiration: parseInt(process.env.TOKEN_EXPIRATION || "86400", 10),
};

console.log("Auth Configuration:");
console.log(`  - Auto Approve Users: ${AUTH_CONFIG.autoApproveUsers}`);
console.log(`  - Allow Tenant Registration: ${AUTH_CONFIG.allowTenantRegistration}`);
console.log(`  - Token Expiration: ${AUTH_CONFIG.tokenExpiration}s`);

//Register Sandbox Manager Lattice
sandboxLatticeManager.registerLattice("default", { baseURL: "https://demo.alphafina.cn" })


const skillStore = new SandboxSkillStore({
  sandboxManager: sandboxLatticeManager.getSandboxLattice("default"),
});

// Remove the default skill store and register our custom one
// This ensures tools like load_skills and load_skill_content can access our skills
storeLatticeManager.removeLattice("default", "skill");
registerStoreLattice("default", "skill", skillStore);

// Configure SkillLatticeManager to use the store
skillLatticeManager.configureStore("default");


//migrateVectorStoreToPGVectorStore();

// 启动fastify服务器
const port = process.env.PORT ? parseInt(process.env.PORT) : 5702;
startServer(port);
