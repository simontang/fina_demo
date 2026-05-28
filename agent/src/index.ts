import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgreSQLAssistantStore, PostgreSQLThreadStore, PostgreSQLDatabaseConfigStore, PostgreSQLWorkspaceStore, PostgreSQLProjectStore, PostgreSQLUserStore, PostgreSQLTenantStore, PostgreSQLUserTenantLinkStore, PostgreSQLMetricsServerConfigStore, PostgreSQLMcpServerConfigStore, PostgreSQLScheduleStorage, PostgreSQLWorkflowTrackingStore, ThreadMessageQueueStore, PostgreSQLEvalStore } from "@axiom-lattice/pg-stores";
import { Pool } from "pg";
import {
  ScheduleType,
} from "@axiom-lattice/protocols";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, "../.env") });
const baseURL = process.env.LLM_BASE_URL || "https://llm.alphafina.cn/v1";

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
  createSandboxProvider,
  registerScheduleLattice,
} from "@axiom-lattice/core";

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

// registerModelLattice("default",
//   {
//     model: "qwen3.5-27b",
//     provider: "openai",
//     streaming: true,
//     apiKeyEnvName: "API_KEY3",
//     baseURL: "https://llm.alphafina.cn/v1",
//     // enableThinking: true,
//   }
// );
// registerModelLattice("gpt-5.4",
//   {
//     model: "gpt-5.4",
//     displayName: "GPT-5.4",
//     provider: "openai",
//     streaming: true,
//     apiKeyEnvName: "OPEN_API",
//     baseURL: "https://new.ibadoo.cn/v1",
//     // enableThinking: true,
//   }
// );
registerModelLattice(
  "kimi-k2.6",

  {
    model: "kimi-k2.6",
    displayName: "kimi-k2.6",
    //model: "qwen3.5-35b-a3b",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,
    modelKwargs: {
      "enable_thinking": false
    }
  }
);
registerModelLattice(
  "deepseek-v4-pro",

  {
    model: "deepseek-v4-pro",
    displayName: "deepseek-v4-pro",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,
    modelKwargs: {
      "enable_thinking": false
    }
  }
);
registerModelLattice(
  "deepseek-v4-flash",

  {
    model: "deepseek-v4-flash",
    displayName: "deepseek-v4-flash",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,
    modelKwargs: {
      "enable_thinking": false
    }
  }
);
registerModelLattice(
  "qwen3.6-35b-a3b",

  {
    model: "qwen3.6-35b-a3b",
    displayName: "qwen3.6-35b-a3b",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,

  }
);
registerModelLattice(
  "qwen3.6-27b",

  {
    model: "qwen3.6-27b",
    displayName: "qwen3.6-27b",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,

  }
);
registerModelLattice(
  "default",

  {
    model: "qwen3.6-plus",
    displayName: "qwen3.6-plus",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,
    modelKwargs: {
      "enable_thinking": false
    }

  }
);
registerModelLattice(
  "doubao-seed-2-0-pro",

  {
    displayName: "Doubao pro 2.0",
    model: "doubao-seed-2-0-pro-260215",
    provider: "volcengine",
    streaming: true,
    apiKeyEnvName: "VOLCENGINE_API_KEY2",
    modelKwargs: {
      "thinking": { "type": "disabled" }
    },
  }
);


// registerModelLattice("default", {
//   model: "qwen-plus",
//   provider: "openai",
//   streaming: true,
//   apiKeyEnvName: "DASHSCOPE_API_KEY",
//   baseURL: "https://llm.alphafina.cn/v1/chat/completions",
// });

if (process.env.NODE_ENV === "production") {
  const globalMemory = PostgresSaver.fromConnString(process.env.DATABASE_URL!);
  //globalMemory.setup();
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

// Initialize and register PostgreSQL WorkflowTrackingStore
const workflowTrackingStore = new PostgreSQLWorkflowTrackingStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});
storeLatticeManager.removeLattice("default", "workflowTracking");
registerStoreLattice("default", "workflowTracking", workflowTrackingStore);
console.log("PostgreSQL WorkflowTrackingStore initialized with auto-migration");

// Initialize and register PostgreSQL EvalStore
const evalStore = new PostgreSQLEvalStore({
  poolConfig: process.env.DATABASE_URL || "",
  autoMigrate: true,
});
storeLatticeManager.removeLattice("default", "eval");
registerStoreLattice("default", "eval", evalStore);
console.log("PostgreSQL EvalStore initialized with auto-migration");

// Initialize and register PostgreSQL ThreadMessageQueueStore
const messageQueueStore = ThreadMessageQueueStore.getInstance();
const messageQueuePool = new Pool({ connectionString: process.env.DATABASE_URL });
(async () => {
  await messageQueueStore.initialize(messageQueuePool, true);
  storeLatticeManager.removeLattice("default", "threadMessageQueue");
  registerStoreLattice("default", "threadMessageQueue", messageQueueStore);
  console.log("PostgreSQL ThreadMessageQueueStore initialized with auto-migration");
})();

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
const sandboxProvider = createSandboxProvider({
  type: (process.env.SANDBOX_PROVIDER_TYPE as "microsandbox-remote") || "microsandbox-remote",
  microsandboxServiceBaseURL: process.env.MICROSANDBOX_SERVICE_BASE_URL!,
});
sandboxLatticeManager.registerLattice("default", sandboxProvider)


  // Initialize and register PostgreSQL ScheduleStorage for scheduled tasks
  const scheduleStorage = new PostgreSQLScheduleStorage({
    poolConfig: process.env.DATABASE_URL || "",
    autoMigrate: false,
  });
  //await scheduleStorage.initialize();
  registerScheduleLattice("default", {
    name: "Default Scheduler",
    description: "Production scheduler with PostgreSQL persistence",
    type: ScheduleType.POSTGRES,
    storage: scheduleStorage,
  });

//migrateVectorStoreToPGVectorStore();

// 启动fastify服务器
const port = process.env.PORT ? parseInt(process.env.PORT) : 5702;
startServer(port);
