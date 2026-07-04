import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { createPgStoreConfig } from "@axiom-lattice/pg-stores";
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
  sqlDatabaseManager,
  metricsServerManager,
  registerScheduleLattice,
  configureStores,
  sandboxLatticeManager,
  createSandboxProvider,
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

// 如果设置了 MODEL_LIST 环境变量，则跳过所有代码注册
if (!process.env.MODEL_LIST) {
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
registerModelLattice(
  "gpt-5.5",

  {
    model: "gpt-5.5",
    displayName: "gpt-5.5",
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,
    modelKwargs: {
      "thinking_effort": "high"
    }
  }
);

}  // end of env var guard: only register hardcoded models if MODEL_LIST is not set


// 环境变量注册模型（优先级高于代码注册）
const extraModels = process.env.MODEL_LIST;
if (extraModels) {
  const modelNames = extraModels.split(",").map(s => s.trim()).filter(Boolean);
  for (const modelName of modelNames) {
    registerModelLattice(modelName, {
      model: modelName,
      displayName: modelName,
      provider: "openai",
      streaming: true,
      apiKeyEnvName: "API_KEY3",
      baseURL: baseURL,
    });
  }
  console.log(`[env] Registered ${modelNames.length} model(s) from MODEL_LIST: ${modelNames.join(", ")}`);
}

const defaultModel = process.env.DEFAULT_MODEL;
if (defaultModel) {
  registerModelLattice("default", {
    model: defaultModel,
    displayName: defaultModel,
    provider: "openai",
    streaming: true,
    apiKeyEnvName: "API_KEY3",
    baseURL: baseURL,
    modelKwargs: {
      "enable_thinking": false
    }
  });
  console.log(`[env] Default model set to: ${defaultModel}`);
}

// registerModelLattice("default", {
//   model: "qwen-plus",
//   provider: "openai",
//   streaming: true,
//   apiKeyEnvName: "DASHSCOPE_API_KEY",
//   baseURL: "https://llm.alphafina.cn/v1/chat/completions",
// });

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

async function initializePgStores(): Promise<void> {
  const connectionString = process.env.DATABASE_URL || "";

  if (!connectionString) {
    console.error("ERROR: DATABASE_URL environment variable is not set");
    process.exit(1);
  }

  console.log("\n🔌 Initializing PostgreSQL stores...\n");

  const stores = await createPgStoreConfig(connectionString);

  await configureStores({
    ...stores,
  });

  // Additional config loading after stores are registered
  sqlDatabaseManager.loadAllConfigsFromStore(stores.database);
  metricsServerManager.loadConfigsFromStore(stores.metrics, "default");
  metricsServerManager.loadConfigsFromStore(stores.metrics, "tenant_3");


  console.log("\n✓ All PostgreSQL stores initialized\n");
}
async function main() {
await initializePgStores();

// Sandbox provider 由 gateway 框架自动根据环境变量注册，无需在此手动配置

  // Register Sandbox Manager Lattice
  const sandboxProviderType = process.env.SANDBOX_PROVIDER_TYPE || "microsandbox";
  const sandboxBaseURL = process.env.SANDBOX_BASE_URL;
  const microsandboxServiceBaseURL = process.env.MICROSANDBOX_SERVICE_BASE_URL;
  const e2bApiKey = process.env.E2B_API_KEY;
  const daytonaApiKey = process.env.DAYTONA_API_KEY;
  const daytonaApiUrl = process.env.DAYTONA_API_URL;
  const daytonaTarget = process.env.DAYTONA_TARGET;

  const sandboxProvider = createSandboxProvider({
    type: sandboxProviderType as any,
    remoteBaseURL: sandboxBaseURL,
    microsandboxServiceBaseURL,
    e2bApiKey,
    e2bTemplate: process.env.E2B_TEMPLATE,
    e2bTimeoutMs: process.env.E2B_TIMEOUT_MS
      ? parseInt(process.env.E2B_TIMEOUT_MS, 10)
      : undefined,
    daytonaApiKey,
    daytonaApiUrl,
    daytonaTarget,
    daytonaTimeout: process.env.DAYTONA_TIMEOUT
      ? parseInt(process.env.DAYTONA_TIMEOUT, 10)
      : undefined,
    daytonaVolumeName: process.env.DAYTONA_VOLUME_NAME,
  });

  sandboxLatticeManager.registerLattice("default", sandboxProvider);

  console.log(`✓ Sandbox provider registered: ${sandboxProviderType}`);

//migrateVectorStoreToPGVectorStore();

// 启动fastify服务器
const port = process.env.PORT ? parseInt(process.env.PORT) : 5702;
startServer(port);
}

// Run main function
main().catch((error) => {
  console.error("Failed to start server:", error);
  process.exit(1);
});
