import dotenv from "dotenv";
import path from "path";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

dotenv.config({ path: path.resolve(__dirname, "../.env") });
import { startServer } from "./gateway";
import {
  registerCheckpointSaver,
  registerModelLattice,
  MemoryLatticeManager,
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

registerModelLattice("default", {
  model: process.env.VOLCENGINE_MODEL || "kimi-k2-250905",
  provider: "volcengine",
  streaming: true,
  apiKeyEnvName: "VOLCENGINE_API_KEY2",
  baseURL: process.env.VOLCENGINE_API_URL || "https://ark.cn-beijing.volces.com/api/v3",
  maxTokens: 32768,
});

// registerModelLattice("default", {
//   model: "qwen-plus",
//   provider: "openai",
//   streaming: true,
//   apiKeyEnvName: "DASHSCOPE_API_KEY",
//   baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
// });

// const globalMemory = PostgresSaver.fromConnString(process.env.DATABASE_URL!);
// globalMemory.setup();
// MemoryLatticeManager.getInstance().removeCheckpointSaver("default");
// registerCheckpointSaver("default", globalMemory);

//migrateVectorStoreToPGVectorStore();

// 启动fastify服务器
const port = process.env.PORT ? parseInt(process.env.PORT) : 6203;
startServer(port);
