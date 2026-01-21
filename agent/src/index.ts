import dotenv from "dotenv";
import path from "path";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

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

registerModelLattice("default", {
  model: process.env.VOLCENGINE_MODEL || "kimi-k2-250905",
  provider: "volcengine",
  streaming: true,
  apiKeyEnvName: "VOLCENGINE_API_KEY2",
  baseURL: process.env.VOLCENGINE_API_URL || "https://ark.cn-beijing.volces.com/api/v3",
  maxTokens: 32768,
});





// Check which path exists, default to first path
let skillsRootDir: string = process.env.NODE_ENV === "production" ? "/app/lattice_store/skills" : path.resolve(__dirname, "../skills")




const skillStore = new FileSystemSkillStore({
  rootDir: skillsRootDir,
});

// Remove the default skill store and register our custom one
// This ensures tools like load_skills and load_skill_content can access our skills
storeLatticeManager.removeLattice("default", "skill");
registerStoreLattice("default", "skill", skillStore);

// Configure SkillLatticeManager to use the store
skillLatticeManager.configureStore("default");

// Test loading skills on startup to verify configuration
(async () => {
  try {
    const skills = await skillStore.getAllSkills();
    console.log(`Loaded ${skills.length} skills from file system:`);
    if (skills.length === 0) {
      console.warn(
        `Warning: No skills found. Please check if the directory exists: ${skillsRootDir}`
      );
    } else {
      skills.forEach((skill) => {
        console.log(`  - ${skill.name}: ${skill.description.substring(0, 50)}...`);
      });
    }
  } catch (error) {
    console.error("Failed to load skills on startup:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
      console.error("Stack:", error.stack);
    }
  }
})();
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
const port = process.env.PORT ? parseInt(process.env.PORT) : 5702;
startServer(port);
