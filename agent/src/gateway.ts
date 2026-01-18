import Fastify, { FastifyInstance } from "fastify";
import multipart from "@fastify/multipart";

import { LatticeGateway } from "@axiom-lattice/gateway";
import { getAgentList, getAgent } from "./controllers/agentController";
import {
  uploadFile,
  uploadMultipleFiles,
  getUploadedFiles,
  deleteFile,
} from "./controllers/fileController";
import { registerDatasetRoutes } from "./routes/datasets";

const { app, startAsHttpEndpoint, configureSwagger } = LatticeGateway;

// 注册路由
export const registerRoutes = (app: FastifyInstance): void => {
  // 注册所有路由到 bff 前缀下
  app.register(async (agentApp) => {
    agentApp.get("/", async (request, reply) => {
      return {
        name: "Research Data Agent API",
        version: "1.0.0",
        status: "running",
        endpoints: {
          health: "/bff/health",
          agents: "/bff/agents",
          files: "/bff/files",
          upload: "/bff/files/upload",
          uploadMultiple: "/bff/files/upload-multiple",
          datasets: "/bff/datasets",
        },
      };
    });

    // Agent management endpoints
    LatticeGateway.registerLatticeRoutes(agentApp);
    agentApp.get("/agents", getAgentList);
    agentApp.get("/agents/:id", getAgent);

    // File upload endpoints
    agentApp.post("/files/upload", uploadFile);
    agentApp.post("/files/upload-multiple", uploadMultipleFiles);
    agentApp.get("/files", getUploadedFiles);
    agentApp.delete("/files/:filename", deleteFile);

    // Dataset management endpoints
    registerDatasetRoutes(agentApp);
  }, { prefix: "/bff" });
};

// 配置并启动服务器
export async function startServer(port: number = 3203) {
  try {
    // Register multipart plugin for file uploads
    await app.register(multipart, {
      limits: {
        fileSize: 50 * 1024 * 1024, // 50MB max file size
        files: 10, // Max 10 files per request
      },
    });

    // 注册路由
    registerRoutes(app);

    await startAsHttpEndpoint({
      port,
      queueServiceConfig: { type: "memory", defaultStartPollingQueue: true },
    });
    console.log(`🚀 Server running on http://localhost:${port}`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}
