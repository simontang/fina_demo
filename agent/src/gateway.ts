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
import { registerPythonProxyRoutes } from "./routes/pythonProxy";
import { registerRtcRoutes } from "./routes/rtc";

const { app, startAsHttpEndpoint, configureSwagger } = LatticeGateway;

function registerApiRoutes(app: FastifyInstance): void {
  // API root information
  app.get("/", async () => {
    return {
      name: "Research Data Agent API",
      version: "1.0.0",
      status: "running",
      endpoints: {
        health: "/api/health",
        agents: "/api/agents",
        files: "/api/files",
        upload: "/api/files/upload",
        uploadMultiple: "/api/files/upload-multiple",
        datasets: "/api/datasets",
        rtc: "/api/rtc",
        runs: "/api/runs",
      },
    };
  });

  // Health check endpoint
  app.get("/health", async () => {
    return {
      status: "ok",
      timestamp: new Date().toISOString(),
    };
  });

  // Agent management endpoints
  app.get("/agents", getAgentList);
  app.get("/agents/:id", getAgent);

  // File upload endpoints
  app.post("/files/upload", uploadFile);
  app.post("/files/upload-multiple", uploadMultipleFiles);
  app.get("/files", getUploadedFiles);
  app.delete("/files/:filename", deleteFile);

  // Dataset management endpoints
  registerDatasetRoutes(app);
}

// 注册路由
export const registerRoutes = (app: FastifyInstance): void => {
  // Python prediction service reverse-proxy (frontend calls /api/v1/* via this agent).
  registerPythonProxyRoutes(app);

  // RTC / Voice Chat APIs (frontend calls /api/rtc/* via this agent).
  registerRtcRoutes(app);

  // Register LatticeGateway routes at root path (required for sub-agent calls to /api/runs)
  LatticeGateway.registerLatticeRoutes(app);

  // Register custom API routes under /api prefix
  app.register(async (apiApp) => registerApiRoutes(apiApp), { prefix: "/api" });
};

// 配置并启动服务器
export async function startServer(port: number = 5702) {
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
