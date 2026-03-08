import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import path from "path";

export default defineConfig(({ mode }) => {
  // 根据环境变量或当前页面判断 base 路径
  // 开发模式下根据请求的 HTML 文件自动判断
  // 生产模式下构建多入口
  
  return {
    plugins: [react()],
    // 动态 base 路径 - 根据构建目标自动调整
    base: "/",
    server: {
      host: "0.0.0.0",
      port: 5701,
      proxy: {
        // Single backend for the frontend: the agent service.
        // The agent reverse-proxies `/api/v1/*` to the Python prediction service.
        "/api/v1": {
          target: "http://localhost:5702",
          changeOrigin: true,
        },
        "/api": {
          target: "http://localhost:5702",
          changeOrigin: true,
        },
      },
    },
    build: {
      rollupOptions: {
        input: {
          admin: path.resolve(__dirname, "index.html"),
          "data-agent": path.resolve(__dirname, "data-agent.html"),
        },
        output: {
          entryFileNames: (chunkInfo) => {
            // 根据入口名称输出到不同目录
            if (chunkInfo.name === "admin") {
              return "admin/[name]-[hash].js";
            }
            if (chunkInfo.name === "data-agent") {
              return "data-agent/[name]-[hash].js";
            }
            return "assets/[name]-[hash].js";
          },
          chunkFileNames: "assets/[name]-[hash].js",
          assetFileNames: (assetInfo) => {
            // 静态资源统一放在 assets 目录
            return "assets/[name]-[hash][extname]";
          },
        },
      },
    },
  };
});
