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
        input: path.resolve(__dirname, "index.html"),
        output: {
          entryFileNames: "assets/[name]-[hash].js",
          chunkFileNames: "assets/[name]-[hash].js",
          assetFileNames: "assets/[name]-[hash][extname]",
        },
      },
    },
  };
});
