import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Base path for sub-application deployment under /admin
  base: "/admin/",
  server: {
    host: "0.0.0.0",
    proxy: {
      "/api/v1/datasets": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/v1/models": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/v1/model-assets": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api": {
        target: "http://localhost:6203",
        changeOrigin: true,
      },
    },
  },
});
