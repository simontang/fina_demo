import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Base path for sub-application deployment under /admin
  base: "/admin/",
  server: {
    host: "0.0.0.0",
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
});
