import React from "react";
import { createRoot } from "react-dom/client";
import "@axiom-lattice/react-sdk/styles.css";
import "./styles/tokens.css";
import "./ui_lattices";
import { appRegistry } from "@axiom-lattice/react-sdk";
import App from "./App";
import DataAgentApp from "./DataAgentApp";
import { loadAppConfig } from "./config";

// Land in the built-in CoSpace app on mobile viewports; desktop keeps Workspace.
// Matches the SDK's own mobile breakpoint (see CoSpace mobile layout design).
const MOBILE_VIEWPORT_QUERY = "(max-width: 768px)";

if (
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia(MOBILE_VIEWPORT_QUERY).matches
) {
  appRegistry.setDefault("cospace");
}

const container = document.getElementById("root") as HTMLElement;
const root = createRoot(container);

loadAppConfig().then(() => {
  root.render(
    // <React.StrictMode>
    // <App />
    <DataAgentApp />
    // </React.StrictMode>
  );
});
