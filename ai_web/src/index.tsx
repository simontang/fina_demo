import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/tokens.css";
import "./ui_lattices";
import App from "./App";
import DataAgentApp from "./DataAgentApp";
import { loadAppConfig } from "./config";

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
