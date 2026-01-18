import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/tokens.css";
import "./ui_lattices";
import App from "./App";

const container = document.getElementById("root") as HTMLElement;
const root = createRoot(container);

root.render(
  // <React.StrictMode>
  <App />
  // </React.StrictMode>
);
