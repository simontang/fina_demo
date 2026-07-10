# Recipe: Adding Custom Pages to the Web App

Register custom pages that appear in the workspace menu or chat panels.

## Overview

The web app is a single-page application. "Pages" are GenUI elements rendered inside a panel system (`ColumnLayout`: menu | main | detail | tools). There are no URL routes — everything is driven by the GenUI registry.

## Pattern: Add a Workspace Page

### Step 1: Create Your Component

```tsx
// my-app/pages/MyDashboard.tsx
import React from "react";
import type { ElementProps } from "@axiom-lattice/react-sdk";

export const MyDashboard: React.FC<ElementProps> = ({ context, data }) => {
  return (
    <div style={{ padding: 24 }}>
      <h2>My Dashboard</h2>
      <p>Thread: {context?.thread_id}</p>
      <p>Assistant: {context?.assistant_id}</p>
      {/* Your custom content */}
    </div>
  );
};
```

### Step 2: Register as GenUI Element

```tsx
// my-app/index.tsx — BEFORE rendering LatticeChatShell
import { regsiterElement } from "@axiom-lattice/react-sdk";
import { MyDashboard } from "./pages/MyDashboard";

regsiterElement("my_dashboard", {
  card_view: () => null,                    // unused for workspace pages
  side_app_view: MyDashboard,               // the panel component
});
```

### Step 3: Add Menu Item

```tsx
import { LatticeChatShell } from "@axiom-lattice/react-sdk";

<LatticeChatShell
  baseURL="http://localhost:4001"
  apiKey="your-key"
  assistantId="my-agent"
  transport="sse"
  enableWorkspace={true}
  workspaceMenuItems={[
    {
      id: "my_dashboard",       // must match the regsiterElement key
      type: "route",             // opens as a workspace content panel
      name: "My Dashboard",
      icon: <DashboardOutlined />,
    },
  ]}
/>
```

The menu item `id` resolves to the GenUI key via a built-in mapping. For custom IDs not in the map, it uses the `id` directly as the component key.

## Pattern: Add a Chat Detail Panel

```tsx
// Register an element that appears when the agent outputs a specific component_key
import { regsiterElement } from "@axiom-lattice/react-sdk";

regsiterElement("stock_analyzer", {
  card_view: ({ data }) => (
    <div className="card">
      <h3>{data.symbol}</h3>
      <span>{data.price}</span>
    </div>
  ),
  side_app_view: ({ data, context }) => (
    <div className="panel">
      <h2>Stock Analysis: {data.symbol}</h2>
      <TradingViewChart symbol={data.symbol} />
      <AnalysisReport threadId={context?.thread_id} />
    </div>
  ),
});
```

The agent can then emit JSON to trigger it:
```json
{
  "type": "genui",
  "component_key": "stock_analyzer",
  "data": { "symbol": "AAPL", "price": 175.50 }
}
```

## Pattern: Add a Drawer Panel (Chat Sidebar)

```tsx
<LatticeChatShell
  sideMenuItems={[
    { id: "tools", type: "drawer", name: "Tools", icon: <ToolOutlined />,
      content: <MyToolsPanel />, title: "Tools" },
  ]}
/>
```

Drawer-type menu items open a slide-out panel from the chat sidebar.

## Pattern: Create a Standalone Page (Outside LatticeChatShell)

For a completely custom page layout that doesn't use LatticeChatShell's panel system:

```tsx
// my-app/MyStandalonePage.tsx
import { AxiomLatticeProvider, useChat, useAgentState } from "@axiom-lattice/react-sdk";

function StandaloneChat() {
  const threadId = "my-thread";
  const { messages, sendMessage, isLoading } = useChat(threadId, "my-agent");
  const { agentState } = useAgentState(threadId, "my-agent");

  return (
    <div>
      <MessageList messages={messages} />
      <InputBox onSend={(text) => sendMessage({ input: { message: text } })} disabled={isLoading} />
      <StateViewer state={agentState} />
    </div>
  );
}

export function MyApp() {
  return (
    <AxiomLatticeProvider
      config={{ baseURL: "http://localhost:4001", apiKey: "...", transport: "sse" }}
    >
      <StandaloneChat />
    </AxiomLatticeProvider>
  );
}
```

## Custom Route Registration Summary

| What you want | How |
|---|---|
| New workspace page | `regsiterElement(key, { side_app_view })` + workspace menu item with matching `id` |
| Chat detail panel | `regsiterElement(key, { card_view, side_app_view })` — triggered by agent output |
| Drawer panel from sidebar | Side menu item with `type: "drawer"` + `content` |
| Completely custom layout | Use `AxiomLatticeProvider` + hooks directly, no LatticeChatShell |
| Custom API endpoint | Register a Fastify route on `LatticeGateway.app` before startup |

## Gotchas

- **`regsiterElement` is misspelled** in the framework — use this exact spelling
- **Menu item `id` must match the GenUI registration key** exactly
- **Register elements BEFORE rendering** — the registry is a global mutable map
- **Workspace pages need `enableWorkspace={true}`** on LatticeChatShell
- **Chat sidebar `type: "route"` is NOT implemented** — use `type: "drawer"` for sidebar panels
- The web app is a **Vite SPA** — there are no URL routes, everything is panel-based
