# Recipe: Customizing UI

Customize the chat interface, menus, and GenUI components.

## Overview

The react-sdk provides `AxiomLatticeProvider` for configuration, `LatticeChatShell` for the full chat UI, hooks for custom integrations, and a GenUI system for agent-rendered widgets.

---

## Step 1: Provider Setup

```tsx
import { AxiomLatticeProvider } from "@axiom-lattice/react-sdk";

function App() {
  return (
    <AxiomLatticeProvider
      config={{
        baseURL: "http://localhost:4001",
        apiKey: "your-api-key",
        assistantId: "default-agent",  // optional default
        transport: "sse",             // "sse" | "ws"
        timeout: 30000,
        headers: {
          "x-tenant-id": "my-tenant",
        },
      }}
      onUnauthorized={() => {
        // Handle 401 — redirect to login, etc.
      }}
    >
      <YourApp />
    </AxiomLatticeProvider>
  );
}
```

**`ClientConfig` fields:**

| Field | Required | Description |
|---|---|---|
| `baseURL` | Yes | Gateway URL |
| `apiKey` | Yes | API key or JWT token |
| `transport` | Yes | `"sse"` or `"ws"` |
| `assistantId` | No | Default assistant |
| `timeout` | No | Request timeout (ms) |
| `headers` | No | Additional HTTP headers |

---

## Step 2: LatticeChatShell (Full Chat UI)

```tsx
import { LatticeChatShell } from "@axiom-lattice/react-sdk";

function ChatPage() {
  return (
    <LatticeChatShell
      baseURL="http://localhost:4001"
      apiKey="your-key"
      assistantId="my-agent"
      transport="sse"
      // Optional customization:
      enableAssistantCreation={false}
      enableAssistantEditing={true}
      enableWorkspace={true}
      enableModelSelector={true}
      showSideMenu={true}
      // Menu customization (via LatticeChatShellConfig)
      sideMenuItems={customMenuItems}
      workspaceMenuItems={customWorkspaceItems}
      // Sidebar
      sidebarMode="fixed"          // "fixed" | "collapsible"
      sidebarDefaultExpanded={true}
      sidebarLogoText="My App"
    />
  );
}
```

See `docs/menu-customization-guide.md` for menu customization details.

---

## Step 3: Custom Chat with Hooks

For custom UI beyond LatticeChatShell, use hooks directly:

```tsx
import { useChat, useAgentState, useAgentGraph } from "@axiom-lattice/react-sdk";

function CustomChat({ threadId }: { threadId: string }) {
  // useChat(threadId, assistantId, options?) — positional params
  const {
    messages,
    isLoading,        // streaming state (NOT isStreaming)
    error,
    sendMessage,      // NOT send
    stopStreaming,    // stop generation
    loadMessages,
    clearMessages,
    clearError,
  } = useChat(threadId, "my-agent");

  // useAgentState(threadId, assistantId, options?) — positional params
  const {
    agentState,       // NOT state
    isLoading: stateLoading,
    error: stateError,
    startPolling,
    stopPolling,
    refresh,
  } = useAgentState(threadId, "my-agent");

  // useAgentGraph(assistantId) — single string param
  const {
    graphImage,       // NOT nodes/edges — returns a renderable graph image
    isLoading: graphLoading,
    fetchGraph,
  } = useAgentGraph("my-agent");

  const handleSend = (message: string) => {
    sendMessage({
      input: { message },
      streaming: true,
    });
  };

  return (
    <div>
      <button onClick={stopStreaming} disabled={!isLoading}>
        Stop
      </button>
      {/* render messages, graphImage, etc. */}
    </div>
  );
}
```

### Available Hooks

| Hook | Signature | Key Return Values |
|---|---|---|
| `useChat` | `(threadId, assistantId, options?)` | `messages`, `isLoading`, `sendMessage`, `stopStreaming` |
| `useAgentState` | `(threadId, assistantId, options?)` | `agentState`, `isLoading`, `startPolling`, `stopPolling` |
| `useAgentGraph` | `(assistantId)` | `graphImage`, `isLoading`, `fetchGraph` |
| `useApi` | `()` | `client` (raw API client) |

There is NO `useThread` hook. Thread management is handled internally by `useChat`.

---

## Step 4: Custom GenUI Elements

GenUI allows agents to render custom widgets in chat. Register elements with `regsiterElement` (note: framework has a typo — `regsiterElement`, not `registerElement`):

```tsx
import { regsiterElement } from "@axiom-lattice/react-sdk";
import type { ElementProps } from "@axiom-lattice/react-sdk";

// Register a custom element
regsiterElement("stock-chart", {
  // Required: how to render in the chat card
  card_view: ({ data }: ElementProps<{ symbol: string }>) => (
    <div className="stock-chart">
      <TradingViewWidget symbol={data.symbol} />
    </div>
  ),
  // Optional: expanded side-app view
  side_app_view: ({ data }: ElementProps<{ symbol: string }>) => (
    <FullScreenChart symbol={data.symbol} />
  ),
  // Optional: action when clicked
  action: (data) => {
    console.log("Chart clicked:", data);
  },
});
```

Agent tools output GenUI JSON like:
```json
{
  "type": "genui",
  "component_key": "stock-chart",
  "data": { "symbol": "AAPL" }
}
```

---

## Gotchas

- **`regsiterElement`** is misspelled in the framework (missing `r` before `s`) — use this exact spelling
- **`ElementMeta`** has `card_view` (React.FC) and `side_app_view?` — NOT `component`/`schema`
- **`AxiomLatticeProvider`** takes `config: ClientConfig` with `baseURL`/`apiKey`/`transport` — NOT `apiBaseUrl`
- **`useChat`** returns `isLoading` (not `isStreaming`), `sendMessage` (not `send`), no `abort` (use `stopStreaming`)
- **`useAgentState`** returns `agentState` (not `state`) and uses positional params, not object
- **`useAgentGraph`** returns `graphImage` (not `nodes`/`edges`) and takes a single string param
- **No `useThread` hook** exists — threads are managed by `useChat` internally
- For full menu/sidebar customization, see `docs/menu-customization-guide.md`
