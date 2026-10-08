# Plugin Tool Inline HTML UI (MCP Apps) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make selected read-only plugin tools render an inline HTML table (MCP App) in the chat console, without changing the JSON text the model receives.

**Architecture:** Each plugin ships a self-contained `*Ui.ts` (no cross-plugin sharing) exporting a `ui://` resource key, a full inline HTML document (MCP Apps handshake + generic read-only table renderer + embedded column config), and a `with*Ui(result)` helper that appends the `mcp_app` fence via `appendUiFence` from `@axiom-lattice/core`. The plugin declares `meta.tools[].ui` + `meta.uiResources`. Requires gateway ≥ 4.9.0 for the console inline-resolution path.

**Tech Stack:** TypeScript, LangChain `tool`/`createMiddleware`, `@axiom-lattice/core` (`appendUiFence`, `PluginRegistry`), `@axiom-lattice/protocols` (`McpUiRef`), Jest + ts-jest, pnpm, tsup.

**Spec:** `docs/superpowers/specs/2026-10-08-plugin-inline-html-ui-design.md`

---

## Appendix A: Canonical self-contained HTML template

Every `*Ui.ts` embeds this exact HTML (inside a TS template literal). Only two things change per file: the `<title>` text and the `var CONFIG = { ... };` object. Do NOT introduce `${` or backticks inside this HTML.

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PLUGIN UI</title>
    <style>
      :root { color-scheme: light dark; }
      * { box-sizing: border-box; }
      body { margin: 0; padding: 14px 16px; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; color: #1f2430; background: transparent; }
      h1 { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
      .subtitle { color: #6b7280; font-size: 12px; margin: 0 0 10px; }
      .summary { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 10px; color: #4b5563; font-size: 12px; }
      .summary .kv b { color: #111827; font-weight: 600; }
      table { width: 100%; border-collapse: collapse; }
      th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #eceef2; vertical-align: top; }
      th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: #6b7280; font-weight: 600; white-space: nowrap; }
      td { max-width: 320px; overflow: hidden; text-overflow: ellipsis; }
      tbody tr:hover td { background: #f9fafb; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
      .num { font-variant-numeric: tabular-nums; }
      .muted { color: #9ca3af; }
      .state { color: #6b7280; padding: 14px 2px; }
      .status { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
      .dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; background: #9ca3af; }
      .dot.ok { background: #22c55e; }
      .dot.bad { background: #ef4444; }
      .dot.warn { background: #f59e0b; }
      table.detail th { width: 200px; color: #6b7280; font-weight: 500; text-transform: none; letter-spacing: 0; font-size: 12px; }
      @media (prefers-color-scheme: dark) {
        body { color: #e5e7eb; }
        h1 { color: #f9fafb; }
        .summary .kv b { color: #f9fafb; }
        th, td { border-bottom-color: #2a2f3a; }
        th { color: #9ca3af; }
        tbody tr:hover td { background: #1b1f27; }
        .subtitle, .summary, .state { color: #9ca3af; }
      }
    </style>
  </head>
  <body>
    <div id="root"><p class="state">Loading…</p></div>
    <script>
      (function () {
        var CONFIG = __CONFIG__;
        var root = document.getElementById("root");
        function post(message) { window.parent.postMessage(message, "*"); }
        function notify(method, params) { post({ jsonrpc: "2.0", method: method, params: params }); }
        var nextId = 1;
        function rpc(method, params) {
          var id = nextId++;
          return new Promise(function (resolve, reject) {
            function onMessage(event) {
              var data = event.data;
              if (!data || data.id !== id) return;
              window.removeEventListener("message", onMessage);
              if (data.error) reject(new Error((data.error && data.error.message) || "rpc error"));
              else resolve(data.result);
            }
            window.addEventListener("message", onMessage);
            post({ jsonrpc: "2.0", id: id, method: method, params: params });
          });
        }
        function sendSize() {
          var width = Math.ceil(document.documentElement.scrollWidth || document.body.scrollWidth);
          var height = Math.ceil(document.documentElement.scrollHeight || document.body.scrollHeight);
          notify("ui/notifications/size-changed", { width: width, height: height });
        }
        function esc(value) {
          return String(value === null || value === undefined ? "" : value).replace(/[&<>"']/g, function (ch) {
            return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
          });
        }
        function getPath(obj, path) {
          if (!path) return undefined;
          var parts = path.split(".");
          var cur = obj;
          for (var i = 0; i < parts.length; i++) {
            if (cur === null || typeof cur !== "object") return undefined;
            cur = cur[parts[i]];
          }
          return cur;
        }
        function truncate(text, max) {
          var s = String(text);
          return s.length > max ? s.slice(0, max - 1) + "…" : s;
        }
        function humanBytes(n) {
          if (!isFinite(n)) return String(n);
          var units = ["B", "KB", "MB", "GB", "TB"];
          var i = 0;
          while (n >= 1024 && i < units.length - 1) { n = n / 1024; i++; }
          return (i === 0 ? n : n.toFixed(1)) + " " + units[i];
        }
        function formatDate(value) {
          var d = new Date(value);
          return isNaN(d.getTime()) ? String(value) : d.toLocaleString();
        }
        function statusClass(status) {
          var s = status.toLowerCase();
          if (/run|ok|success|active|enabled|valid|complete/.test(s)) return "ok";
          if (/fail|error|invalid|deleted|disabled/.test(s)) return "bad";
          if (/pend|provision|request|stop|process|warn/.test(s)) return "warn";
          return "";
        }
        function jsonCell(value) {
          return '<span class="mono">' + esc(truncate(JSON.stringify(value), 120)) + "</span>";
        }
        function fmt(format, value) {
          if (value === null || value === undefined || value === "") return '<span class="muted">—</span>';
          switch (format) {
            case "mono": return '<span class="mono">' + esc(value) + "</span>";
            case "number": return '<span class="num">' + esc(value) + "</span>";
            case "bool": return value ? "✓" : "✗";
            case "count": return Array.isArray(value) ? String(value.length) : esc(value);
            case "status": return '<span class="status"><span class="dot ' + statusClass(String(value)) + '"></span>' + esc(value) + "</span>";
            case "bytes": return esc(humanBytes(Number(value)));
            case "datetime": return esc(formatDate(value));
            case "json": return jsonCell(value);
            default:
              if (typeof value === "object") return jsonCell(value);
              return esc(value);
          }
        }
        function pickRows(data) {
          if (Array.isArray(data)) return data;
          if (CONFIG.rowsPath) {
            var value = getPath(data, CONFIG.rowsPath);
            return Array.isArray(value) ? value : undefined;
          }
          if (data && typeof data === "object") {
            var keys = Object.keys(data);
            for (var i = 0; i < keys.length; i++) {
              if (Array.isArray(data[keys[i]])) return data[keys[i]];
            }
          }
          return undefined;
        }
        function columnsFor(rows) {
          var cols = (CONFIG.columns || []).slice();
          if (CONFIG.dynamic !== false) {
            var seen = {};
            for (var i = 0; i < cols.length; i++) seen[cols[i].key] = true;
            for (var r = 0; r < rows.length; r++) {
              var row = rows[r];
              if (row && typeof row === "object" && !Array.isArray(row)) {
                for (var k in row) {
                  if (!seen[k]) { seen[k] = true; cols.push({ key: k }); }
                }
              }
            }
          }
          return cols;
        }
        function renderSummary(data) {
          if (!CONFIG.summary || !data || typeof data !== "object") return "";
          var parts = [];
          for (var i = 0; i < CONFIG.summary.length; i++) {
            var key = CONFIG.summary[i];
            if (data[key] !== undefined && data[key] !== null && data[key] !== "") {
              parts.push('<span class="kv"><b>' + esc(key) + "</b> " + esc(data[key]) + "</span>");
            }
          }
          return parts.length ? '<div class="summary">' + parts.join("") + "</div>" : "";
        }
        function renderTable(rows) {
          if (!rows.length) return '<p class="state">' + esc(CONFIG.emptyText || "No data") + "</p>";
          var cols = columnsFor(rows);
          var html = "<table><thead><tr>";
          for (var c = 0; c < cols.length; c++) html += "<th>" + esc(cols[c].label || cols[c].key) + "</th>";
          html += "</tr></thead><tbody>";
          for (var r = 0; r < rows.length; r++) {
            html += "<tr>";
            for (var c2 = 0; c2 < cols.length; c2++) html += "<td>" + fmt(cols[c2].format, rows[r] ? rows[r][cols[c2].key] : undefined) + "</td>";
            html += "</tr>";
          }
          html += "</tbody></table>";
          return html;
        }
        function renderDetail(data) {
          var html = '<table class="detail"><tbody>';
          for (var k in data) {
            if (data[k] !== null && typeof data[k] === "object") continue;
            html += "<tr><th>" + esc(k) + "</th><td>" + fmt("text", data[k]) + "</td></tr>";
          }
          html += "</tbody></table>";
          return html;
        }
        function render(data) {
          var html = "<h1>" + esc(CONFIG.title) + "</h1>";
          if (CONFIG.subtitle) html += '<p class="subtitle">' + esc(CONFIG.subtitle) + "</p>";
          if (data === null || data === undefined) {
            root.innerHTML = html + '<p class="state">No result</p>';
            sendSize();
            return;
          }
          html += renderSummary(data);
          var rows = pickRows(data);
          if (rows) html += renderTable(rows);
          else if (CONFIG.detail !== false && data && typeof data === "object") html += renderDetail(data);
          else html += '<p class="state">' + esc(CONFIG.emptyText || "No data") + "</p>";
          root.innerHTML = html;
          sendSize();
        }
        function readResult(params) {
          if (!params || typeof params !== "object") return null;
          if (params.structuredContent && typeof params.structuredContent === "object") return params.structuredContent;
          var content = params.content;
          var text = content && content[0] && typeof content[0].text === "string" ? content[0].text : "";
          if (!text) return null;
          try { return JSON.parse(text); } catch (err) { return null; }
        }
        // 1) Register the tool-result listener BEFORE initializing so the data the
        // host pushes right after the handshake is never missed.
        window.addEventListener("message", function (event) {
          var data = event.data;
          if (!data || typeof data !== "object" || typeof data.method !== "string") return;
          if (data.method === "ui/notifications/tool-result") render(readResult(data.params));
        });
        if (typeof ResizeObserver !== "undefined") new ResizeObserver(sendSize).observe(document.documentElement);
        else window.addEventListener("resize", sendSize);
        // 2) MCP Apps lifecycle: initialize, then signal readiness.
        rpc("ui/initialize", {
          protocolVersion: "2025-06-18",
          capabilities: {},
          clientInfo: { name: "plugin-ui", version: "1.0.0" },
          appCapabilities: { availableDisplayModes: ["inline"] }
        }).then(function () {
          notify("ui/notifications/initialized", {});
          sendSize();
        }).catch(function (err) {
          root.innerHTML = '<p class="state">Failed to initialize: ' + esc((err && err.message) || err) + "</p>";
          sendSize();
        });
      })();
    </script>
  </body>
</html>
```

---

## Task 0: Upgrade `@axiom-lattice/*` dependencies

**Files:**
- Modify: `agent/package.json`, `agent/pnpm-lock.yaml`
- Modify: `ai_web/package.json`, `ai_web/pnpm-lock.yaml`

- [ ] **Step 1: Upgrade the agent packages**

Run (in `agent/`):
```bash
pnpm run up_lattice
```
Expected: `@axiom-lattice/core|gateway|pg-stores|protocols` move to their latest versions; `package.json` + `pnpm-lock.yaml` change. Confirm `@axiom-lattice/gateway` is ≥ `4.9.0`:
```bash
node -p "require('./node_modules/@axiom-lattice/gateway/package.json').version"
```

- [ ] **Step 2: Upgrade the web packages**

Run (in `ai_web/`):
```bash
pnpm run up_lattice
```

- [ ] **Step 3: Verify existing agent tests still pass**

Run (in `agent/`):
```bash
pnpm test
```
Expected: all existing tests pass. If any fail due to the upgrade, stop and resolve before continuing.

- [ ] **Step 4: Commit**

```bash
git add agent/package.json agent/pnpm-lock.yaml ai_web/package.json ai_web/pnpm-lock.yaml
git commit -m "chore(deps): bump @axiom-lattice packages for plugin inline UI (gateway >= 4.9.0)"
```

---

## Task 1: Storage UI module

**Files:**
- Create: `agent/src/agents/platform_service/storage/storageUi.ts`
- Test: `agent/src/agents/platform_service/storage/__tests__/storageUi.test.ts`

- [ ] **Step 1: Write the failing test**

Create `agent/src/agents/platform_service/storage/__tests__/storageUi.test.ts`:

```ts
jest.mock("@axiom-lattice/core", () => ({
  appendUiFence: (content: unknown, ref: unknown) =>
    typeof content === "string"
      ? content + "\n\n```mcp_app\n" + JSON.stringify(ref) + "\n```"
      : content,
}));

import { describe, expect, it } from "@jest/globals";
import { FILES_UI_RESOURCE, FILES_APP_HTML, withFilesUi } from "../storageUi";

describe("storageUi", () => {
  it("declares the files ui resource", () => {
    expect(FILES_UI_RESOURCE).toBe("ui://storage/files");
  });

  it("html carries the mcp apps handshake and embedded config", () => {
    expect(FILES_APP_HTML).toContain("ui/initialize");
    expect(FILES_APP_HTML).toContain("ui/notifications/initialized");
    expect(FILES_APP_HTML).toContain("ui/notifications/tool-result");
    expect(FILES_APP_HTML).toContain("ui/notifications/size-changed");
    expect(FILES_APP_HTML).toContain('"rowsPath": "files"');
    expect(FILES_APP_HTML).toContain("No files");
  });

  it("appends the fence with the storage plugin ref for a successful payload", () => {
    const out = withFilesUi(JSON.stringify({ path: "", files: [], total: 0 }));
    expect(out).toContain("```mcp_app");
    expect(out).toContain('"pluginType":"storage"');
    expect(out).toContain('"resource":"ui://storage/files"');
  });

  it("returns error payloads and non-json unchanged", () => {
    const err = JSON.stringify({ ok: false, code: "X", message: "nope" });
    expect(withFilesUi(err)).toBe(err);
    expect(withFilesUi("Error: boom")).toBe("Error: boom");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/storage/__tests__/storageUi.test.ts
```
Expected: FAIL — cannot find module `../storageUi`.

- [ ] **Step 3: Implement `storageUi.ts`**

Create `agent/src/agents/platform_service/storage/storageUi.ts`. Paste **Appendix A** inside the template literal, replacing `<title>PLUGIN UI</title>` with `<title>Files</title>` and `__CONFIG__` with the object below:

```ts
import { appendUiFence } from "@axiom-lattice/core";
import type { McpUiRef } from "@axiom-lattice/protocols";

export const FILES_UI_RESOURCE = "ui://storage/files";

export const FILES_UI_REF: McpUiRef = {
  kind: "plugin",
  pluginType: "storage",
  resource: FILES_UI_RESOURCE,
  displayMode: "inline",
};

// CONFIG to embed in Appendix A (as `var CONFIG = { ... };`):
// {
//   "title": "Files",
//   "rowsPath": "files",
//   "summary": ["path", "total", "page", "size", "totalPages"],
//   "emptyText": "No files",
//   "columns": [
//     { "key": "filename", "label": "Name" },
//     { "key": "fullPath", "label": "Path", "format": "mono" },
//     { "key": "size", "label": "Size", "format": "bytes" },
//     { "key": "mime", "label": "MIME" },
//     { "key": "fileCategory", "label": "Category" },
//     { "key": "usage", "label": "Usage" },
//     { "key": "version", "label": "Ver", "format": "number" },
//     { "key": "status", "label": "Status", "format": "status" },
//     { "key": "createdAt", "label": "Created", "format": "datetime" },
//     { "key": "uuid", "label": "UUID", "format": "mono" }
//   ]
// }

export const FILES_APP_HTML = `<!doctype html>
<html lang="en">
  ... Appendix A body with <title>Files</title> and
      var CONFIG = { "title": "Files", "rowsPath": "files", "summary": ["path", "total", "page", "size", "totalPages"], "emptyText": "No files", "columns": [ { "key": "filename", "label": "Name" }, { "key": "fullPath", "label": "Path", "format": "mono" }, { "key": "size", "label": "Size", "format": "bytes" }, { "key": "mime", "label": "MIME" }, { "key": "fileCategory", "label": "Category" }, { "key": "usage", "label": "Usage" }, { "key": "version", "label": "Ver", "format": "number" }, { "key": "status", "label": "Status", "format": "status" }, { "key": "createdAt", "label": "Created", "format": "datetime" }, { "key": "uuid", "label": "UUID", "format": "mono" } ] };
`;

/**
 * Append the Files MCP App fence to a tool result.
 *
 * Returns the result unchanged when it is not successful JSON (parse failure,
 * an `{ ok: false }` payload, or an `Error: ...` string) so error output never
 * renders a table.
 */
export function withFilesUi(result: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(result);
  } catch {
    return result;
  }
  if (parsed && typeof parsed === "object" && (parsed as { ok?: unknown }).ok === false) {
    return result;
  }
  return appendUiFence(result, FILES_UI_REF) as string;
}
```

**Important:** the actual `FILES_APP_HTML` must contain the FULL Appendix A document (all CSS + JS), with only the title and CONFIG substituted. The comment block above is a note to the implementer, not part of the file.

- [ ] **Step 4: Run the test to verify it passes**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/storage/__tests__/storageUi.test.ts
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add agent/src/agents/platform_service/storage/storageUi.ts agent/src/agents/platform_service/storage/__tests__/storageUi.test.ts
git commit -m "feat(agent): storage files MCP App UI module"
```

---

## Task 2: Wire the storage plugin

**Files:**
- Modify: `agent/src/agents/platform_service/storage/plugin.ts`
- Modify: `agent/src/agents/platform_service/__tests__/registration.test.ts`
- Modify: `agent/src/agents/platform_service/__tests__/barrel.test.ts`

- [ ] **Step 1: Add `appendUiFence` to the test mocks**

In `agent/src/agents/platform_service/__tests__/registration.test.ts` (lines 1-5) and `agent/src/agents/platform_service/__tests__/barrel.test.ts` (lines 1-5), add `appendUiFence` to the mocked `@axiom-lattice/core`. Example for both:

```ts
jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
  resolvePluginConnections: jest.fn(),
  appendUiFence: jest.fn((content: unknown) => content),
}));
```

(`barrel.test.ts` currently has no `resolvePluginConnections`; keep its mock otherwise unchanged, just add `appendUiFence`.)

- [ ] **Step 2: Add failing meta-invariant assertions to `registration.test.ts`**

Inside `describe("storage plugin", ...)`, add:

```ts
  it("declares the files ui resource and links it to the list tool", () => {
    expect(storagePlugin.meta.tools?.map((t) => t.name).sort()).toEqual([
      "delete",
      "get_download_url",
      "get_metadata",
      "list",
      "upload",
    ]);
    const listTool = storagePlugin.meta.tools?.find((t) => t.name === "list");
    expect(listTool?.ui?.resource).toBe("ui://storage/files");
    expect(Object.keys(storagePlugin.meta.uiResources ?? {})).toEqual(["ui://storage/files"]);
  });
```

- [ ] **Step 3: Run to verify it fails**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/__tests__/registration.test.ts
```
Expected: FAIL — `meta.tools` is undefined / `uiResources` missing.

- [ ] **Step 4: Update `plugin.ts`**

In `agent/src/agents/platform_service/storage/plugin.ts`:

1. Add the import:
```ts
import { FILES_UI_RESOURCE, FILES_APP_HTML, withFilesUi } from "./storageUi";
```

2. Inside `meta`, add `tools` and `uiResources` (keep the existing `openExpose`, `configSchema`, `defaultConfig`):
```ts
    tools: [
      { name: "upload", description: "Upload a file from the agent sandbox to unified storage." },
      { name: "list", description: "List files in unified storage, filterable and paginated.", ui: { resource: FILES_UI_RESOURCE, displayMode: "inline" } },
      { name: "get_metadata", description: "Get file metadata by uuid." },
      { name: "get_download_url", description: "Get a time-limited download link for a file." },
      { name: "delete", description: "Soft-delete one version of a file." },
    ],
    uiResources: {
      [FILES_UI_RESOURCE]: { html: FILES_APP_HTML },
    },
```

3. Wrap only the `list` tool's executor so its result carries the fence. Replace the existing `list` tool definition:
```ts
        tool(
          (input: z.infer<typeof SCHEMAS.list>, exeConfig) =>
            storageList(input, exeConfig, pluginConfig),
          {
            name: "list",
            description:
              "List files in unified storage, filterable by directory/name/attributes/time, returned paginated.",
            schema: SCHEMAS.list,
          },
        ),
```
with:
```ts
        tool(
          async (input: z.infer<typeof SCHEMAS.list>, exeConfig) =>
            withFilesUi(await storageList(input, exeConfig, pluginConfig)),
          {
            name: "list",
            description:
              "List files in unified storage, filterable by directory/name/attributes/time, returned paginated.",
            schema: SCHEMAS.list,
          },
        ),
```

- [ ] **Step 5: Run tests to verify they pass**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/__tests__/registration.test.ts src/agents/platform_service/__tests__/barrel.test.ts src/agents/platform_service/__tests__/storage.test.ts
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/src/agents/platform_service/storage/plugin.ts agent/src/agents/platform_service/__tests__/registration.test.ts agent/src/agents/platform_service/__tests__/barrel.test.ts
git commit -m "feat(agent): render storage.list as inline MCP App table"
```

---

## Task 3: Business Objects UI module

**Files:**
- Create: `agent/src/agents/platform_service/business_objects/businessObjectsUi.ts`
- Test: `agent/src/agents/platform_service/business_objects/__tests__/businessObjectsUi.test.ts`

- [ ] **Step 1: Write the failing test**

Create `agent/src/agents/platform_service/business_objects/__tests__/businessObjectsUi.test.ts`:

```ts
jest.mock("@axiom-lattice/core", () => ({
  appendUiFence: (content: unknown, ref: unknown) =>
    typeof content === "string"
      ? content + "\n\n```mcp_app\n" + JSON.stringify(ref) + "\n```"
      : content,
}));

import { describe, expect, it } from "@jest/globals";
import {
  OBJECTS_UI_RESOURCE,
  OBJECTS_APP_HTML,
  RECORDS_UI_RESOURCE,
  RECORDS_APP_HTML,
  withObjectsUi,
  withRecordsUi,
} from "../businessObjectsUi";

describe("businessObjectsUi", () => {
  it("declares both ui resources", () => {
    expect(OBJECTS_UI_RESOURCE).toBe("ui://business-objects/objects");
    expect(RECORDS_UI_RESOURCE).toBe("ui://business-objects/records");
  });

  it("both apps carry the handshake and their config", () => {
    for (const html of [OBJECTS_APP_HTML, RECORDS_APP_HTML]) {
      expect(html).toContain("ui/initialize");
      expect(html).toContain("ui/notifications/tool-result");
      expect(html).toContain("ui/notifications/size-changed");
    }
    expect(OBJECTS_APP_HTML).toContain("No objects");
    expect(RECORDS_APP_HTML).toContain('"rowsPath": "rows"');
  });

  it("appends the matching fence for successful payloads", () => {
    const objects = withObjectsUi(JSON.stringify([{ objectKey: "customer" }]));
    expect(objects).toContain('"resource":"ui://business-objects/objects"');
    const records = withRecordsUi(JSON.stringify({ objectKey: "customer", rows: [] }));
    expect(records).toContain('"resource":"ui://business-objects/records"');
  });

  it("returns error payloads and non-json unchanged", () => {
    const err = JSON.stringify({ ok: false, code: "X", message: "nope" });
    expect(withObjectsUi(err)).toBe(err);
    expect(withRecordsUi("Error: boom")).toBe("Error: boom");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/business_objects/__tests__/businessObjectsUi.test.ts
```
Expected: FAIL — cannot find module `../businessObjectsUi`.

- [ ] **Step 3: Implement `businessObjectsUi.ts`**

Create `agent/src/agents/platform_service/business_objects/businessObjectsUi.ts` with the same shape as `storageUi.ts` (imports, two resources, two `with*Ui` helpers). Build both HTML constants from **Appendix A**, substituting the title and CONFIG.

`OBJECTS_APP_HTML` — title `Business Objects`, CONFIG:
```json
{
  "title": "Business Objects",
  "emptyText": "No objects",
  "dynamic": false,
  "columns": [
    { "key": "objectKey", "label": "Key", "format": "mono" },
    { "key": "displayName", "label": "Name" },
    { "key": "tableName", "label": "Table", "format": "mono" },
    { "key": "storeKey", "label": "Store", "format": "mono" },
    { "key": "status", "label": "Status", "format": "status" },
    { "key": "deleteMode", "label": "Delete" },
    { "key": "fields", "label": "Fields", "format": "count" },
    { "key": "indexes", "label": "Indexes", "format": "count" },
    { "key": "description", "label": "Description" }
  ]
}
```

`RECORDS_APP_HTML` — title `Records`, CONFIG:
```json
{
  "title": "Records",
  "rowsPath": "rows",
  "summary": ["objectKey", "page", "pageSize", "total"],
  "emptyText": "No records",
  "dynamic": true,
  "columns": [
    { "key": "id", "label": "ID", "format": "mono" }
  ]
}
```

Constants and helpers:
```ts
import { appendUiFence } from "@axiom-lattice/core";
import type { McpUiRef } from "@axiom-lattice/protocols";

export const OBJECTS_UI_RESOURCE = "ui://business-objects/objects";
export const RECORDS_UI_RESOURCE = "ui://business-objects/records";

export const OBJECTS_UI_REF: McpUiRef = {
  kind: "plugin", pluginType: "business-objects", resource: OBJECTS_UI_RESOURCE, displayMode: "inline",
};
export const RECORDS_UI_REF: McpUiRef = {
  kind: "plugin", pluginType: "business-objects", resource: RECORDS_UI_RESOURCE, displayMode: "inline",
};

export const OBJECTS_APP_HTML = `... Appendix A with title "Business Objects" + CONFIG ...`;
export const RECORDS_APP_HTML = `... Appendix A with title "Records" + CONFIG ...`;

function append(result: string, ref: McpUiRef): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(result);
  } catch {
    return result;
  }
  if (parsed && typeof parsed === "object" && (parsed as { ok?: unknown }).ok === false) {
    return result;
  }
  return appendUiFence(result, ref) as string;
}

export function withObjectsUi(result: string): string {
  return append(result, OBJECTS_UI_REF);
}
export function withRecordsUi(result: string): string {
  return append(result, RECORDS_UI_REF);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/business_objects/__tests__/businessObjectsUi.test.ts
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add agent/src/agents/platform_service/business_objects/businessObjectsUi.ts agent/src/agents/platform_service/business_objects/__tests__/businessObjectsUi.test.ts
git commit -m "feat(agent): business-objects MCP App UI modules"
```

---

## Task 4: Wire the business-objects plugin

**Files:**
- Modify: `agent/src/agents/platform_service/business_objects/plugin.ts`
- Modify: `agent/src/agents/platform_service/__tests__/registration.test.ts`

- [ ] **Step 1: Add failing meta-invariant assertions**

Inside the `business-objects` describe block in `registration.test.ts` (or a new block), add:

```ts
  it("links the object/record ui resources to their tools", () => {
    const byName = new Map(businessObjectPlugin.meta.tools?.map((t) => [t.name, t]));
    expect(byName.get("list_objects")?.ui?.resource).toBe("ui://business-objects/objects");
    expect(byName.get("query_records")?.ui?.resource).toBe("ui://business-objects/records");
    expect(Object.keys(businessObjectPlugin.meta.uiResources ?? {}).sort()).toEqual([
      "ui://business-objects/objects",
      "ui://business-objects/records",
    ]);
  });
```

- [ ] **Step 2: Run to verify it fails**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/__tests__/registration.test.ts
```
Expected: FAIL — `uiResources` missing / no `ui` on tools.

- [ ] **Step 3: Update `plugin.ts`**

In `agent/src/agents/platform_service/business_objects/plugin.ts`:

1. Add the import:
```ts
import {
  OBJECTS_UI_RESOURCE,
  OBJECTS_APP_HTML,
  RECORDS_UI_RESOURCE,
  RECORDS_APP_HTML,
  withObjectsUi,
  withRecordsUi,
} from "./businessObjectsUi";
```

2. In `meta.tools`, add `ui` to the two entries:
```ts
      {
        name: "list_objects",
        description: "List Business Object definitions visible to the configured BO store key.",
        ui: { resource: OBJECTS_UI_RESOURCE, displayMode: "inline" },
      },
```
```ts
      {
        name: "query_records",
        description: "Query Business Object records by objectKey.",
        ui: { resource: RECORDS_UI_RESOURCE, displayMode: "inline" },
      },
```

3. In `meta`, add:
```ts
    uiResources: {
      [OBJECTS_UI_RESOURCE]: { html: OBJECTS_APP_HTML },
      [RECORDS_UI_RESOURCE]: { html: RECORDS_APP_HTML },
    },
```

4. In `middleware`, wrap the two tool executors. Replace:
```ts
        tool((input: z.infer<typeof schemas.empty>, exeConfig) => boObjectList(input, exeConfig, pluginConfig), {
          name: "list_objects",
          description: "List Business Object definitions visible to the configured BO store key.",
          schema: schemas.empty,
        }),
```
with:
```ts
        tool(async (input: z.infer<typeof schemas.empty>, exeConfig) => withObjectsUi(await boObjectList(input, exeConfig, pluginConfig)), {
          name: "list_objects",
          description: "List Business Object definitions visible to the configured BO store key.",
          schema: schemas.empty,
        }),
```
And replace:
```ts
        tool(
          (input: z.infer<typeof schemas.recordQuery>, exeConfig) =>
            boRecordQuery(input, exeConfig, pluginConfig),
          {
            name: "query_records",
            description:
              "Query Business Object records by objectKey. The object definition resolves the store; do not pass storeKey.",
            schema: schemas.recordQuery,
          },
        ),
```
with:
```ts
        tool(
          async (input: z.infer<typeof schemas.recordQuery>, exeConfig) =>
            withRecordsUi(await boRecordQuery(input, exeConfig, pluginConfig)),
          {
            name: "query_records",
            description:
              "Query Business Object records by objectKey. The object definition resolves the store; do not pass storeKey.",
            schema: schemas.recordQuery,
          },
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run (in `agent/`):
```bash
pnpm test -- src/agents/platform_service/__tests__/registration.test.ts src/agents/platform_service/__tests__/barrel.test.ts src/agents/platform_service/__tests__/business_objects.test.ts
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/src/agents/platform_service/business_objects/plugin.ts agent/src/agents/platform_service/__tests__/registration.test.ts
git commit -m "feat(agent): render business-objects list/query as inline MCP App tables"
```

---

## Task 5: Semantic Metrics UI module

**Files:**
- Create: `agent/src/agents/semantic_metrics/semanticMetricsUi.ts`
- Test: `agent/src/agents/semantic_metrics/__tests__/semanticMetricsUi.test.ts`

- [ ] **Step 1: Write the failing test**

Create `agent/src/agents/semantic_metrics/__tests__/semanticMetricsUi.test.ts`:

```ts
jest.mock("@axiom-lattice/core", () => ({
  appendUiFence: (content: unknown, ref: unknown) =>
    typeof content === "string"
      ? content + "\n\n```mcp_app\n" + JSON.stringify(ref) + "\n```"
      : content,
}));

import { describe, expect, it } from "@jest/globals";
import { META_UI_RESOURCE, META_APP_HTML, withMetaUi } from "../semanticMetricsUi";

describe("semanticMetricsUi", () => {
  it("declares the meta ui resource", () => {
    expect(META_UI_RESOURCE).toBe("ui://semantic-metrics/meta");
  });

  it("html carries the handshake and config", () => {
    expect(META_APP_HTML).toContain("ui/initialize");
    expect(META_APP_HTML).toContain("ui/notifications/tool-result");
    expect(META_APP_HTML).toContain("ui/notifications/size-changed");
    expect(META_APP_HTML).toContain('"rowsPath": "items"');
    expect(META_APP_HTML).toContain("No entries");
  });

  it("appends the fence for successful payloads", () => {
    const out = withMetaUi(JSON.stringify({ items: [], total: 0 }));
    expect(out).toContain("```mcp_app");
    expect(out).toContain('"pluginType":"semantic-metrics"');
    expect(out).toContain('"resource":"ui://semantic-metrics/meta"');
  });

  it("returns error payloads and non-json unchanged", () => {
    expect(withMetaUi("Error: boom")).toBe("Error: boom");
    const err = JSON.stringify({ ok: false, code: "X" });
    expect(withMetaUi(err)).toBe(err);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (in `agent/`):
```bash
pnpm test -- src/agents/semantic_metrics/__tests__/semanticMetricsUi.test.ts
```
Expected: FAIL — cannot find module `../semanticMetricsUi`.

- [ ] **Step 3: Implement `semanticMetricsUi.ts`**

Create `agent/src/agents/semantic_metrics/semanticMetricsUi.ts` (same shape as `storageUi.ts`), built from **Appendix A** with title `Semantic Meta` and CONFIG:

```json
{
  "title": "Semantic Meta",
  "rowsPath": "items",
  "summary": ["total"],
  "emptyText": "No entries",
  "dynamic": true,
  "columns": []
}
```

Constants and helper:
```ts
import { appendUiFence } from "@axiom-lattice/core";
import type { McpUiRef } from "@axiom-lattice/protocols";

export const META_UI_RESOURCE = "ui://semantic-metrics/meta";

export const META_UI_REF: McpUiRef = {
  kind: "plugin",
  pluginType: "semantic-metrics",
  resource: META_UI_RESOURCE,
  displayMode: "inline",
};

export const META_APP_HTML = `... Appendix A with title "Semantic Meta" + CONFIG ...`;

export function withMetaUi(result: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(result);
  } catch {
    return result;
  }
  if (parsed && typeof parsed === "object" && (parsed as { ok?: unknown }).ok === false) {
    return result;
  }
  return appendUiFence(result, META_UI_REF) as string;
}
```

Note: for `read_table_meta` / `read_metric_meta` the payload is a single object; because `rowsPath` is set to `items`, `pickRows` returns `undefined` and the App renders the detail (key/value) view automatically.

- [ ] **Step 4: Run the test to verify it passes**

Run (in `agent/`):
```bash
pnpm test -- src/agents/semantic_metrics/__tests__/semanticMetricsUi.test.ts
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add agent/src/agents/semantic_metrics/semanticMetricsUi.ts agent/src/agents/semantic_metrics/__tests__/semanticMetricsUi.test.ts
git commit -m "feat(agent): semantic-metrics meta MCP App UI module"
```

---

## Task 6: Wire `metrics_meta_tool` and the semantic-metrics plugin

**Files:**
- Modify: `agent/src/agents/semantic_metrics/tools/metrics_meta_tool.ts`
- Modify: `agent/src/agents/semantic_metrics/plugin.ts`
- Modify: `agent/src/agents/semantic_metrics/__tests__/coreMock.ts`
- Modify: `agent/src/agents/semantic_metrics/__tests__/plugin.test.ts`

- [ ] **Step 1: Add `appendUiFence` to `coreMock.ts`**

In `agent/src/agents/semantic_metrics/__tests__/coreMock.ts`, add to the returned object:

```ts
    appendUiFence: (content: unknown, ref: unknown) =>
      typeof content === "string"
        ? content + "\n\n```mcp_app\n" + JSON.stringify(ref) + "\n```"
        : content,
```

- [ ] **Step 2: Add failing assertions**

In `agent/src/agents/semantic_metrics/__tests__/plugin.test.ts`, inside `describe("semanticMetricsPlugin", ...)`, add:

```ts
  it("declares the meta ui resource linked to metrics_meta_tool", () => {
    const metaTool = semanticMetricsPlugin.meta.tools?.find((t) => t.name === "metrics_meta_tool");
    expect(metaTool?.ui?.resource).toBe("ui://semantic-metrics/meta");
    expect(Object.keys(semanticMetricsPlugin.meta.uiResources ?? {})).toEqual(["ui://semantic-metrics/meta"]);
  });
```

In `agent/src/agents/semantic_metrics/__tests__/metrics_meta_tool.test.ts`, add a test that a read action carries the fence and a write action does not:

```ts
  it("attaches the MCP App fence to read actions only", async () => {
    jest.spyOn(SemanticMetricsV2Client.prototype, "listTables").mockResolvedValue([]);
    const tool = createMetricsMetaTool(toolParams());
    const read = await tool.invoke({ action: "list_tables", connectionKey: "primary", datasourceId: "15" }, runtimeConfig());
    expect(read).toContain("```mcp_app");
    jest.spyOn(SemanticMetricsV2Client.prototype, "createTable").mockResolvedValue({ ok: true });
    const write = await tool.invoke(
      { action: "create_table", connectionKey: "primary", datasourceId: "15", payload: { objectKey: "x" } },
      runtimeConfig(),
    );
    expect(write).not.toContain("```mcp_app");
  });
```

- [ ] **Step 3: Run to verify failure**

Run (in `agent/`):
```bash
pnpm test -- src/agents/semantic_metrics/__tests__/plugin.test.ts src/agents/semantic_metrics/__tests__/metrics_meta_tool.test.ts
```
Expected: FAIL — `uiResources` missing; read action lacks the fence.

- [ ] **Step 4: Update `metrics_meta_tool.ts`**

At the top of `agent/src/agents/semantic_metrics/tools/metrics_meta_tool.ts`, add:
```ts
import { withMetaUi } from "../semanticMetricsUi";
```

Then wrap only the four read actions in the switch:

```ts
          case "list_tables": {
            return withMetaUi(JSON.stringify(await client.listTables(ds), null, 2));
          }
          case "read_table_meta": {
            if (!input.objectKey) throw new Error("objectKey is required for read_table_meta");
            return withMetaUi(JSON.stringify(await client.getTable(ds, input.objectKey), null, 2));
          }
```
```ts
          case "list_metrics": {
            return withMetaUi(JSON.stringify(await client.listMetrics(ds), null, 2));
          }
          case "read_metric_meta": {
            if (!input.objectKey) throw new Error("objectKey is required for read_metric_meta");
            return withMetaUi(JSON.stringify(await client.getMetric(ds, input.objectKey), null, 2));
          }
```

Leave `create_table` / `update_table` / `create_metric` / `update_metric` unchanged.

- [ ] **Step 5: Update `plugin.ts`**

In `agent/src/agents/semantic_metrics/plugin.ts`:

1. Add the import:
```ts
import { META_UI_RESOURCE, META_APP_HTML } from "./semanticMetricsUi";
```

2. In `meta.tools`, add `ui` to the `metrics_meta_tool` entry:
```ts
      {
        name: "metrics_meta_tool",
        description: "Publish and maintain runtime semantic tables and metrics",
        ui: { resource: META_UI_RESOURCE, displayMode: "inline" },
      },
```

3. In `meta`, add:
```ts
    uiResources: {
      [META_UI_RESOURCE]: { html: META_APP_HTML },
    },
```

- [ ] **Step 6: Run the full agent test suite**

Run (in `agent/`):
```bash
pnpm test
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add agent/src/agents/semantic_metrics/tools/metrics_meta_tool.ts agent/src/agents/semantic_metrics/plugin.ts agent/src/agents/semantic_metrics/__tests__/coreMock.ts agent/src/agents/semantic_metrics/__tests__/plugin.test.ts agent/src/agents/semantic_metrics/__tests__/metrics_meta_tool.test.ts
git commit -m "feat(agent): render metrics_meta_tool read actions as inline MCP App"
```

---

## Task 7: Build and manual verification

**Files:** none (verification only)

- [ ] **Step 1: Build the agent**

Run (in `agent/`):
```bash
pnpm build
```
Expected: build succeeds, no TypeScript errors.

- [ ] **Step 2: Build the web console**

Run (in `ai_web/`):
```bash
pnpm build
```
Expected: build succeeds.

- [ ] **Step 3: Manual smoke test**

Start the agent and open the chat console. Trigger each tool and confirm the inline table renders:
- `list_objects` → objects table
- `query_records` → records table (dynamic columns, `id` first)
- `list` (storage) → files table with summary line
- `metrics_meta_tool` with `action: "list_tables"` → table; with `action: "read_table_meta"` → detail card

Confirm:
- Styles are identical across all four (same header/row/format conventions).
- Empty results and error results render no table (error text only).
- Height auto-fits; light/dark both legible.
- The model still receives the original JSON (fence is display-only).

- [ ] **Step 4: Commit (only if verification produced changes)**

If no changes were needed, nothing to commit. If a fix was required, commit it with a descriptive message.

---

## Self-Review

**Spec coverage:**
- §4 mechanism (ui + uiResources + manual fence) → Tasks 1-6.
- §5 self-contained `*Ui.ts` contract + canonical template → Appendix A + Tasks 1/3/5.
- §6.1 business-objects (2 resources, columns) → Tasks 3-4.
- §6.2 storage (meta.tools completion, files columns) → Tasks 1-2.
- §6.3 semantic-metrics (1 resource, list/detail, read-only fence) → Tasks 5-6.
- §7 dependency upgrade → Task 0.
- §8 testing → test steps in every task.
- §9 verification → Task 7.

**Placeholder scan:** No "TBD"/"TODO". The only indirection is "paste Appendix A with title/CONFIG substituted", which is fully specified by the two substitutions given in each task; the full HTML is present in Appendix A.

**Type consistency:** `withFilesUi` / `withObjectsUi` / `withRecordsUi` / `withMetaUi` all take `string` and return `string`. Resource constants and `McpUiRef.pluginType` match each plugin's `meta.type` (`storage`, `business-objects`, `semantic-metrics`). `appendUiFence` is mocked in every test file that imports a `*Ui.ts`.
