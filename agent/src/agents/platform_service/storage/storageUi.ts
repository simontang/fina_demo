import { appendUiFence } from "@axiom-lattice/core";
import type { McpUiRef } from "@axiom-lattice/protocols";

export const FILES_UI_RESOURCE = "ui://storage/files";

export const FILES_UI_REF: McpUiRef = {
  kind: "plugin",
  pluginType: "storage",
  resource: FILES_UI_RESOURCE,
  displayMode: "inline",
};

export const FILES_APP_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Files</title>
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
        var CONFIG = { "title": "Files", "rowsPath": "files", "summary": ["path", "total", "page", "size", "totalPages"], "emptyText": "No files", "columns": [ { "key": "filename", "label": "Name" }, { "key": "fullPath", "label": "Path", "format": "mono" }, { "key": "size", "label": "Size", "format": "bytes" }, { "key": "mime", "label": "MIME" }, { "key": "fileCategory", "label": "Category" }, { "key": "usage", "label": "Usage" }, { "key": "version", "label": "Ver", "format": "number" }, { "key": "status", "label": "Status", "format": "status" }, { "key": "createdAt", "label": "Created", "format": "datetime" }, { "key": "uuid", "label": "UUID", "format": "mono" } ] };
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
        window.addEventListener("message", function (event) {
          var data = event.data;
          if (!data || typeof data !== "object" || typeof data.method !== "string") return;
          if (data.method === "ui/notifications/tool-result") render(readResult(data.params));
        });
        if (typeof ResizeObserver !== "undefined") new ResizeObserver(sendSize).observe(document.documentElement);
        else window.addEventListener("resize", sendSize);
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
</html>`;

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
