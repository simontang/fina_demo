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

  it("renders nested detail values as JSON", () => {
    expect(META_APP_HTML).toContain('class="json"');
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
