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
    expect(FILES_APP_HTML.indexOf("ui/notifications/tool-result")).toBeLessThan(
      FILES_APP_HTML.indexOf('rpc("ui/initialize"'),
    );
  });

  it("renders nested detail values as JSON", () => {
    expect(FILES_APP_HTML).toContain('class="json"');
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

  it("appends the fence for arrays, null, and ok:true payloads", () => {
    for (const body of ["[]", "null", JSON.stringify({ ok: true, files: [] })]) {
      expect(withFilesUi(body)).toContain("```mcp_app");
    }
  });
});
