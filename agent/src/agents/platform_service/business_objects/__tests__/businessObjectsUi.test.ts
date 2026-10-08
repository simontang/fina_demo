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
