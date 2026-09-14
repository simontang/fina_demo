jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (o: unknown) => o,
  tool: (fn: unknown, cfg: Record<string, unknown>) => ({ ...cfg, invoke: fn }),
}));

import { PluginRegistry } from "@axiom-lattice/core";
import { storagePlugin } from "../storage/plugin";

describe("storage plugin", () => {
  it("registers a plugin with type storage", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(storagePlugin);
    expect(storagePlugin.meta.type).toBe("storage");
  });

  it("exposes exactly the non-upload tools to Open, readOnly/destructive annotated", () => {
    const expose = storagePlugin.meta.openExpose as Array<
      string | { name: string; readOnly?: boolean; destructive?: boolean }
    >;
    const names = expose.map((e) => (typeof e === "string" ? e : e.name));
    expect(names.sort()).toEqual([
      "storage_delete",
      "storage_get_download_url",
      "storage_get_metadata",
      "storage_list",
    ]);
  });

  it("openExpose names all exist among middleware tools (invariant)", async () => {
    const mw = await storagePlugin.middleware!({});
    const toolNames = ((mw as { tools: Array<{ name: string }> }).tools ?? []).map((t) => t.name);
    const expose = storagePlugin.meta.openExpose as Array<string | { name: string }>;
    for (const e of expose) {
      const name = typeof e === "string" ? e : e.name;
      expect(toolNames).toContain(name);
    }
    expect(toolNames).toContain("storage_upload");
  });
});
