jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (o: unknown) => o,
  tool: (fn: unknown, cfg: Record<string, unknown>) => ({ ...cfg, invoke: fn }),
}));

import { PluginRegistry } from "@axiom-lattice/core";
import "../index";

describe("platform_service barrel", () => {
  it("registers both plugins when the barrel is imported", () => {
    const types = (PluginRegistry.register as jest.Mock).mock.calls
      .map(([p]) => (p as { meta: { type: string } }).meta.type)
      .sort();
    expect(types).toEqual(["storage", "webhooks"]);
  });
});
