jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  resolvePluginConnections: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (o: unknown) => o,
  tool: (fn: unknown, cfg: Record<string, unknown>) => ({ ...cfg, invoke: fn }),
}));

import { PluginRegistry } from "@axiom-lattice/core";
import { volcanoAsrPlugin } from "../plugin";
import { mockResponse } from "./mockResponse";

beforeAll(() => {
  delete process.env.VOLC_ASR_API_KEY;
});

describe("volcano-asr plugin", () => {
  afterEach(() => jest.restoreAllMocks());

  it("registers a plugin with type volcano-asr", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(volcanoAsrPlugin);
    expect(volcanoAsrPlugin.meta.type).toBe("volcano-asr");
  });

  it("declares exactly the four ASR tools", async () => {
    const mw = (await volcanoAsrPlugin.middleware!({})) as {
      tools: Array<{ name: string }>;
    };
    expect(mw.tools.map((t) => t.name).sort()).toEqual([
      "get_transcription",
      "recognize",
      "recognize_flash",
      "submit_transcription",
    ]);
    expect(volcanoAsrPlugin.meta.tools?.map((t) => t.name).sort()).toEqual([
      "get_transcription",
      "recognize",
      "recognize_flash",
      "submit_transcription",
    ]);
  });

  it("openExpose names all exist among middleware tools, get_transcription readOnly", async () => {
    const mw = (await volcanoAsrPlugin.middleware!({})) as {
      tools: Array<{ name: string }>;
    };
    const toolNames = mw.tools.map((t) => t.name);
    const expose = (volcanoAsrPlugin.meta.openExpose ?? []).map((e) =>
      typeof e === "string" ? { name: e, readOnly: false } : e,
    );
    for (const e of expose) expect(toolNames).toContain(e.name);
    expect(expose.find((e) => e.name === "get_transcription")?.readOnly).toBe(true);
  });

  it("connection.test reports success for a valid key", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000000", message: "ok" }));

    const result = await volcanoAsrPlugin.connection!.test!({
      apiKey: "k",
      baseUrl: "https://openspeech.bytedance.com",
    });

    expect(result.ok).toBe(true);
    expect(result.message).toContain("连接正常");
    expect(result.details).toBeDefined();
  });

  it("connection.test reports auth failure and masks the key", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "45000010", message: "Invalid X-Api-Key" }));

    const result = await volcanoAsrPlugin.connection!.test!({
      apiKey: "secret-key-123456",
      baseUrl: "https://openspeech.bytedance.com",
    });

    expect(result.ok).toBe(false);
    expect(result.message).toContain("鉴权失败");
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["X-Api-Key"]).toBe("secret-key-123456");
    expect(JSON.stringify(result.details)).not.toContain("secret-key-123456");
  });

  it("connection.test reports a missing api key", async () => {
    const result = await volcanoAsrPlugin.connection!.test!({ baseUrl: "https://x" });
    expect(result.ok).toBe(false);
    expect(result.message).toContain("API Key");
  });
});
