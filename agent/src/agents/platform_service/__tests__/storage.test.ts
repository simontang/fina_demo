jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
}));

import { getSandBoxManager } from "@axiom-lattice/core";
import { storageUpload } from "../storage/executors";

const downloadFile = jest.fn();
(getSandBoxManager as jest.Mock).mockReturnValue({
  getSandboxFromConfig: jest.fn(async () => ({ file: { downloadFile } })),
});

const rawConfig = {
  _resolvedConnections: [{ config: { baseUrl: "http://svc:5707", apiKey: "k" } }],
};
const exeConfig = {
  configurable: {
    runConfig: {
      tenantId: "t1",
      assistant_id: "a1",
      thread_id: "th1",
      projectId: "p1",
      workspaceId: "w1",
    },
  },
};

describe("storageUpload", () => {
  beforeEach(() => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ uuid: "u1", fullPath: "reports/r.csv" }),
    } as Response);
  });
  afterEach(() => jest.restoreAllMocks());

  it("reads bytes from the sandbox and PUTs them with metadata headers", async () => {
    downloadFile.mockResolvedValue(Buffer.from("a,b\n1,2\n"));
    const out = await storageUpload(
      { sandboxPath: "/project/out/r.csv", logicalPath: "reports/2026-09" },
      exeConfig,
      rawConfig,
    );
    expect(JSON.parse(out)).toMatchObject({ uuid: "u1" });

    expect(downloadFile).toHaveBeenCalledWith({ file: "/project/out/r.csv" });
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(String(url)).toMatch(/^http:\/\/svc:5707\/api\/v1\/files\/[0-9a-f]{32}$/);
    expect(init.method).toBe("PUT");
    expect(init.headers["X-Tenant-Id"]).toBe("t1");
    expect(init.headers["X-File-Name"]).toBe("r.csv");
    expect(init.headers["X-File-Path"]).toBe("reports/2026-09");
  });

  it("rejects files over the size cap without uploading", async () => {
    downloadFile.mockResolvedValue(Buffer.alloc(1024));
    process.env.STORAGE_TOOL_MAX_UPLOAD_BYTES = "10";
    const out = await storageUpload({ sandboxPath: "/project/big.bin" }, exeConfig, rawConfig);
    expect(JSON.parse(out).ok).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
    delete process.env.STORAGE_TOOL_MAX_UPLOAD_BYTES;
  });

  it("returns a structured error when the sandbox is missing", async () => {
    const out = await storageUpload(
      { sandboxPath: "/x" },
      { configurable: { runConfig: {} } },
      rawConfig,
    );
    expect(JSON.parse(out).ok).toBe(false);
  });
});
