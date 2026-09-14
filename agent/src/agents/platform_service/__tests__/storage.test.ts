jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
}));

import { getSandBoxManager } from "@axiom-lattice/core";
import { storageUpload } from "../storage/executors";

const downloadFile = jest.fn();
const listPath = jest.fn();
const getSandboxFromConfig = jest.fn();
(getSandBoxManager as jest.Mock).mockReturnValue({ getSandboxFromConfig });

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
    downloadFile.mockReset();
    listPath.mockReset().mockResolvedValue({ files: [] });
    getSandboxFromConfig
      .mockReset()
      .mockResolvedValue({ file: { downloadFile, listPath } });
  });
  afterEach(() => {
    jest.restoreAllMocks();
    delete process.env.STORAGE_TOOL_MAX_UPLOAD_BYTES;
  });

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
    expect(init.headers["Content-Type"]).toBe("text/csv");
    expect(Buffer.from(init.body).toString()).toBe("a,b\n1,2\n");
  });

  it("rejects files over the size cap after download without uploading", async () => {
    downloadFile.mockResolvedValue(Buffer.alloc(1024));
    process.env.STORAGE_TOOL_MAX_UPLOAD_BYTES = "10";
    const out = await storageUpload({ sandboxPath: "/project/big.bin" }, exeConfig, rawConfig);
    expect(JSON.parse(out).ok).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("rejects over-cap files using the known size without downloading", async () => {
    process.env.STORAGE_TOOL_MAX_UPLOAD_BYTES = "10";
    listPath.mockResolvedValueOnce({
      files: [{ path: "/project/big.bin", size: 1024 }],
    });
    const out = await storageUpload({ sandboxPath: "/project/big.bin" }, exeConfig, rawConfig);
    expect(JSON.parse(out).ok).toBe(false);
    expect(downloadFile).not.toHaveBeenCalled();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("uploads a file exactly at the size cap", async () => {
    process.env.STORAGE_TOOL_MAX_UPLOAD_BYTES = "10";
    listPath.mockResolvedValueOnce({
      files: [{ path: "/project/edge.bin", size: 10 }],
    });
    downloadFile.mockResolvedValue(Buffer.alloc(10));
    const out = await storageUpload({ sandboxPath: "/project/edge.bin" }, exeConfig, rawConfig);
    expect(JSON.parse(out)).toMatchObject({ uuid: "u1" });
    expect(downloadFile).toHaveBeenCalledWith({ file: "/project/edge.bin" });
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it("returns a structured error when sandbox resolution fails", async () => {
    getSandboxFromConfig.mockRejectedValueOnce(new Error("no sandbox"));
    const out = await storageUpload({ sandboxPath: "/x" }, exeConfig, rawConfig);
    expect(JSON.parse(out).ok).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("returns a structured error when the tenant is missing", async () => {
    const out = await storageUpload(
      { sandboxPath: "/x" },
      { configurable: { runConfig: {} } },
      rawConfig,
    );
    expect(JSON.parse(out).ok).toBe(false);
  });
});
