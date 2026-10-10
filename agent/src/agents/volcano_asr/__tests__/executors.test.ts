jest.mock("@axiom-lattice/core", () => ({
  resolvePluginConnections: jest.fn(),
}));

import { resolvePluginConnections } from "@axiom-lattice/core";
import {
  getTranscription,
  recognize,
  recognizeFlash,
  submitTranscription,
} from "../executors";
import { mockResponse } from "./mockResponse";

const RESOLVED = {
  _resolvedConnections: [
    { config: { apiKey: "test-key", baseUrl: "https://openspeech.bytedance.com" } },
  ],
};

beforeAll(() => {
  delete process.env.VOLC_ASR_API_KEY;
});

describe("volcano-asr executors", () => {
  afterEach(() => {
    jest.restoreAllMocks();
    (resolvePluginConnections as jest.Mock).mockReset();
  });

  it("submit_transcription posts to submit with sequence -1 and returns requestId", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000000", message: "ok", requestId: "task-1" }));

    const out = JSON.parse(await submitTranscription({ audioUrl: "https://a.mp3" }, {}, RESOLVED));

    expect(out.requestId).toBe("task-1");
    expect(out.audioUrl).toBe("https://a.mp3");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Api-Key"]).toBe("test-key");
    expect(headers["X-Api-Sequence"]).toBe("-1");
    const body = JSON.parse(init.body as string);
    expect(body.audio.url).toBe("https://a.mp3");
    expect(body.request.enable_itn).toBe(true);
  });

  it("returns NO_API_KEY when the connection has no apiKey", async () => {
    const out = JSON.parse(
      await submitTranscription(
        { audioUrl: "https://a.mp3" },
        {},
        { _resolvedConnections: [{ config: { baseUrl: "https://x" } }] },
      ),
    );
    expect(out.ok).toBe(false);
    expect(out.code).toBe("NO_API_KEY");
  });

  it("returns SUBMIT_FAILED on a non-success status code", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "45000010", message: "bad key" }));

    const out = JSON.parse(await submitTranscription({ audioUrl: "https://a.mp3" }, {}, RESOLVED));

    expect(out.ok).toBe(false);
    expect(out.code).toBe("SUBMIT_FAILED");
    expect(out.statusCode).toBe("45000010");
  });

  it("get_transcription polls a processing task until it completes", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(mockResponse({ statusCode: "20000001", message: "processing" }))
      .mockResolvedValueOnce(
        mockResponse({ statusCode: "20000000", body: { result: { text: "final text" } } }),
      );

    const out = JSON.parse(
      await getTranscription({ requestId: "task-1", intervalSeconds: 0 }, {}, RESOLVED),
    );

    expect(out.status).toBe("completed");
    expect(out.text).toBe("final text");
    expect(out.attempts).toBe(2);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("get_transcription returns timeout when the task never completes", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000001", message: "processing" }));

    const out = JSON.parse(
      await getTranscription(
        { requestId: "task-1", timeoutSeconds: 0.05, intervalSeconds: 0 },
        {},
        RESOLVED,
      ),
    );

    expect(out.status).toBe("timeout");
    expect(out.hint).toContain("get_transcription");
  });

  it("get_transcription returns QUERY_FAILED on a terminal error status", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "45000001", message: "not found" }));

    const out = JSON.parse(
      await getTranscription({ requestId: "task-1", intervalSeconds: 0 }, {}, RESOLVED),
    );

    expect(out.ok).toBe(false);
    expect(out.code).toBe("QUERY_FAILED");
  });

  it("recognize submits and then polls to completion", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(mockResponse({ statusCode: "20000000", requestId: "task-9" }))
      .mockResolvedValueOnce(
        mockResponse({ statusCode: "20000000", body: { result: { text: "done" } } }),
      );

    const out = JSON.parse(
      await recognize({ audioUrl: "https://a.mp3", intervalSeconds: 0 }, {}, RESOLVED),
    );

    expect(out.status).toBe("completed");
    expect(out.requestId).toBe("task-9");
    expect(out.text).toBe("done");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("recognize_flash requires format", async () => {
    const out = JSON.parse(await recognizeFlash({ audioUrl: "https://a.mp3" }, {}, RESOLVED));
    expect(out.ok).toBe(false);
    expect(out.code).toBe("BAD_REQUEST");
  });

  it("recognize_flash uses the turbo resource id and returns text", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(
        mockResponse({ statusCode: "20000000", body: { result: { text: "flash text" } } }),
      );

    const out = JSON.parse(
      await recognizeFlash({ audioUrl: "https://a.mp3", format: "mp3" }, {}, RESOLVED),
    );

    expect(out.mode).toBe("flash");
    expect(out.text).toBe("flash text");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash");
    expect((init.headers as Record<string, string>)["X-Api-Resource-Id"]).toBe(
      "volc.bigasr.auc_turbo",
    );
  });

  it("resolves the connection through the selector with plugin type volcano-asr", async () => {
    (resolvePluginConnections as jest.Mock).mockResolvedValue([
      { key: "c1", config: { apiKey: "k", baseUrl: "https://y" } },
    ]);
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000000", requestId: "t" }));

    await submitTranscription(
      { audioUrl: "https://a.mp3" },
      { configurable: { runConfig: { tenantId: "t1" } } },
      { connections: ["c1"], connectAll: false },
    );

    expect(resolvePluginConnections).toHaveBeenCalledWith(
      "volcano-asr",
      { connections: ["c1"], connectAll: false },
      { tenantId: "t1" },
    );
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toBe("https://y/api/v3/auc/bigmodel/submit");
  });
});
