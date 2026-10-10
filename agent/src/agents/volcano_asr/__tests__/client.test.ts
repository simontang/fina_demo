import { callApi, extractText, isProcessing, isSuccess } from "../client";
import { mockResponse } from "./mockResponse";

describe("volcano-asr client", () => {
  afterEach(() => jest.restoreAllMocks());

  it("classifies success and processing status codes", () => {
    expect(isSuccess("20000000")).toBe(true);
    expect(isSuccess("20000001")).toBe(false);
    expect(isProcessing("20000001")).toBe(true);
    expect(isProcessing("20000002")).toBe(true);
    expect(isProcessing("20000000")).toBe(false);
  });

  it("extracts text from flat, nested, and utterance payloads", () => {
    expect(extractText({ result: { text: "nested" } })).toBe("nested");
    expect(extractText({ text: "flat" })).toBe("flat");
    expect(extractText({ result: { utterances: [{ text: "a" }, { text: "b" }] } })).toBe("ab");
    expect(extractText(null)).toBe("");
  });

  it("sends auth headers and parses response headers", async () => {
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue(
      mockResponse({
        statusCode: "20000000",
        message: "OK",
        requestId: "rid-1",
        body: { result: { text: "hey" } },
      }),
    );

    const res = await callApi({
      baseUrl: "https://x",
      apiKey: "k",
      resourceId: "volc.seedasr.auc",
      path: "/submit",
      body: { a: 1 },
      requestId: "req-1",
      sequence: -1,
    });

    expect(res.statusCode).toBe("20000000");
    expect(res.requestId).toBe("rid-1");
    expect(res.body).toEqual({ result: { text: "hey" } });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://x/submit");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Api-Key"]).toBe("k");
    expect(headers["X-Api-Resource-Id"]).toBe("volc.seedasr.auc");
    expect(headers["X-Api-Request-Id"]).toBe("req-1");
    expect(headers["X-Api-Sequence"]).toBe("-1");
  });
});
