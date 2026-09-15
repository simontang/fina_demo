import { describe, expect, it } from "vitest";
import { GatewayError, toErrorResponse } from "../src/lib/errors";
import { fetchWithTimeout } from "../src/lib/http";

describe("toErrorResponse", () => {
  it("maps GatewayError", () => {
    const res = toErrorResponse(new GatewayError(404, "NOT_FOUND", "nope", { taskId: "t1" }));
    expect(res.statusCode).toBe(404);
    expect(res.body).toEqual({ code: "NOT_FOUND", message: "nope", upstream: { taskId: "t1" } });
  });

  it("maps unknown errors to 500", () => {
    const res = toErrorResponse(new Error("boom"));
    expect(res.statusCode).toBe(500);
    expect(res.body.code).toBe("INTERNAL_ERROR");
  });

  it("maps multipart file-too-large to 413", () => {
    const res = toErrorResponse(Object.assign(new Error("too large"), { code: "FST_REQ_FILE_TOO_LARGE" }));
    expect(res.statusCode).toBe(413);
    expect(res.body.code).toBe("PAYLOAD_TOO_LARGE");
  });
});

describe("fetchWithTimeout", () => {
  it("returns the response on success", async () => {
    const fetchImpl = (async () => new Response("{}", { status: 200 })) as typeof fetch;
    const res = await fetchWithTimeout(fetchImpl, "http://x", { method: "GET" }, 1000);
    expect(res.status).toBe(200);
  });

  it("maps AbortError to 504", async () => {
    const fetchImpl = (async () => {
      const e = new Error("aborted");
      e.name = "AbortError";
      throw e;
    }) as typeof fetch;
    await expect(fetchWithTimeout(fetchImpl, "http://x", {}, 10)).rejects.toMatchObject({
      statusCode: 504,
      code: "UPSTREAM_TIMEOUT",
    });
  });
});
