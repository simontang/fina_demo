import { buildCdpTargetUrl } from "../cdpProxy";

describe("buildCdpTargetUrl", () => {
  const base = "http://127.0.0.1:5706";

  it("rewrites the segment definition collection path", () => {
    expect(buildCdpTargetUrl("/api/cdp/segment-definitions", base)).toBe(
      "http://127.0.0.1:5706/api/v1/segment-definitions"
    );
  });

  it("rewrites nested resource paths", () => {
    expect(buildCdpTargetUrl("/api/cdp/segment-definitions/7/process", base)).toBe(
      "http://127.0.0.1:5706/api/v1/segment-definitions/7/process"
    );
  });

  it("preserves query parameters", () => {
    expect(buildCdpTargetUrl("/api/cdp/segment-definitions?status=active&page=2", base)).toBe(
      "http://127.0.0.1:5706/api/v1/segment-definitions?status=active&page=2"
    );
  });

  it("rewrites the exact proxy root", () => {
    expect(buildCdpTargetUrl("/api/cdp", base)).toBe("http://127.0.0.1:5706/api/v1");
  });
});
