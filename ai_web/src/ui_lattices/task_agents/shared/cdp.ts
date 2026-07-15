export const CDP_API_BASE = "/api/cdp";

export interface CdpApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

export function unwrapCdpResponse<T>(response: CdpApiResponse<T>): T {
  if (response.code !== 200) {
    throw new Error(response.message || "CDP request failed");
  }
  return response.data;
}

export function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}
