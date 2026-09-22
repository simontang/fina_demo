import { GatewayError } from "../lib/errors";
import type { McpCaller } from "./mcp";

export type BoRecord = Record<string, unknown>;

export type BoFilter = { field: string; op?: string; value: unknown };

export type BoTools = {
  getRecord(objectKey: string, id: string): Promise<BoRecord | undefined>;
  queryRecords(objectKey: string, filters: BoFilter[]): Promise<BoRecord[]>;
  createRecord(objectKey: string, data: BoRecord): Promise<BoRecord>;
  updateRecord(objectKey: string, id: string, data: BoRecord): Promise<BoRecord>;
  deleteRecords(objectKey: string, ids: string[]): Promise<number>;
};

function parse(text: string, structured: unknown): any {
  if (structured && typeof structured === "object") return structured;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

export function createBoTools(mcp: McpCaller): BoTools {
  async function call(tool: string, args: Record<string, unknown>): Promise<any> {
    const { text, structured } = await mcp.callTool(`business-objects_${tool}`, args);
    return parse(text, structured);
  }

  function fail(data: any, tool: string): never {
    throw new GatewayError(502, "UPSTREAM_ERROR", String(data?.error ?? `${tool} failed`));
  }

  return {
    async getRecord(objectKey, id) {
      let data: any;
      try {
        data = await call("get_record", { objectKey, id });
      } catch (err) {
        if (err instanceof GatewayError && /not found/i.test(err.message)) return undefined;
        throw err;
      }
      if (!data || data.success === false || data.error) return undefined;
      const record = data.record ?? data;
      if (!record || typeof record !== "object" || typeof record.id !== "string") return undefined;
      return { ...(record.data ?? {}), id: record.id };
    },

    async queryRecords(objectKey, filters) {
      const data = await call("query_records", { objectKey, filters });
      if (!data || data.success === false || data.error) fail(data, "query_records");
      const rows = data.rows ?? data.data?.rows ?? [];
      return Array.isArray(rows) ? rows : [];
    },

    async createRecord(objectKey, input) {
      const data = await call("create_record", { objectKey, data: input });
      if (!data || data.success === false || data.error) fail(data, "create_record");
      const record = data.record ?? data;
      return { ...(record.data ?? {}), id: record.id };
    },

    async updateRecord(objectKey, id, input) {
      const data = await call("update_record", { objectKey, id, data: input });
      if (!data || data.success === false || data.error) fail(data, "update_record");
      const record = data.record ?? data;
      return { ...(record.data ?? {}), id: record.id ?? id };
    },

    async deleteRecords(objectKey, ids) {
      if (ids.length === 0) return 0;
      const data = await call("delete_records", { objectKey, ids, confirm: true });
      if (!data || data.success === false || data.error) fail(data, "delete_records");
      return typeof data.deleted === "number" ? data.deleted : 0;
    },
  };
}
