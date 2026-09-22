import { describe, expect, it, vi } from "vitest";
import { createBoTools } from "../src/upstream/boTools";
import type { McpCaller } from "../src/upstream/mcp";

function mcpReturning(payload: unknown): McpCaller {
  return { callTool: vi.fn(async () => ({ text: JSON.stringify(payload), isError: false })) };
}

describe("boTools", () => {
  it("getRecord merges data + id and calls business-objects_get_record", async () => {
    const mcp = mcpReturning({
      id: "t1",
      objectKey: "tag_definition",
      data: { tag_id: "t1", tag_group: "g", tag_name: "n" },
    });
    const bo = createBoTools(mcp);
    await expect(bo.getRecord("tag_definition", "t1")).resolves.toEqual({
      tag_id: "t1",
      tag_group: "g",
      tag_name: "n",
      id: "t1",
    });
    expect(mcp.callTool).toHaveBeenCalledWith("business-objects_get_record", {
      objectKey: "tag_definition",
      id: "t1",
    });
  });

  it("getRecord returns undefined for a not-found payload", async () => {
    const bo = createBoTools(mcpReturning({ success: false, error: "record not found" }));
    await expect(bo.getRecord("tag_definition", "nope")).resolves.toBeUndefined();
  });

  it("queryRecords returns rows and forwards filters", async () => {
    const mcp = mcpReturning({ objectKey: "customer_tag", total: 1, rows: [{ id: "c1", tag_id: "t1" }] });
    const bo = createBoTools(mcp);
    await expect(
      bo.queryRecords("customer_tag", [{ field: "customer_no", op: "eq", value: "cus_1" }]),
    ).resolves.toEqual([{ id: "c1", tag_id: "t1" }]);
    expect(mcp.callTool).toHaveBeenCalledWith("business-objects_query_records", {
      objectKey: "customer_tag",
      filters: [{ field: "customer_no", op: "eq", value: "cus_1" }],
    });
  });

  it("createRecord merges data + id", async () => {
    const mcp = mcpReturning({ id: "new1", objectKey: "customer_tag", data: { tag_id: "t1" } });
    const bo = createBoTools(mcp);
    await expect(bo.createRecord("customer_tag", { customer_no: "c" })).resolves.toEqual({
      tag_id: "t1",
      id: "new1",
    });
  });

  it("updateRecord calls business-objects_update_record with objectKey/id/data", async () => {
    const mcp = mcpReturning({ id: "c1", objectKey: "customer_tag", data: { tag_key: "g2" } });
    const bo = createBoTools(mcp);
    await expect(bo.updateRecord("customer_tag", "c1", { tag_key: "g2" })).resolves.toEqual({
      tag_key: "g2",
      id: "c1",
    });
    expect(mcp.callTool).toHaveBeenCalledWith("business-objects_update_record", {
      objectKey: "customer_tag",
      id: "c1",
      data: { tag_key: "g2" },
    });
  });

  it("deleteRecords passes confirm:true and returns the count", async () => {
    const mcp = mcpReturning({ objectKey: "customer_tag", deleted: 2, ids: ["a", "b"] });
    const bo = createBoTools(mcp);
    await expect(bo.deleteRecords("customer_tag", ["a", "b"])).resolves.toBe(2);
    expect(mcp.callTool).toHaveBeenCalledWith("business-objects_delete_records", {
      objectKey: "customer_tag",
      ids: ["a", "b"],
      confirm: true,
    });
  });

  it("deleteRecords short-circuits on an empty id list", async () => {
    const mcp = mcpReturning({ deleted: 0 });
    const bo = createBoTools(mcp);
    await expect(bo.deleteRecords("customer_tag", [])).resolves.toBe(0);
    expect(mcp.callTool).not.toHaveBeenCalled();
  });

  it("queryRecords maps an error payload to 502", async () => {
    const bo = createBoTools(mcpReturning({ success: false, error: "boom" }));
    await expect(bo.queryRecords("customer_tag", [])).rejects.toMatchObject({ statusCode: 502 });
  });
});
