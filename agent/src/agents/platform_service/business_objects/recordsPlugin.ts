import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
import { platformServiceConnection } from "../connection";
import {
  boConnectionKey,
  boRecordCreate,
  boRecordDelete,
  boRecordGet,
  boRecordQuery,
  boRecordUpdate,
} from "./executors";

const identifier = z.string().regex(/^[a-z][a-z0-9_]{0,62}$/);

const filter = z.object({
  field: identifier,
  op: z.enum(["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"]).optional(),
  value: z.unknown(),
});

const sort = z.object({
  field: identifier,
  direction: z.enum(["asc", "desc"]).optional(),
});

const schemas = {
  create: z.object({
    objectKey: identifier,
    data: z.record(z.unknown()),
  }),
  get: z.object({
    objectKey: identifier,
    id: z.string(),
  }),
  update: z.object({
    objectKey: identifier,
    id: z.string(),
    data: z.record(z.unknown()),
  }),
  delete: z.object({
    objectKey: identifier,
    id: z.string(),
    confirm: z.boolean().optional(),
  }),
  query: z.object({
    objectKey: identifier,
    filters: z.array(filter).optional(),
    sort: z.array(sort).optional(),
    page: z.number().int().optional(),
    pageSize: z.number().int().optional(),
  }),
};

export const businessObjectRecordsPlugin: Plugin = {
  meta: {
    type: "business-object-records",
    name: "Business Object Records",
    description:
      "CRUDQ runtime data API for Business Objects. objectKey resolves the backing store through platform-service definitions and grants; callers never pass storeKey.",
    version: "1.0.0",
    configSchema: {
      type: "object",
      properties: {
        connections: {
          type: "array",
          title: "Connections",
          widget: "connectionSelect",
          items: { type: "string" },
        },
        connectAll: { type: "boolean", title: "Connect all available connections" },
      },
    },
    defaultConfig: { connections: [], connectAll: false },
    openExpose: [
      { name: "query_records", readOnly: true },
      { name: "get_record", readOnly: true },
      { name: "create_record" },
      { name: "update_record" },
      { name: "delete_record", destructive: true },
    ],
  },
  connection: {
    ...platformServiceConnection,
    discover: async (config) => {
      const conn = connectionFromConfig(config);
      const rows = await request<Array<{ objectKey: string; displayName?: string; storeKey?: string }>>({
        conn,
        method: "GET",
        path: "/api/v1/bo/objects",
        headers: { "X-BO-Connection-Key": boConnectionKey(conn) },
      });
      return rows.map((row) => ({
        id: row.objectKey,
        name: row.displayName || row.objectKey,
        description: row.storeKey ? `store: ${row.storeKey}` : undefined,
      }));
    },
  },
  middleware: (rawConfig) =>
    createMiddleware({
      name: "BusinessObjectRecords",
      tools: [
        tool((input: z.infer<typeof schemas.query>, exeConfig) => boRecordQuery(input, exeConfig, rawConfig), {
          name: "query_records",
          description:
            "Query Business Object records by objectKey. The object definition resolves the store; do not pass storeKey.",
          schema: schemas.query,
        }),
        tool((input: z.infer<typeof schemas.get>, exeConfig) => boRecordGet(input, exeConfig, rawConfig), {
          name: "get_record",
          description: "Get one Business Object record by objectKey and id.",
          schema: schemas.get,
        }),
        tool((input: z.infer<typeof schemas.create>, exeConfig) => boRecordCreate(input, exeConfig, rawConfig), {
          name: "create_record",
          description: "Create one Business Object record. Data is validated by the platform-service object schema.",
          schema: schemas.create,
        }),
        tool((input: z.infer<typeof schemas.update>, exeConfig) => boRecordUpdate(input, exeConfig, rawConfig), {
          name: "update_record",
          description: "Patch one Business Object record by objectKey and id.",
          schema: schemas.update,
        }),
        tool((input: z.infer<typeof schemas.delete>, exeConfig) => boRecordDelete(input, exeConfig, rawConfig), {
          name: "delete_record",
          description: "Soft-delete one Business Object record. Requires user confirmation, then pass confirm:true.",
          schema: schemas.delete,
        }),
      ],
    }),
};

PluginRegistry.register(businessObjectRecordsPlugin);
