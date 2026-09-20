import { PluginRegistry } from "@axiom-lattice/core";
import { AgentType, type Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
import { platformServiceConnection } from "../connection";
import {
  boStoreKey,
  boObjectCreate,
  boObjectDelete,
  boObjectGet,
  boObjectList,
  boObjectUpdate,
  boRecordCreate,
  boRecordDelete,
  boRecordGet,
  boRecordQuery,
  boRecordUpdate,
  boStoreCreate,
  boStoreKeyCreate,
  boStoreKeyDelete,
  boStoreKeyList,
  boStoreKeyUpdate,
  boStoreList,
  boStoreTest,
} from "./executors";
import { BUSINESS_OBJECTS_BUILDER_PROMPT } from "./prompt";
import { BUSINESS_OBJECTS_MODELING_SKILL } from "./skill";

const identifier = z.string().regex(/^[a-z][a-z0-9_]{0,62}$/);

const field = z.object({
  key: identifier,
  type: z.enum([
    "string",
    "text",
    "integer",
    "long",
    "decimal",
    "boolean",
    "date",
    "datetime",
    "json",
  ]),
  required: z.boolean().optional(),
  maxLength: z.number().int().optional(),
  precision: z.number().int().optional(),
  scale: z.number().int().optional(),
  description: z.string().optional(),
});

const index = z.object({
  name: identifier.optional(),
  fields: z.array(identifier).min(1),
  unique: z.boolean().optional(),
});

const objectDefinition = z.object({
  storeKey: identifier.optional(),
  objectKey: identifier,
  displayName: z.string().optional(),
  description: z.string().optional(),
  fields: z.array(field).min(1).optional(),
  indexes: z.array(index).optional(),
  status: z.number().int().optional(),
});

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
  empty: z.object({}),
  storeCreate: z.object({
    storeKey: identifier,
    name: z.string().optional(),
    description: z.string().optional(),
    jdbcUrl: z.string().startsWith("jdbc:postgresql:"),
    username: z.string(),
    password: z.string(),
    status: z.number().int().optional(),
  }),
  storeKey: z.object({ storeKey: identifier }),
  storeKeyCreate: z.object({
    storeKey: identifier,
    keyName: identifier,
    rawKey: z.string().optional(),
    permissions: z.array(z.enum(["READ", "WRITE", "MANAGE"])).optional(),
    status: z.number().int().optional(),
  }),
  storeKeyUpdate: z.object({
    storeKey: identifier,
    keyId: z.number().int(),
    keyName: identifier.optional(),
    rawKey: z.string().optional(),
    permissions: z.array(z.enum(["READ", "WRITE", "MANAGE"])).optional(),
    status: z.number().int().optional(),
  }),
  storeKeyDelete: z.object({
    storeKey: identifier,
    keyId: z.number().int(),
    confirm: z.boolean().optional(),
  }),
  objectGet: z.object({ objectKey: identifier }),
  objectCreate: objectDefinition.extend({ storeKey: identifier }),
  objectUpdate: objectDefinition.extend({ fields: z.array(field).min(1) }),
  objectDelete: z.object({
    objectKey: identifier,
    confirm: z.boolean().optional(),
  }),
  recordCreate: z.object({
    objectKey: identifier,
    data: z.record(z.unknown()),
  }),
  recordGet: z.object({
    objectKey: identifier,
    id: z.string(),
  }),
  recordUpdate: z.object({
    objectKey: identifier,
    id: z.string(),
    data: z.record(z.unknown()),
  }),
  recordDelete: z.object({
    objectKey: identifier,
    id: z.string(),
    confirm: z.boolean().optional(),
  }),
  recordQuery: z.object({
    objectKey: identifier,
    filters: z.array(filter).optional(),
    sort: z.array(sort).optional(),
    page: z.number().int().optional(),
    pageSize: z.number().int().optional(),
  }),
};

export const businessObjectPlugin: Plugin = {
  meta: {
    type: "business-objects",
    name: "Business Objects",
    description:
      "Business Object stores, store-bound API keys, object definitions and record CRUDQ. PERMISSION MODEL — each connection uses one BO store key; query-only agents enable this middleware with allowedTools set to the read tools; schema/record writes belong to the built-in 'business-objects-builder' agent.",
    version: "1.0.0",
    category: "data",
    capabilityBundleEligible: true,
    tools: [
      { name: "list_stores", description: "List Business Object stores." },
      {
        name: "create_store",
        description:
          "Create a Business Object store backed by one PostgreSQL database. Store access keys are created separately.",
      },
      { name: "test_store", description: "Test connectivity to a Business Object store." },
      { name: "list_store_keys", description: "List API keys for one Business Object store." },
      { name: "create_store_key", description: "Create a store-bound Business Object API key." },
      { name: "update_store_key", description: "Update or rotate a store-bound Business Object API key." },
      { name: "delete_store_key", description: "Disable a store-bound Business Object API key." },
      {
        name: "list_objects",
        description: "List Business Object definitions visible to the configured BO store key.",
      },
      { name: "get_object", description: "Get one Business Object definition by objectKey." },
      {
        name: "create_object",
        description: "Create a Business Object definition and synchronize it into PostgreSQL DDL.",
      },
      {
        name: "update_object",
        description:
          "Update a Business Object definition. v1 supports additive columns only; no field removal/type changes.",
      },
      {
        name: "delete_object",
        description:
          "Soft-delete a Business Object definition. Requires user confirmation, then pass confirm:true.",
      },
      { name: "query_records", description: "Query Business Object records by objectKey." },
      { name: "get_record", description: "Get one Business Object record by objectKey and id." },
      {
        name: "create_record",
        description:
          "Create one Business Object record. Data is validated by the platform-service object schema.",
      },
      { name: "update_record", description: "Patch one Business Object record by objectKey and id." },
      {
        name: "delete_record",
        description:
          "Soft-delete one Business Object record. Requires user confirmation, then pass confirm:true.",
      },
    ],
    openExpose: [
      { name: "list_stores", readOnly: true },
      { name: "test_store", readOnly: true },
      { name: "list_store_keys", readOnly: true },
      { name: "list_objects", readOnly: true },
      { name: "get_object", readOnly: true },
      { name: "query_records", readOnly: true },
      { name: "get_record", readOnly: true },
    ],
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
  },
  connection: {
    ...platformServiceConnection,
    fields: platformServiceConnection.fields.map((field) =>
      field.key === "boStoreKey" ? { ...field, required: true } : field,
    ),
    test: async (config) => {
      try {
        const conn = connectionFromConfig(config);
        const store = await request<{ storeKey?: string; name?: string }>({
          conn,
          method: "GET",
          path: "/api/v1/bo/stores/current",
          headers: { "X-BO-Connection-Key": boStoreKey(conn) },
        });
        const label = store.name || store.storeKey || "authorized store";
        return { ok: true, message: `Connected; ${label} authorized` };
      } catch (err) {
        return { ok: false, message: err instanceof Error ? err.message : String(err) };
      }
    },
    discover: async (config) => {
      const conn = connectionFromConfig(config);
      const rows = await request<Array<{ objectKey: string; displayName?: string; storeKey?: string }>>({
        conn,
        method: "GET",
        path: "/api/v1/bo/objects",
        headers: { "X-BO-Connection-Key": boStoreKey(conn) },
      });
      return rows.map((row) => ({
        id: row.objectKey,
        name: row.displayName || row.objectKey,
        description: row.storeKey ? `store: ${row.storeKey}` : undefined,
      }));
    },
  },
  skills: {
    "business-objects-modeling": BUSINESS_OBJECTS_MODELING_SKILL,
  },
  agents: {
    "business-objects-builder": {
      key: "business-objects-builder",
      name: "Business Objects Builder",
      description:
        "Interactively design and build Business Object stores, object definitions, fields and indexes; verify each step with record queries.",
      type: AgentType.DEEP_AGENT,
      prompt: BUSINESS_OBJECTS_BUILDER_PROMPT,
      middleware: [
        {
          id: "business-objects",
          type: "business-objects",
          name: "Business Objects",
          description: "Manage stores/objects and run record CRUDQ for verification",
          enabled: true,
          config: { connections: [], connectAll: true },
        },
        {
          id: "skill",
          type: "skill",
          name: "Skill",
          description: "Load the business-objects-modeling policy",
          enabled: true,
          config: { readAll: false, skills: ["business-objects-modeling", "task-definition"] },
        },
        {
          id: "task",
          type: "task",
          name: "Task",
          description: "Persistent TaskItems as the planning surface",
          enabled: true,
          config: {},
        },
        {
          id: "ask_user_to_clarify",
          type: "ask_user_to_clarify",
          name: "Ask User",
          description: "Confirm modeling decisions before writes",
          enabled: true,
          config: {},
        },
        {
          id: "filesystem",
          type: "filesystem",
          name: "Filesystem",
          description: "Read user-provided data dictionaries or sample data",
          enabled: true,
          config: {},
        },
      ],
    },
  },
  middleware: (rawConfig) => {
    const pluginConfig = { ...rawConfig, connectionType: "business-objects" };
    return createMiddleware({
      name: "BusinessObjects",
      tools: [
        tool((input: z.infer<typeof schemas.empty>, exeConfig) => boStoreList(input, exeConfig, pluginConfig), {
          name: "list_stores",
          description: "List Business Object stores.",
          schema: schemas.empty,
        }),
        tool(
          (input: z.infer<typeof schemas.storeCreate>, exeConfig) =>
            boStoreCreate(input, exeConfig, pluginConfig),
          {
            name: "create_store",
            description:
              "Create a Business Object store backed by one PostgreSQL database. Store access keys are created separately.",
            schema: schemas.storeCreate,
          },
        ),
        tool((input: z.infer<typeof schemas.storeKey>, exeConfig) => boStoreTest(input, exeConfig, pluginConfig), {
          name: "test_store",
          description: "Test connectivity to a Business Object store.",
          schema: schemas.storeKey,
        }),
        tool(
          (input: z.infer<typeof schemas.storeKey>, exeConfig) =>
            boStoreKeyList(input, exeConfig, pluginConfig),
          {
            name: "list_store_keys",
            description: "List API keys for one Business Object store.",
            schema: schemas.storeKey,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.storeKeyCreate>, exeConfig) =>
            boStoreKeyCreate(input, exeConfig, pluginConfig),
          {
            name: "create_store_key",
            description: "Create a store-bound Business Object API key. If rawKey is omitted, the platform returns it once.",
            schema: schemas.storeKeyCreate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.storeKeyUpdate>, exeConfig) =>
            boStoreKeyUpdate(input, exeConfig, pluginConfig),
          {
            name: "update_store_key",
            description: "Update permissions/status or rotate a Business Object store API key.",
            schema: schemas.storeKeyUpdate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.storeKeyDelete>, exeConfig) =>
            boStoreKeyDelete(input, exeConfig, pluginConfig),
          {
            name: "delete_store_key",
            description: "Disable a Business Object store API key. Requires user confirmation, then pass confirm:true.",
            schema: schemas.storeKeyDelete,
          },
        ),
        tool((input: z.infer<typeof schemas.empty>, exeConfig) => boObjectList(input, exeConfig, pluginConfig), {
          name: "list_objects",
          description: "List Business Object definitions visible to the configured BO store key.",
          schema: schemas.empty,
        }),
        tool(
          (input: z.infer<typeof schemas.objectGet>, exeConfig) =>
            boObjectGet(input, exeConfig, pluginConfig),
          {
            name: "get_object",
            description: "Get one Business Object definition by objectKey.",
            schema: schemas.objectGet,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.objectCreate>, exeConfig) =>
            boObjectCreate(input, exeConfig, pluginConfig),
          {
            name: "create_object",
            description: "Create a Business Object definition and synchronize it into PostgreSQL DDL.",
            schema: schemas.objectCreate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.objectUpdate>, exeConfig) =>
            boObjectUpdate(input, exeConfig, pluginConfig),
          {
            name: "update_object",
            description:
              "Update a Business Object definition. v1 supports additive columns only; no field removal/type changes.",
            schema: schemas.objectUpdate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.objectDelete>, exeConfig) =>
            boObjectDelete(input, exeConfig, pluginConfig),
          {
            name: "delete_object",
            description:
              "Soft-delete a Business Object definition. Requires user confirmation, then pass confirm:true.",
            schema: schemas.objectDelete,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordQuery>, exeConfig) =>
            boRecordQuery(input, exeConfig, pluginConfig),
          {
            name: "query_records",
            description:
              "Query Business Object records by objectKey. The object definition resolves the store; do not pass storeKey.",
            schema: schemas.recordQuery,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordGet>, exeConfig) =>
            boRecordGet(input, exeConfig, pluginConfig),
          {
            name: "get_record",
            description: "Get one Business Object record by objectKey and id.",
            schema: schemas.recordGet,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordCreate>, exeConfig) =>
            boRecordCreate(input, exeConfig, pluginConfig),
          {
            name: "create_record",
            description:
              "Create one Business Object record. Data is validated by the platform-service object schema.",
            schema: schemas.recordCreate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordUpdate>, exeConfig) =>
            boRecordUpdate(input, exeConfig, pluginConfig),
          {
            name: "update_record",
            description: "Patch one Business Object record by objectKey and id.",
            schema: schemas.recordUpdate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordDelete>, exeConfig) =>
            boRecordDelete(input, exeConfig, pluginConfig),
          {
            name: "delete_record",
            description:
              "Soft-delete one Business Object record. Requires user confirmation, then pass confirm:true.",
            schema: schemas.recordDelete,
          },
        ),
      ],
    });
  },
};

PluginRegistry.register(businessObjectPlugin);
