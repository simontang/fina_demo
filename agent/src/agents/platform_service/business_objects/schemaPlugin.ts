import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
import { platformServiceConnection } from "../connection";
import {
  boConnectionKey,
  boObjectCreate,
  boObjectDelete,
  boObjectGet,
  boObjectList,
  boObjectUpdate,
  boStoreCreate,
  boStoreGrantList,
  boStoreGrantUpsert,
  boStoreList,
  boStoreTest,
} from "./executors";

const identifier = z.string().regex(/^[a-z][a-z0-9_]{0,62}$/);

const field = z.object({
  key: identifier,
  type: z.enum(["string", "text", "integer", "long", "decimal", "boolean", "date", "datetime", "json"]),
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
  storeGrant: z.object({
    storeKey: identifier,
    granteeKey: identifier.optional(),
    canRead: z.boolean().optional(),
    canWrite: z.boolean().optional(),
    canManage: z.boolean().optional(),
    status: z.number().int().optional(),
  }),
  objectGet: z.object({ objectKey: identifier }),
  objectCreate: objectDefinition.extend({ storeKey: identifier }),
  objectUpdate: objectDefinition.extend({ fields: z.array(field).min(1) }),
  objectDelete: z.object({
    objectKey: identifier,
    confirm: z.boolean().optional(),
  }),
};

export const businessObjectSchemaPlugin: Plugin = {
  meta: {
    type: "business-object-schema",
    name: "Business Object Schema",
    description:
      "Manage Business Object stores, store grants, and object definitions. Definitions are synchronized by platform-service into PostgreSQL DDL.",
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
      { name: "list_stores", readOnly: true },
      { name: "test_store", readOnly: true },
      { name: "list_store_grants", readOnly: true },
      { name: "list_objects", readOnly: true },
      { name: "get_object", readOnly: true },
      { name: "create_store" },
      { name: "grant_store" },
      { name: "create_object" },
      { name: "update_object" },
      { name: "delete_object", destructive: true },
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
      name: "BusinessObjectSchema",
      tools: [
        tool((input: z.infer<typeof schemas.empty>, exeConfig) => boStoreList(input, exeConfig, rawConfig), {
          name: "list_stores",
          description: "List Business Object stores.",
          schema: schemas.empty,
        }),
        tool((input: z.infer<typeof schemas.storeCreate>, exeConfig) => boStoreCreate(input, exeConfig, rawConfig), {
          name: "create_store",
          description: "Create a Business Object store backed by one PostgreSQL database and create the default store grant.",
          schema: schemas.storeCreate,
        }),
        tool((input: z.infer<typeof schemas.storeKey>, exeConfig) => boStoreTest(input, exeConfig, rawConfig), {
          name: "test_store",
          description: "Test connectivity to a Business Object store.",
          schema: schemas.storeKey,
        }),
        tool((input: z.infer<typeof schemas.storeKey>, exeConfig) => boStoreGrantList(input, exeConfig, rawConfig), {
          name: "list_store_grants",
          description: "List grants for one Business Object store.",
          schema: schemas.storeKey,
        }),
        tool((input: z.infer<typeof schemas.storeGrant>, exeConfig) => boStoreGrantUpsert(input, exeConfig, rawConfig), {
          name: "grant_store",
          description: "Create or update a Business Object store grant.",
          schema: schemas.storeGrant,
        }),
        tool((input: z.infer<typeof schemas.empty>, exeConfig) => boObjectList(input, exeConfig, rawConfig), {
          name: "list_objects",
          description: "List Business Object definitions visible to the configured BO grant key.",
          schema: schemas.empty,
        }),
        tool((input: z.infer<typeof schemas.objectGet>, exeConfig) => boObjectGet(input, exeConfig, rawConfig), {
          name: "get_object",
          description: "Get one Business Object definition by objectKey.",
          schema: schemas.objectGet,
        }),
        tool((input: z.infer<typeof schemas.objectCreate>, exeConfig) => boObjectCreate(input, exeConfig, rawConfig), {
          name: "create_object",
          description: "Create a Business Object definition and synchronize it into PostgreSQL DDL.",
          schema: schemas.objectCreate,
        }),
        tool((input: z.infer<typeof schemas.objectUpdate>, exeConfig) => boObjectUpdate(input, exeConfig, rawConfig), {
          name: "update_object",
          description: "Update a Business Object definition. v1 supports additive columns only; no field removal/type changes.",
          schema: schemas.objectUpdate,
        }),
        tool((input: z.infer<typeof schemas.objectDelete>, exeConfig) => boObjectDelete(input, exeConfig, rawConfig), {
          name: "delete_object",
          description: "Soft-delete a Business Object definition. Requires user confirmation, then pass confirm:true.",
          schema: schemas.objectDelete,
        }),
      ],
    }),
};

PluginRegistry.register(businessObjectSchemaPlugin);
