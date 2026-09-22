# BO Delete Mode (soft/hard) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each Business Object choose `soft` or `hard` record deletion (default `hard`), and fix the soft-delete + unique-index conflict with partial unique indexes.

**Architecture:** `deleteMode` is stored inside the object definition schema JSON and is exposed on the object API. Unique indexes are always created as partial indexes (`WHERE "deleted" = 0`). Record deletes branch on the object's mode. Unique-constraint violations map to HTTP 409.

**Tech Stack:** Java 21 / Spring Boot / jOOQ / PostgreSQL (platform-service); TypeScript / zod / langchain plugin tooling (agent); JUnit 5 + AssertJ (Java tests); Jest (TS tests).

**Design spec:** `docs/superpowers/specs/2026-09-22-bo-delete-mode-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `platform-service/src/main/java/com/fina/platform/bo/BusinessObjectSqlSupport.java` | SQL generation: partial unique indexes; `deleteMode` normalization |
| `platform-service/src/main/java/com/fina/platform/bo/BusinessObjectDtos.java` | Request/response records now carry `deleteMode` |
| `platform-service/src/main/java/com/fina/platform/bo/BusinessObjectService.java` | Persist/validate `deleteMode`; hard vs soft delete branching |
| `platform-service/src/main/java/com/fina/platform/exception/GlobalExceptionHandler.java` | Map unique violations to 409 |
| `platform-service/src/test/java/com/fina/platform/bo/BusinessObjectSqlSupportTest.java` | Unit tests for SQL + mode normalization |
| `platform-service/src/test/java/com/fina/platform/exception/GlobalExceptionHandlerTest.java` | New: 409 mapping test |
| `agent/src/agents/platform_service/business_objects/executors.ts` | `ObjectDefinitionInput` carries `deleteMode` |
| `agent/src/agents/platform_service/business_objects/plugin.ts` | Tool schema + descriptions expose `deleteMode` |
| `agent/src/agents/platform_service/business_objects/skill.ts` | Modeling policy: ask user, default hard |
| `agent/src/agents/platform_service/business_objects/prompt.ts` | Builder prompt: ask delete mode before create |
| `agent/src/agents/platform_service/__tests__/business_objects.test.ts` | Executor forwarding test |
| `agent/src/agents/platform_service/__tests__/builder_assets.test.ts` | Skill/prompt content test |

---

### Task 1: Partial unique index + `deleteMode` normalization (pure SQL support)

**Files:**
- Modify: `platform-service/src/main/java/com/fina/platform/bo/BusinessObjectSqlSupport.java`
- Test: `platform-service/src/test/java/com/fina/platform/bo/BusinessObjectSqlSupportTest.java`

- [ ] **Step 1: Write the failing tests**

Append these two tests to `BusinessObjectSqlSupportTest` (before the closing `}`):

```java
    @Test
    void uniqueIndexesArePartialOverLiveRows() {
        List<FieldDefinition> fields = BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("customer_no", "string", true, 64, null, null, null),
                new FieldDefinition("tag_key", "string", true, 64, null, null, null)
        ));
        List<IndexDefinition> indexes = BusinessObjectSqlSupport.normalizeIndexes(List.of(
                new IndexDefinition("uk_tag_customer_key", List.of("customer_no", "tag_key"), true),
                new IndexDefinition("idx_tag_key", List.of("tag_key"), false)
        ), fields);

        List<String> sql = BusinessObjectSqlSupport.createIndexSql("bo_customer_tag", "customer_tag", indexes);

        assertThat(sql).contains(
                "DROP INDEX IF EXISTS \"idx_customer_tag_uk_tag_customer_key\"",
                "CREATE UNIQUE INDEX IF NOT EXISTS \"idx_customer_tag_uk_tag_customer_key\" "
                        + "ON \"bo_customer_tag\" (\"customer_no\", \"tag_key\") WHERE \"deleted\" = 0",
                "CREATE INDEX IF NOT EXISTS \"idx_customer_tag_idx_tag_key\" "
                        + "ON \"bo_customer_tag\" (\"tag_key\")");
        assertThat(sql).noneMatch(s -> s.contains("idx_tag_key") && s.contains("WHERE"));
    }

    @Test
    void deleteModeDefaultsToHardAndRejectsOtherValues() {
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode(null)).isEqualTo("hard");
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode("  ")).isEqualTo("hard");
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode("SOFT")).isEqualTo("soft");
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode("hard")).isEqualTo("hard");

        assertThatThrownBy(() -> BusinessObjectSqlSupport.normalizeDeleteMode("purge"))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("deleteMode must be");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./gradlew test --tests 'com.fina.platform.bo.BusinessObjectSqlSupportTest'`
Expected: FAIL — `normalizeDeleteMode` does not exist; index SQL has no `WHERE "deleted" = 0`.

- [ ] **Step 3: Implement the SQL support changes**

In `BusinessObjectSqlSupport.java`, add constants + helper near the top of the class (after `FIELD_TYPES`):

```java
    static final String DELETE_MODE_SOFT = "soft";
    static final String DELETE_MODE_HARD = "hard";
```

Add the normalization helper (e.g. after `normalizeIndexes`):

```java
    static String normalizeDeleteMode(String value) {
        if (value == null || value.isBlank()) {
            return DELETE_MODE_HARD;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if (!DELETE_MODE_SOFT.equals(normalized) && !DELETE_MODE_HARD.equals(normalized)) {
            throw ApiException.badRequest("deleteMode must be 'soft' or 'hard'");
        }
        return normalized;
    }
```

Replace the body of `createIndexSql` with:

```java
    static List<String> createIndexSql(String tableName, String objectKey, List<IndexDefinition> indexes) {
        List<String> sql = new ArrayList<>();
        for (IndexDefinition index : indexes) {
            String indexName = "idx_" + objectKey + "_" + index.name();
            if (indexName.length() > 63) {
                indexName = indexName.substring(0, 63);
            }
            boolean unique = Boolean.TRUE.equals(index.unique());
            String columns = index.fields().stream().map(BusinessObjectSqlSupport::quote)
                    .collect(java.util.stream.Collectors.joining(", "));
            if (unique) {
                sql.add("DROP INDEX IF EXISTS " + quote(indexName));
                sql.add("CREATE UNIQUE INDEX IF NOT EXISTS " + quote(indexName)
                        + " ON " + quote(tableName) + " (" + columns + ") WHERE \"deleted\" = 0");
            } else {
                sql.add("CREATE INDEX IF NOT EXISTS " + quote(indexName)
                        + " ON " + quote(tableName) + " (" + columns + ")");
            }
        }
        return sql;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew test --tests 'com.fina.platform.bo.BusinessObjectSqlSupportTest'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add platform-service/src/main/java/com/fina/platform/bo/BusinessObjectSqlSupport.java \
        platform-service/src/test/java/com/fina/platform/bo/BusinessObjectSqlSupportTest.java
git commit -m "feat(bo): partial unique indexes and deleteMode normalization"
```

---

### Task 2: Plumb `deleteMode` through DTOs and service

**Files:**
- Modify: `platform-service/src/main/java/com/fina/platform/bo/BusinessObjectDtos.java`
- Modify: `platform-service/src/main/java/com/fina/platform/bo/BusinessObjectService.java`

- [ ] **Step 1: Add `deleteMode` to the DTOs**

In `BusinessObjectDtos.java`, change `ObjectDefinitionRequest` to add a trailing field:

```java
    public record ObjectDefinitionRequest(
            String storeKey,
            String objectKey,
            String displayName,
            String description,
            List<FieldDefinition> fields,
            List<IndexDefinition> indexes,
            Integer status,
            String deleteMode
    ) {
    }
```

Change `ObjectDefinitionResponse` to add a trailing field:

```java
    public record ObjectDefinitionResponse(
            Long id,
            Long storeId,
            String storeKey,
            String objectKey,
            String tableName,
            String displayName,
            String description,
            List<FieldDefinition> fields,
            List<IndexDefinition> indexes,
            Integer status,
            String deleteMode
    ) {
    }
```

- [ ] **Step 2: Extend `ObjectSchema` to carry the mode**

In `BusinessObjectService.java`, replace the `ObjectSchema` record definition (around line 1034):

```java
    private record ObjectSchema(List<FieldDefinition> fields, List<IndexDefinition> indexes, String deleteMode) {
        private ObjectSchema {
            fields = fields == null ? List.of() : List.copyOf(fields);
            indexes = indexes == null ? List.of() : List.copyOf(indexes);
            deleteMode = BusinessObjectSqlSupport.normalizeDeleteMode(deleteMode);
        }

        boolean softDelete() {
            return BusinessObjectSqlSupport.DELETE_MODE_SOFT.equals(deleteMode);
        }
    }
```

- [ ] **Step 3: Persist the mode on create**

In `createObject`, change the schema construction and the response mapping:

```java
        List<IndexDefinition> indexes = BusinessObjectSqlSupport.normalizeIndexes(request.indexes(), fields);
        ObjectSchema schema = new ObjectSchema(fields, indexes,
                BusinessObjectSqlSupport.normalizeDeleteMode(request.deleteMode()));
```

- [ ] **Step 4: Reject mode changes on update**

In `updateObject`, after `ObjectDefinition existing = loadDefinition(...)` and the `storeKey` check, insert:

```java
        String requestedMode = BusinessObjectSqlSupport.normalizeDeleteMode(request.deleteMode());
        if (!requestedMode.equals(existing.schema().deleteMode())) {
            throw ApiException.badRequest("deleteMode cannot be changed after object creation; recreate the object");
        }
```

Then change the schema construction in `updateObject`:

```java
        ObjectSchema schema = new ObjectSchema(fields, indexes, existing.schema().deleteMode());
```

- [ ] **Step 5: Echo the mode in responses**

In `definitionFromRow` (around line 774), change the returned response to append `schema.deleteMode()`:

```java
        return new ObjectDefinitionResponse(id, storeId, storeKey, objectKey, tableName,
                displayName, description, schema.fields(), schema.indexes(), status, schema.deleteMode());
```

In `ObjectDefinition.toResponse()` (around line 1045), do the same:

```java
            return new ObjectDefinitionResponse(id, storeId, storeKey, objectKey, tableName,
                    displayName, description, schema.fields(), schema.indexes(), status, schema.deleteMode());
```

- [ ] **Step 6: Branch delete behavior on the mode**

Replace `deleteRecord` (around line 407) with:

```java
    public Map<String, Object> deleteRecord(BoAuthContext auth, String objectKey, String id) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.WRITE);
        DSLContext dsl = storeRuntime(definition.storeId()).dsl();
        int updated;
        if (definition.schema().softDelete()) {
            updated = dsl.update(table(definition))
                    .set(deletedField(), 1)
                    .set(field("updated_at", LocalDateTime.class), LocalDateTime.now())
                    .where(idField().eq(id).and(deletedField().eq(0)))
                    .execute();
        } else {
            updated = dsl.deleteFrom(table(definition))
                    .where(idField().eq(id))
                    .execute();
        }
        return Map.of("deleted", updated > 0, "id", id);
    }
```

In `deleteRecords`, replace the inner loop body (inside `transactionResult`) with:

```java
            List<String> removed = new ArrayList<>(requested.size());
            for (String id : requested) {
                int updated = definition.schema().softDelete()
                        ? tx.update(table)
                                .set(deletedField(), 1)
                                .set(field("updated_at", LocalDateTime.class), LocalDateTime.now())
                                .where(idField().eq(id).and(deletedField().eq(0)))
                                .execute()
                        : tx.deleteFrom(table).where(idField().eq(id)).execute();
                if (updated > 0) {
                    removed.add(id);
                }
            }
            return removed;
```

Also update the `deleteRecords` javadoc first line so it no longer claims unconditional soft-delete:

```java
    /**
     * Delete many records atomically. Hard-delete objects remove rows; soft-delete
     * objects set deleted = 1. Idempotent: ids that do not exist are ignored, not an
     * error; {@code deleted} counts the rows actually changed. Runs in a transaction
     * on the store database.
     */
```

- [ ] **Step 7: Compile**

Run: `./gradlew compileJava`
Expected: BUILD SUCCESSFUL.

- [ ] **Step 8: Run the BO unit tests**

Run: `./gradlew test --tests 'com.fina.platform.bo.*'`
Expected: PASS (existing tests still green).

- [ ] **Step 9: Commit**

```bash
git add platform-service/src/main/java/com/fina/platform/bo/BusinessObjectDtos.java \
        platform-service/src/main/java/com/fina/platform/bo/BusinessObjectService.java
git commit -m "feat(bo): per-object soft/hard delete mode"
```

---

### Task 3: Map unique violations to HTTP 409

**Files:**
- Modify: `platform-service/src/main/java/com/fina/platform/exception/GlobalExceptionHandler.java`
- Test: `platform-service/src/test/java/com/fina/platform/exception/GlobalExceptionHandlerTest.java` (create)

- [ ] **Step 1: Write the failing test**

Create `platform-service/src/test/java/com/fina/platform/exception/GlobalExceptionHandlerTest.java`:

```java
package com.fina.platform.exception;

import org.junit.jupiter.api.Test;
import org.jooq.exception.DataAccessException;

import java.sql.SQLException;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class GlobalExceptionHandlerTest {

    private final GlobalExceptionHandler handler = new GlobalExceptionHandler();

    @Test
    void uniqueViolationMapsTo409() {
        DataAccessException error = new DataAccessException("insert",
                new SQLException("duplicate key value violates unique constraint", "23505"));

        var response = handler.handleDataAccess(error);

        assertThat(response.getStatusCode().value()).isEqualTo(409);
        assertThat(response.getBody()).containsEntry("code", "CONFLICT");
    }

    @Test
    void otherDataAccessErrorsStay500() {
        DataAccessException error = new DataAccessException("select",
                new SQLException("connection reset", "08006"));

        var response = handler.handleDataAccess(error);

        assertThat(response.getStatusCode().value()).isEqualTo(500);
        assertThat(response.getBody()).containsEntry("code", "INTERNAL_ERROR");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew test --tests 'com.fina.platform.exception.GlobalExceptionHandlerTest'`
Expected: FAIL — `handleDataAccess` does not exist.

- [ ] **Step 3: Implement the handler**

In `GlobalExceptionHandler.java`, add these imports:

```java
import org.jooq.exception.DataAccessException;
import java.sql.SQLException;
```

and add the handler + helper methods:

```java
    /** jOOQ wraps store-database errors; unique violations are a client conflict, not a server fault. */
    @ExceptionHandler(DataAccessException.class)
    public ResponseEntity<Map<String, Object>> handleDataAccess(DataAccessException e) {
        if (isUniqueViolation(e)) {
            return ResponseEntity.status(409)
                    .body(Map.of("code", "CONFLICT", "message", "unique constraint violated"));
        }
        log.error("database error", e);
        return ResponseEntity.status(500)
                .body(Map.of("code", "INTERNAL_ERROR", "message", String.valueOf(e.getMessage())));
    }

    private boolean isUniqueViolation(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof SQLException sql && "23505".equals(sql.getSQLState())) {
                return true;
            }
            if (t instanceof org.springframework.dao.DuplicateKeyException) {
                return true;
            }
        }
        return false;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./gradlew test --tests 'com.fina.platform.exception.GlobalExceptionHandlerTest'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add platform-service/src/main/java/com/fina/platform/exception/GlobalExceptionHandler.java \
        platform-service/src/test/java/com/fina/platform/exception/GlobalExceptionHandlerTest.java
git commit -m "feat(bo): map unique constraint violations to 409"
```

---

### Task 4: Expose `deleteMode` in the agent BO plugin

**Files:**
- Modify: `agent/src/agents/platform_service/business_objects/executors.ts`
- Modify: `agent/src/agents/platform_service/business_objects/plugin.ts`
- Modify: `agent/src/agents/platform_service/business_objects/skill.ts`
- Modify: `agent/src/agents/platform_service/business_objects/prompt.ts`
- Test: `agent/src/agents/platform_service/__tests__/business_objects.test.ts`
- Test: `agent/src/agents/platform_service/__tests__/builder_assets.test.ts`

- [ ] **Step 1: Write the failing tests**

In `agent/src/agents/platform_service/__tests__/business_objects.test.ts`, add this test inside `describe("business object executors", ...)`:

```ts
  it("forwards deleteMode when creating object definitions", async () => {
    await boObjectCreate({
      storeKey: "crm_store",
      objectKey: "customer_tag",
      fields: [{ key: "tag_key", type: "string", required: true }],
      deleteMode: "soft",
    }, exeConfig, rawConfig);

    const [, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(JSON.parse(init.body)).toMatchObject({
      storeKey: "crm_store",
      objectKey: "customer_tag",
      deleteMode: "soft",
    });
  });
```

In `agent/src/agents/platform_service/__tests__/builder_assets.test.ts`, add this test inside the `describe`:

```ts
  it("documents the delete mode policy and default", () => {
    const content = BUSINESS_OBJECTS_MODELING_SKILL.content;
    expect(content).toContain("## 5.1 删除模式");
    expect(content).toContain("deleteMode");
    expect(content).toContain("物理删除");
    expect(BUSINESS_OBJECTS_BUILDER_PROMPT).toContain("deleteMode");
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd agent && pnpm exec jest src/agents/platform_service/__tests__/business_objects.test.ts src/agents/platform_service/__tests__/builder_assets.test.ts`
Expected: FAIL — `deleteMode` is dropped by the executor type and skill content lacks the section.

- [ ] **Step 3: Add `deleteMode` to the executor input type**

In `executors.ts`, change `ObjectDefinitionInput`:

```ts
export interface ObjectDefinitionInput {
  storeKey?: string;
  objectKey: string;
  displayName?: string;
  description?: string;
  fields?: FieldDefinition[];
  indexes?: IndexDefinition[];
  status?: number;
  deleteMode?: "soft" | "hard";
}
```

- [ ] **Step 4: Add `deleteMode` to the tool schema and descriptions**

In `plugin.ts`, change the `objectDefinition` zod schema:

```ts
const objectDefinition = z.object({
  storeKey: identifier.optional(),
  objectKey: identifier,
  displayName: z.string().optional(),
  description: z.string().optional(),
  fields: z.array(field).min(1).optional(),
  indexes: z.array(index).optional(),
  status: z.number().int().optional(),
  deleteMode: z.enum(["soft", "hard"]).optional(),
});
```

Also update the `meta.tools` descriptions for `create_object` and `delete_record` / `delete_records`:

```ts
      {
        name: "create_object",
        description:
          "Create a Business Object definition and synchronize it into PostgreSQL DDL. Optional deleteMode: 'hard' (default, physical delete) or 'soft' (keeps deleted rows).",
      },
```

```ts
      {
        name: "delete_record",
        description:
          "Delete one Business Object record (respecting the object's deleteMode). Requires user confirmation, then pass confirm:true.",
      },
```

```ts
      {
        name: "delete_records",
        description:
          "Delete many Business Object records by id atomically (1-500), respecting the object's deleteMode. Idempotent: missing ids are ignored. Requires user confirmation, then pass confirm:true.",
      },
```

- [ ] **Step 5: Update the modeling skill**

In `skill.ts`, change the `**Record**` bullet (section 1):

```
- **Record**：object 下的一行数据，按 id 读写；删除模式由对象的 deleteMode 决定（soft 软删 / hard 物理删除），默认物理删除。
```

Add a new section right after section 5 (before `## 6. v1 演进约束`):

```
## 5.1 删除模式（deleteMode）
- create_object 可传 deleteMode：\`hard\`（物理删除，**默认**）或 \`soft\`（软删，保留 deleted=1 行，可审计）。
- **建表前必须询问用户**需要物理删除还是软删除；用户未明确时用默认 \`hard\`。
- update_object **不能修改** deleteMode；需要改就重建对象。
- soft 对象的唯一索引是部分唯一索引（WHERE deleted = 0），删除后可再用同名键。
```

Update `## 9. 标准工作流` step 4:

```
4. 先询问用户 deleteMode（默认物理删除），再 create_object（或 update_object 加列）。
```

Add to `## 10. 验收清单`:

```
- [ ] 已与用户确认 deleteMode（默认物理删除）。
```

- [ ] **Step 6: Update the builder prompt**

In `prompt.ts`, add one operating rule line before the "Track progress" line:

```
- Before create_object, ask the user whether records should be hard-deleted (default, no history) or soft-deleted (keeps history), and pass deleteMode accordingly.
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd agent && pnpm exec jest src/agents/platform_service`
Expected: PASS (all platform_service suites).

- [ ] **Step 8: Type-check via build**

Run: `cd agent && pnpm build`
Expected: tsup build succeeds.

- [ ] **Step 9: Commit**

```bash
git add agent/src/agents/platform_service/business_objects/executors.ts \
        agent/src/agents/platform_service/business_objects/plugin.ts \
        agent/src/agents/platform_service/business_objects/skill.ts \
        agent/src/agents/platform_service/business_objects/prompt.ts \
        agent/src/agents/platform_service/__tests__/business_objects.test.ts \
        agent/src/agents/platform_service/__tests__/builder_assets.test.ts
git commit -m "feat(bo): expose deleteMode in BO plugin and builder policy"
```

---

### Task 5: Full verification

**Files:** none (verification only)

- [ ] **Step 1: platform-service full test suite**

Run: `cd platform-service && ./gradlew test`
Expected: BUILD SUCCESSFUL, all tests pass.

- [ ] **Step 2: agent test suite**

Run: `cd agent && pnpm exec jest`
Expected: all suites pass.

- [ ] **Step 3: agent build**

Run: `cd agent && pnpm build`
Expected: build succeeds.

- [ ] **Step 4: Record evidence**

Paste the actual `Tests: N passed` / `BUILD SUCCESSFUL` lines into the task/PR description before claiming completion.

---

### Task 6: Deploy and rebuild BO objects (ops)

**Files:** none (deploy + live data)

- [ ] **Step 1: Deploy the updated platform-service and agent**

Use the project's existing pipeline (Docker images via CI / `docker-deploy-38.sh`, agent deploy as usual). Confirm `GET /api/v1/bo/objects` returns the new `deleteMode` field.

- [ ] **Step 2: Confirm all BO tables are empty before rebuilding**

Run against the `elc` store database (schema `bo`):

```sql
SELECT 'bo_customer' t, count(*) FROM bo.bo_customer
UNION ALL SELECT 'bo_customer_tag', count(*) FROM bo.bo_customer_tag
UNION ALL SELECT 'bo_customer_activity', count(*) FROM bo.bo_customer_activity
UNION ALL SELECT 'bo_product', count(*) FROM bo.bo_product
UNION ALL SELECT 'bo_voice_record', count(*) FROM bo.bo_voice_record;
```

Expected: all counts `0`. If any table has data, STOP and get user confirmation before dropping.

- [ ] **Step 3: Dump the existing object definitions**

Before dropping, capture each definition so it can be replayed with a `deleteMode` added. For each of `customer`, `customer_tag`, `customer_activity`, `product`, `voice_record`:

```bash
curl -sS "http://localhost:5707/api/v1/bo/objects/customer_tag" \
  -H "X-Bo-Connection-Key: <bo-store-key>" > /tmp/bo-customer_tag.json
```

(The response `fields` / `indexes` arrays are exactly what Step 5 replays; keep these files.)

- [ ] **Step 4: Drop the existing tables and object definitions**

Store DB (`elc`, schema `bo`):

```sql
DROP TABLE IF EXISTS bo.bo_customer_tag, bo.bo_customer, bo.bo_customer_activity, bo.bo_product, bo.bo_voice_record;
```

Platform DB (`bo_object_definitions` / `bo_object_ddl_history` rows for store `elc`, `store_id = 1`):

```sql
DELETE FROM bo_object_ddl_history WHERE store_id = 1;
DELETE FROM bo_object_definitions WHERE store_id = 1;
```

- [ ] **Step 5: Recreate the objects with chosen `deleteMode`**

Use the `business-objects-builder` agent (it asks the delete-mode question), or call platform-service directly with the admin key. Convert each dumped definition (`/tmp/bo-<objectKey>.json`) into a create request by keeping `storeKey`, `objectKey`, `displayName`, `description`, `fields`, `indexes` and adding `deleteMode`. Recommended: default `hard` except where delete history is required. Example for `customer_tag` with **hard** delete:

```bash
jq '{storeKey, objectKey, displayName, description, fields, indexes, deleteMode: "hard"}' \
  /tmp/bo-customer_tag.json > /tmp/create-customer_tag.json

curl -sS -X POST "http://localhost:5707/api/v1/bo/objects" \
  -H "X-Api-Key: <admin-key>" -H "Content-Type: application/json" \
  -d @/tmp/create-customer_tag.json
```

Repeat for `customer`, `customer_activity`, `product`, `voice_record`. Choose `"deleteMode": "soft"` for any object that needs delete history.

- [ ] **Step 6: Verify the partial index was created**

Run in the store DB:

```sql
SELECT indexname, indexdef FROM pg_indexes
WHERE schemaname = 'bo' AND indexname LIKE 'idx_%_uk_%';
```

Expected: each unique index `indexdef` ends with `WHERE (deleted = 0)`.

- [ ] **Step 7: Live-verify both delete modes**

Through the Open MCP endpoint (as done earlier with `business-objects_create_record` / `query_records` / `delete_records`):

1. Create a probe record, delete it, then create a record with the **same unique key** again — for a `soft` object this MUST succeed (proves the fix); for a `hard` object it also succeeds.
2. Confirm `query_records` returns `total = 0` after deletion and that no leftover records remain.

- [ ] **Step 8: Clean up probe data and report**

Delete all probe records and confirm the tables are empty again.
