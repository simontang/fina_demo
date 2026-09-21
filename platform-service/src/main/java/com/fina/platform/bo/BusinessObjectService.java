package com.fina.platform.bo;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.platform.bo.BusinessObjectDtos.BatchCreateResponse;
import com.fina.platform.bo.BusinessObjectDtos.BatchDeleteRequest;
import com.fina.platform.bo.BusinessObjectDtos.BatchDeleteResponse;
import com.fina.platform.bo.BusinessObjectDtos.BatchRecordRequest;
import com.fina.platform.bo.BusinessObjectDtos.FieldDefinition;
import com.fina.platform.bo.BusinessObjectDtos.Filter;
import com.fina.platform.bo.BusinessObjectDtos.IndexDefinition;
import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionRequest;
import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionResponse;
import com.fina.platform.bo.BusinessObjectDtos.QueryRequest;
import com.fina.platform.bo.BusinessObjectDtos.QueryResponse;
import com.fina.platform.bo.BusinessObjectDtos.RecordRequest;
import com.fina.platform.bo.BusinessObjectDtos.RecordResponse;
import com.fina.platform.bo.BusinessObjectDtos.Sort;
import com.fina.platform.bo.BusinessObjectDtos.StoreApiKeyRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreApiKeyResponse;
import com.fina.platform.bo.BusinessObjectDtos.StoreRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreResponse;
import com.fina.platform.exception.ApiException;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.jooq.Condition;
import org.jooq.DSLContext;
import org.jooq.Field;
import org.jooq.JSONB;
import org.jooq.SQLDialect;
import org.jooq.SortField;
import org.jooq.Table;
import org.jooq.impl.DSL;
import org.jooq.impl.SQLDataType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.sql.DataSource;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class BusinessObjectService {
    private static final int DEFAULT_PAGE_SIZE = 50;
    private static final int MAX_PAGE_SIZE = 500;
    private static final int MAX_BATCH_RECORDS = 500;
    private static final SecureRandom RANDOM = new SecureRandom();

    static {
        System.setProperty("org.jooq.no-logo", "true");
        System.setProperty("org.jooq.no-tips", "true");
    }

    private final JdbcTemplate jdbcTemplate;
    private final ObjectMapper objectMapper;
    private final Map<String, StoreRuntime> storeRuntimes = new ConcurrentHashMap<>();

    public BoAuthContext authenticate(HttpServletRequest request) {
        String connectionKey = headerIgnoreCase(request, "X-BO-Connection-Key");
        if (isBlank(connectionKey)) {
            throw new ApiException(401, "BO_STORE_KEY_REQUIRED", "X-BO-Connection-Key header is required");
        }
        List<BoGrant> grants = jdbcTemplate.query("""
                SELECT s.id AS store_id, s.store_key, k.key_name, k.permissions_json
                  FROM bo_store_api_keys k
                  JOIN bo_stores s
                    ON s.id = k.store_id
                   AND s.status = 1
                   AND s.deleted = 0
                 WHERE k.key_hash = ?
                   AND k.status = 1
                   AND k.deleted = 0
                """, (rs, rowNum) -> {
            Set<String> permissions = permissions(rs.getString("permissions_json"));
            return new BoGrant(
                    rs.getLong("store_id"),
                    rs.getString("store_key"),
                    rs.getString("key_name"),
                    permissions.contains(Grant.READ.name()),
                    permissions.contains(Grant.WRITE.name()),
                    permissions.contains(Grant.MANAGE.name())
            );
        }, hash(connectionKey.trim()));
        if (grants.isEmpty()) {
            throw new ApiException(401, "BO_STORE_KEY_INVALID", "BO store key is invalid");
        }
        return new BoAuthContext("store-key", List.copyOf(grants));
    }

    public List<StoreResponse> listStores() {
        return jdbcTemplate.query("""
                SELECT s.id, s.store_key, s.name, s.description, s.jdbc_url, s.schema_name, s.username, s.status
                FROM bo_stores s
                WHERE s.deleted = 0
                ORDER BY s.store_key
                """, (rs, rowNum) -> new StoreResponse(
                rs.getLong("id"),
                rs.getString("store_key"),
                rs.getString("name"),
                rs.getString("description"),
                rs.getString("jdbc_url"),
                rs.getString("schema_name"),
                rs.getString("username"),
                rs.getInt("status")
        ));
    }

    @Transactional
    public StoreResponse createStore(StoreRequest request) {
        String storeKey = BusinessObjectSqlSupport.requireIdentifier(request.storeKey(), "storeKey");
        if (request.jdbcUrl() == null || !request.jdbcUrl().startsWith("jdbc:postgresql:")) {
            throw ApiException.badRequest("jdbcUrl must be a PostgreSQL JDBC URL");
        }
        if (isBlank(request.username())) {
            throw ApiException.badRequest("username is required");
        }
        if (request.password() == null) {
            throw ApiException.badRequest("password is required");
        }
        String name = isBlank(request.name()) ? storeKey : request.name().trim();
        String schemaName = isBlank(request.schemaName())
                ? "public"
                : BusinessObjectSqlSupport.requireIdentifier(request.schemaName(), "schemaName");
        Integer status = request.status() == null ? 1 : request.status();

        Long id = jdbcTemplate.queryForObject("""
                INSERT INTO bo_stores (store_key, name, description, jdbc_url, schema_name, username, password, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """, Long.class, storeKey, name, request.description(), request.jdbcUrl().trim(), schemaName,
                request.username().trim(), request.password(), status);
        return getStoreById(id);
    }

    public Map<String, Object> testStore(String storeKey) {
        StoreConnection connection = loadStoreConnection(storeKey);
        JdbcTemplate template = storeRuntime(connection.storeId()).jdbc();
        Integer ok = template.queryForObject("SELECT 1", Integer.class);
        return Map.of("ok", ok != null && ok == 1, "storeKey", storeKey);
    }

    public StoreResponse currentStore(BoAuthContext auth) {
        return getStoreById(singleGrant(auth).storeId());
    }

    public List<StoreApiKeyResponse> listStoreApiKeys(String storeKey) {
        StoreConnection store = loadStoreConnection(storeKey);
        return jdbcTemplate.query("""
                SELECT id, store_id, key_name, permissions_json, status, created_at, updated_at
                FROM bo_store_api_keys
                WHERE store_id = ? AND deleted = 0
                ORDER BY key_name
                """, (rs, rowNum) -> new StoreApiKeyResponse(
                rs.getLong("id"),
                rs.getLong("store_id"),
                store.storeKey(),
                rs.getString("key_name"),
                permissions(rs.getString("permissions_json")).stream().sorted().toList(),
                rs.getInt("status"),
                rs.getTimestamp("created_at").toLocalDateTime(),
                rs.getTimestamp("updated_at").toLocalDateTime(),
                null
        ), store.storeId());
    }

    @Transactional
    public StoreApiKeyResponse createStoreApiKey(String storeKey, StoreApiKeyRequest request) {
        StoreConnection store = loadStoreConnection(storeKey);
        String keyName = BusinessObjectSqlSupport.requireIdentifier(request.keyName(), "keyName");
        List<String> normalizedPermissions = normalizePermissions(request.permissions());
        String rawKey = isBlank(request.rawKey()) ? generateKey() : request.rawKey().trim();
        Integer status = request.status() == null ? 1 : request.status();
        Long id = jdbcTemplate.queryForObject("""
                INSERT INTO bo_store_api_keys (store_id, key_name, key_hash, permissions_json, status)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """, Long.class, store.storeId(), keyName, hash(rawKey), writeJson(normalizedPermissions), status);
        StoreApiKeyResponse stored = getStoreApiKey(store, id);
        return new StoreApiKeyResponse(stored.id(), stored.storeId(), stored.storeKey(), stored.keyName(),
                stored.permissions(), stored.status(), stored.createdAt(), stored.updatedAt(),
                isBlank(request.rawKey()) ? rawKey : null);
    }

    @Transactional
    public StoreApiKeyResponse updateStoreApiKey(String storeKey, Long keyId, StoreApiKeyRequest request) {
        StoreConnection store = loadStoreConnection(storeKey);
        StoreApiKeyResponse existing = getStoreApiKey(store, keyId);
        String keyName = isBlank(request.keyName())
                ? existing.keyName()
                : BusinessObjectSqlSupport.requireIdentifier(request.keyName(), "keyName");
        List<String> normalizedPermissions = request.permissions() == null
                ? existing.permissions()
                : normalizePermissions(request.permissions());
        Integer status = request.status() == null ? existing.status() : request.status();
        if (isBlank(request.rawKey())) {
            jdbcTemplate.update("""
                    UPDATE bo_store_api_keys
                    SET key_name = ?, permissions_json = ?, status = ?, updated_at = now()
                    WHERE id = ? AND store_id = ? AND deleted = 0
                    """, keyName, writeJson(normalizedPermissions), status, keyId, store.storeId());
        } else {
            jdbcTemplate.update("""
                    UPDATE bo_store_api_keys
                    SET key_name = ?, key_hash = ?, permissions_json = ?, status = ?, updated_at = now()
                    WHERE id = ? AND store_id = ? AND deleted = 0
                    """, keyName, hash(request.rawKey().trim()), writeJson(normalizedPermissions),
                    status, keyId, store.storeId());
        }
        return getStoreApiKey(store, keyId);
    }

    @Transactional
    public Map<String, Object> deleteStoreApiKey(String storeKey, Long keyId) {
        StoreConnection store = loadStoreConnection(storeKey);
        int updated = jdbcTemplate.update("""
                UPDATE bo_store_api_keys
                SET deleted = 1, status = 0, updated_at = now()
                WHERE id = ? AND store_id = ? AND deleted = 0
                """, keyId, store.storeId());
        if (updated == 0) {
            throw ApiException.notFound("store API key not found: " + keyId);
        }
        return Map.of("deleted", true, "id", keyId);
    }

    public List<ObjectDefinitionResponse> listObjects(BoAuthContext auth) {
        List<Long> readableStores = auth.grants().stream()
                .filter(BoGrant::canRead)
                .map(BoGrant::storeId)
                .distinct()
                .toList();
        if (readableStores.isEmpty()) {
            return List.of();
        }
        String placeholders = readableStores.stream().map(ignored -> "?").collect(Collectors.joining(","));
        List<Object> args = new ArrayList<>();
        args.addAll(readableStores);
        return jdbcTemplate.query("""
                SELECT d.id, d.store_id, s.store_key, d.object_key, d.table_name,
                       d.display_name, d.description, d.schema_json, d.status
                FROM bo_object_definitions d
                JOIN bo_stores s ON s.id = d.store_id
                WHERE d.deleted = 0 AND d.store_id IN (%s)
                ORDER BY s.store_key, d.object_key
                """.formatted(placeholders), (rs, rowNum) -> definitionFromRow(rs.getLong("id"),
                rs.getLong("store_id"),
                rs.getString("store_key"),
                rs.getString("object_key"),
                rs.getString("table_name"),
                rs.getString("display_name"),
                rs.getString("description"),
                rs.getString("schema_json"),
                rs.getInt("status")), args.toArray());
    }

    @Transactional
    public ObjectDefinitionResponse createObject(BoAuthContext auth, ObjectDefinitionRequest request) {
        BoGrant targetGrant = resolveTargetGrant(auth, request.storeKey());
        StoreConnection connection = loadStoreConnection(targetGrant.storeId());
        requireGrant(auth, connection.storeId(), Grant.MANAGE);
        String objectKey = BusinessObjectSqlSupport.requireIdentifier(request.objectKey(), "objectKey");
        String tableName = BusinessObjectSqlSupport.tableNameFor(objectKey);
        List<FieldDefinition> fields = BusinessObjectSqlSupport.normalizeFields(request.fields());
        List<IndexDefinition> indexes = BusinessObjectSqlSupport.normalizeIndexes(request.indexes(), fields);
        ObjectSchema schema = new ObjectSchema(fields, indexes);

        JdbcTemplate store = storeRuntime(connection.storeId()).jdbc();
        List<String> ddl = new ArrayList<>();
        ddl.add(BusinessObjectSqlSupport.createTableSql(tableName, fields));
        ddl.addAll(BusinessObjectSqlSupport.createIndexSql(tableName, objectKey, indexes));
        executeDdl(connection.storeId(), objectKey, store, ddl);

        Long id = jdbcTemplate.queryForObject("""
                INSERT INTO bo_object_definitions
                    (store_id, object_key, table_name, display_name, description, schema_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """, Long.class, connection.storeId(), objectKey, tableName,
                isBlank(request.displayName()) ? objectKey : request.displayName().trim(),
                request.description(), writeJson(schema), request.status() == null ? 1 : request.status());
        return getObjectById(id);
    }

    public ObjectDefinitionResponse getObject(BoAuthContext auth, String objectKey) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.READ);
        return definition.toResponse();
    }

    @Transactional
    public ObjectDefinitionResponse updateObject(BoAuthContext auth, String objectKey, ObjectDefinitionRequest request) {
        ObjectDefinition existing = loadDefinition(auth, objectKey);
        requireGrant(auth, existing.storeId(), Grant.MANAGE);
        String requestedStore = isBlank(request.storeKey()) ? existing.storeKey()
                : BusinessObjectSqlSupport.requireIdentifier(request.storeKey(), "storeKey");
        if (!requestedStore.equals(existing.storeKey())) {
            throw ApiException.badRequest("storeKey cannot be changed after object creation");
        }
        List<FieldDefinition> fields = BusinessObjectSqlSupport.normalizeFields(request.fields());
        List<IndexDefinition> indexes = BusinessObjectSqlSupport.normalizeIndexes(request.indexes(), fields);
        List<String> ddl = new ArrayList<>(BusinessObjectSqlSupport.alterTableSql(
                existing.tableName(), existing.schema().fields(), fields));
        ddl.addAll(BusinessObjectSqlSupport.createIndexSql(existing.tableName(), existing.objectKey(), indexes));
        executeDdl(existing.storeId(), existing.objectKey(), storeRuntime(existing.storeId()).jdbc(), ddl);

        ObjectSchema schema = new ObjectSchema(fields, indexes);
        jdbcTemplate.update("""
                UPDATE bo_object_definitions
                   SET display_name = ?, description = ?, schema_json = ?, status = ?, updated_at = now()
                 WHERE id = ? AND deleted = 0
                """, isBlank(request.displayName()) ? existing.displayName() : request.displayName().trim(),
                request.description(), writeJson(schema), request.status() == null ? existing.status() : request.status(),
                existing.id());
        return getObject(auth, existing.objectKey());
    }

    @Transactional
    public Map<String, Object> deleteObject(BoAuthContext auth, String objectKey) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.MANAGE);
        int updated = jdbcTemplate.update("""
                UPDATE bo_object_definitions
                   SET deleted = 1, updated_at = now()
                 WHERE id = ? AND deleted = 0
                """, definition.id());
        return Map.of("deleted", updated > 0, "objectKey", definition.objectKey());
    }

    public RecordResponse createRecord(BoAuthContext auth, String objectKey, RecordRequest request) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.WRITE);
        Map<String, Object> data = request == null || request.data() == null ? Map.of() : request.data();
        validateRecord(definition.schema(), data, true);
        String id = java.util.UUID.randomUUID().toString().replace("-", "");
        DSLContext dsl = storeRuntime(definition.storeId()).dsl();
        Table<?> table = table(definition);
        Map<Field<?>, Object> values = new LinkedHashMap<>();
        values.put(field("id", String.class), id);
        for (FieldDefinition field : definition.schema().fields()) {
            if (!data.containsKey(field.key())) {
                continue;
            }
            values.put(runtimeField(field), jooqValue(field, data.get(field.key())));
        }
        dsl.insertInto(table).set(values).execute();
        return getRecord(auth, objectKey, id);
    }

    public RecordResponse getRecord(BoAuthContext auth, String objectKey, String id) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.READ);
        BusinessObjectSqlSupport.requireIdentifier(definition.tableName(), "tableName");
        Map<String, Object> row = storeRuntime(definition.storeId()).dsl()
                .selectFrom(table(definition))
                .where(idField().eq(id).and(deletedField().eq(0)))
                .fetchOneMap();
        if (row == null) {
            throw ApiException.notFound("record not found");
        }
        return toRecord(definition, row);
    }

    public RecordResponse updateRecord(BoAuthContext auth, String objectKey, String id, RecordRequest request) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.WRITE);
        Map<String, Object> data = request == null || request.data() == null ? Map.of() : request.data();
        validateRecord(definition.schema(), data, false);
        if (data.isEmpty()) {
            return getRecord(auth, objectKey, id);
        }
        Map<Field<?>, Object> assignments = new LinkedHashMap<>();
        Map<String, FieldDefinition> fields = BusinessObjectSqlSupport.byKey(definition.schema().fields());
        for (Map.Entry<String, Object> entry : data.entrySet()) {
            FieldDefinition field = fields.get(entry.getKey());
            assignments.put(runtimeField(field), jooqValue(field, entry.getValue()));
        }
        assignments.put(field("updated_at", LocalDateTime.class), LocalDateTime.now());
        int updated = storeRuntime(definition.storeId()).dsl()
                .update(table(definition))
                .set(assignments)
                .where(idField().eq(id).and(deletedField().eq(0)))
                .execute();
        if (updated == 0) {
            throw ApiException.notFound("record not found");
        }
        return getRecord(auth, objectKey, id);
    }

    public Map<String, Object> deleteRecord(BoAuthContext auth, String objectKey, String id) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.WRITE);
        int updated = storeRuntime(definition.storeId()).dsl()
                .update(table(definition))
                .set(deletedField(), 1)
                .set(field("updated_at", LocalDateTime.class), LocalDateTime.now())
                .where(idField().eq(id).and(deletedField().eq(0)))
                .execute();
        return Map.of("deleted", updated > 0, "id", id);
    }

    /**
     * Create many records atomically. All records are validated against the object
     * schema first; any validation or insert failure rolls the whole batch back.
     * Runs in a transaction on the store database (the platform metadata
     * transaction does not cover the per-store datasource).
     */
    public BatchCreateResponse createRecords(BoAuthContext auth, String objectKey, BatchRecordRequest request) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.WRITE);
        List<Map<String, Object>> records = request == null || request.records() == null
                ? List.of()
                : request.records();
        if (records.isEmpty()) {
            throw ApiException.badRequest("records is required");
        }
        if (records.size() > MAX_BATCH_RECORDS) {
            throw ApiException.badRequest("records exceeds the maximum batch size of " + MAX_BATCH_RECORDS);
        }
        List<Map<String, Object>> payloads = new ArrayList<>(records.size());
        for (Map<String, Object> record : records) {
            Map<String, Object> data = record == null ? Map.of() : record;
            validateRecord(definition.schema(), data, true);
            payloads.add(data);
        }
        Table<?> table = table(definition);
        List<String> ids = storeRuntime(definition.storeId()).dsl().transactionResult(configuration -> {
            DSLContext tx = DSL.using(configuration);
            List<String> created = new ArrayList<>(payloads.size());
            for (Map<String, Object> data : payloads) {
                String id = java.util.UUID.randomUUID().toString().replace("-", "");
                Map<Field<?>, Object> values = new LinkedHashMap<>();
                values.put(field("id", String.class), id);
                for (FieldDefinition field : definition.schema().fields()) {
                    if (!data.containsKey(field.key())) {
                        continue;
                    }
                    values.put(runtimeField(field), jooqValue(field, data.get(field.key())));
                }
                tx.insertInto(table).set(values).execute();
                created.add(id);
            }
            return created;
        });
        return new BatchCreateResponse(objectKey, ids.size(), ids);
    }

    /**
     * Soft-delete many records by id atomically. Idempotent: ids that do not exist
     * (or were already deleted) are ignored, not an error; {@code deleted} counts
     * the rows actually updated. Runs in a transaction on the store database.
     */
    public BatchDeleteResponse deleteRecords(BoAuthContext auth, String objectKey, BatchDeleteRequest request) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.WRITE);
        List<String> requested = new ArrayList<>();
        if (request != null && request.ids() != null) {
            for (String id : request.ids()) {
                if (id == null || id.isBlank()) {
                    continue;
                }
                String normalized = id.trim();
                if (!requested.contains(normalized)) {
                    requested.add(normalized);
                }
            }
        }
        if (requested.isEmpty()) {
            throw ApiException.badRequest("ids is required");
        }
        if (requested.size() > MAX_BATCH_RECORDS) {
            throw ApiException.badRequest("ids exceeds the maximum batch size of " + MAX_BATCH_RECORDS);
        }
        Table<?> table = table(definition);
        List<String> deleted = storeRuntime(definition.storeId()).dsl().transactionResult(configuration -> {
            DSLContext tx = DSL.using(configuration);
            List<String> removed = new ArrayList<>(requested.size());
            for (String id : requested) {
                int updated = tx.update(table)
                        .set(deletedField(), 1)
                        .set(field("updated_at", LocalDateTime.class), LocalDateTime.now())
                        .where(idField().eq(id).and(deletedField().eq(0)))
                        .execute();
                if (updated > 0) {
                    removed.add(id);
                }
            }
            return removed;
        });
        return new BatchDeleteResponse(objectKey, deleted.size(), deleted);
    }

    public QueryResponse queryRecords(BoAuthContext auth, String objectKey, QueryRequest request) {
        ObjectDefinition definition = loadDefinition(auth, objectKey);
        requireGrant(auth, definition.storeId(), Grant.READ);
        RuntimeQuery query = buildQuery(definition, request);
        DSLContext dsl = storeRuntime(definition.storeId()).dsl();
        Long total = dsl.selectCount()
                .from(table(definition))
                .where(query.condition())
                .fetchOne(0, Long.class);
        List<Map<String, Object>> rows = dsl.selectFrom(table(definition))
                .where(query.condition())
                .orderBy(query.sortFields())
                .limit(query.pageSize())
                .offset((query.page() - 1) * query.pageSize())
                .fetchMaps();
        return new QueryResponse(objectKey, query.page(), query.pageSize(), total == null ? 0L : total,
                rows.stream().map(this::cleanRow).toList());
    }

    private RuntimeQuery buildQuery(ObjectDefinition definition, QueryRequest request) {
        int page = request == null || request.page() == null ? 1 : Math.max(1, request.page());
        int pageSize = request == null || request.pageSize() == null
                ? DEFAULT_PAGE_SIZE
                : Math.min(MAX_PAGE_SIZE, Math.max(1, request.pageSize()));
        Map<String, FieldDefinition> fields = BusinessObjectSqlSupport.byKey(definition.schema().fields());
        Condition condition = deletedField().eq(0);
        List<Filter> filters = request == null || request.filters() == null ? List.of() : request.filters();
        for (Filter filter : filters) {
            String field = allowedQueryField(filter.field(), fields);
            FieldDefinition fieldDefinition = fields.get(field);
            String op = filter.op() == null ? "eq" : filter.op().trim().toLowerCase();
            switch (op) {
                case "eq" -> condition = condition.and(conditionEq(field, fieldDefinition, filter.value()));
                case "ne" -> condition = condition.and(conditionNe(field, fieldDefinition, filter.value()));
                case "gt", "gte", "lt", "lte" -> {
                    if (fieldDefinition != null && BusinessObjectSqlSupport.isJsonField(fieldDefinition)) {
                        throw ApiException.badRequest("range filters are not supported for json fields: " + field);
                    }
                    condition = condition.and(conditionCompare(field, op, filter.value()));
                }
                case "contains" -> {
                    if (fieldDefinition != null && !List.of("string", "text").contains(fieldDefinition.type())) {
                        throw ApiException.badRequest("contains filter is only supported for string/text fields: " + field);
                    }
                    condition = condition.and(field(field, String.class).containsIgnoreCase(String.valueOf(filter.value())));
                }
                case "in" -> {
                    if (fieldDefinition != null && BusinessObjectSqlSupport.isJsonField(fieldDefinition)) {
                        throw ApiException.badRequest("in filters are not supported for json fields: " + field);
                    }
                    if (!(filter.value() instanceof List<?> list) || list.isEmpty()) {
                        throw ApiException.badRequest("in filter value must be a non-empty array");
                    }
                    condition = condition.and(field(field, Object.class).in(list));
                }
                default -> throw ApiException.badRequest("unsupported filter op: " + filter.op());
            }
        }
        return new RuntimeQuery(condition, buildOrder(request == null ? null : request.sort(), fields), page, pageSize);
    }

    private List<SortField<?>> buildOrder(List<Sort> sorts, Map<String, FieldDefinition> fields) {
        if (sorts == null || sorts.isEmpty()) {
            return List.of(field("created_at", LocalDateTime.class).desc());
        }
        return sorts.stream().map(sort -> {
            String field = allowedQueryField(sort.field(), fields);
            Field<Object> sortField = field(field, Object.class);
            return "desc".equalsIgnoreCase(sort.direction()) ? sortField.desc() : sortField.asc();
        }).collect(Collectors.toList());
    }

    private String allowedQueryField(String field, Map<String, FieldDefinition> fields) {
        String key = BusinessObjectSqlSupport.requireIdentifier(field, "query field");
        if (!fields.containsKey(key) && !List.of("id", "created_at", "updated_at").contains(key)) {
            throw ApiException.badRequest("unknown query field: " + key);
        }
        return key;
    }

    private Condition conditionEq(String fieldName, FieldDefinition fieldDefinition, Object value) {
        if (fieldDefinition != null && BusinessObjectSqlSupport.isJsonField(fieldDefinition)) {
            return DSL.field(DSL.name(fieldName), SQLDataType.JSONB)
                    .eq((JSONB) jooqValue(fieldDefinition, value));
        }
        return field(fieldName, Object.class).eq(value);
    }

    private Condition conditionNe(String fieldName, FieldDefinition fieldDefinition, Object value) {
        if (fieldDefinition != null && BusinessObjectSqlSupport.isJsonField(fieldDefinition)) {
            return DSL.field(DSL.name(fieldName), SQLDataType.JSONB)
                    .ne((JSONB) jooqValue(fieldDefinition, value));
        }
        return field(fieldName, Object.class).ne(value);
    }

    private Condition conditionCompare(String fieldName, String op, Object value) {
        Field<Object> runtimeField = field(fieldName, Object.class);
        return switch (op) {
            case "gt" -> DSL.condition("{0} > {1}", runtimeField, DSL.val(value));
            case "gte" -> DSL.condition("{0} >= {1}", runtimeField, DSL.val(value));
            case "lt" -> DSL.condition("{0} < {1}", runtimeField, DSL.val(value));
            case "lte" -> DSL.condition("{0} <= {1}", runtimeField, DSL.val(value));
            default -> throw ApiException.badRequest("unsupported filter op: " + op);
        };
    }

    private Table<?> table(ObjectDefinition definition) {
        return DSL.table(DSL.name(definition.tableName()));
    }

    private <T> Field<T> field(String fieldName, Class<T> type) {
        return DSL.field(DSL.name(BusinessObjectSqlSupport.requireIdentifier(fieldName, "fieldName")), type);
    }

    private Field<?> runtimeField(FieldDefinition field) {
        BusinessObjectSqlSupport.requireIdentifier(field.key(), "field.key");
        if (BusinessObjectSqlSupport.isJsonField(field)) {
            return DSL.field(DSL.name(field.key()), SQLDataType.JSONB);
        }
        return field(field.key(), Object.class);
    }

    private Field<String> idField() {
        return field("id", String.class);
    }

    private Field<Integer> deletedField() {
        return field("deleted", Integer.class);
    }

    private void validateRecord(ObjectSchema schema, Map<String, Object> data, boolean create) {
        Map<String, FieldDefinition> fields = BusinessObjectSqlSupport.byKey(schema.fields());
        for (String key : data.keySet()) {
            BusinessObjectSqlSupport.requireIdentifier(key, "record field");
            if (!fields.containsKey(key)) {
                throw ApiException.badRequest("unknown record field: " + key);
            }
        }
        if (create) {
            for (FieldDefinition field : schema.fields()) {
                if (Boolean.TRUE.equals(field.required()) && (!data.containsKey(field.key()) || data.get(field.key()) == null)) {
                    throw ApiException.badRequest("required field is missing: " + field.key());
                }
            }
        }
    }

    private Object jooqValue(FieldDefinition field, Object value) {
        if (value == null) {
            return null;
        }
        if (BusinessObjectSqlSupport.isJsonField(field)) {
            return JSONB.valueOf(writeJson(value));
        }
        return value;
    }

    private Map<String, Object> cleanRow(Map<String, Object> row) {
        Map<String, Object> clean = new LinkedHashMap<>(row);
        clean.remove("deleted");
        clean.replaceAll((key, value) -> normalizeJdbcValue(value));
        return clean;
    }

    private Object normalizeJdbcValue(Object value) {
        if (value == null) {
            return null;
        }
        if ("org.postgresql.util.PGobject".equals(value.getClass().getName())) {
            try {
                String type = String.valueOf(value.getClass().getMethod("getType").invoke(value));
                String text = (String) value.getClass().getMethod("getValue").invoke(value);
                if ("json".equalsIgnoreCase(type) || "jsonb".equalsIgnoreCase(type)) {
                    return objectMapper.readValue(text, Object.class);
                }
                return text;
            } catch (Exception e) {
                return String.valueOf(value);
            }
        }
        if (value instanceof JSONB jsonb) {
            try {
                return objectMapper.readValue(jsonb.data(), Object.class);
            } catch (Exception e) {
                return jsonb.data();
            }
        }
        return value;
    }

    private RecordResponse toRecord(ObjectDefinition definition, Map<String, Object> row) {
        Map<String, Object> data = cleanRow(row);
        String id = String.valueOf(data.remove("id"));
        return new RecordResponse(id, definition.objectKey(), data);
    }

    private ObjectDefinition loadDefinition(BoAuthContext auth, String objectKey) {
        String normalizedObjectKey = BusinessObjectSqlSupport.requireIdentifier(objectKey, "objectKey");
        List<Long> grantedStoreIds = auth.grants().stream()
                .map(BoGrant::storeId)
                .distinct()
                .toList();
        if (grantedStoreIds.isEmpty()) {
            throw ApiException.notFound("business object not found: " + objectKey);
        }
        String placeholders = grantedStoreIds.stream().map(ignored -> "?").collect(Collectors.joining(","));
        List<Object> args = new ArrayList<>();
        args.add(normalizedObjectKey);
        args.addAll(grantedStoreIds);
        List<ObjectDefinition> rows = jdbcTemplate.query("""
                SELECT d.id, d.store_id, s.store_key, d.object_key, d.table_name,
                       d.display_name, d.description, d.schema_json, d.status
                FROM bo_object_definitions d
                JOIN bo_stores s ON s.id = d.store_id
                WHERE d.object_key = ? AND d.store_id IN (%s) AND d.deleted = 0 AND d.status = 1
                """.formatted(placeholders), (rs, rowNum) -> new ObjectDefinition(
                rs.getLong("id"),
                rs.getLong("store_id"),
                rs.getString("store_key"),
                rs.getString("object_key"),
                rs.getString("table_name"),
                rs.getString("display_name"),
                rs.getString("description"),
                readSchema(rs.getString("schema_json")),
                rs.getInt("status")
        ), args.toArray());
        if (rows.isEmpty()) {
            throw ApiException.notFound("business object not found: " + objectKey);
        }
        if (rows.size() > 1) {
            throw ApiException.conflict("BO_OBJECT_AMBIGUOUS",
                    "business object key is available in multiple granted stores: " + objectKey);
        }
        return rows.get(0);
    }

    private ObjectDefinitionResponse getObjectById(Long id) {
        List<ObjectDefinitionResponse> rows = jdbcTemplate.query("""
                SELECT d.id, d.store_id, s.store_key, d.object_key, d.table_name,
                       d.display_name, d.description, d.schema_json, d.status
                FROM bo_object_definitions d
                JOIN bo_stores s ON s.id = d.store_id
                WHERE d.id = ?
                """, (rs, rowNum) -> definitionFromRow(rs.getLong("id"),
                rs.getLong("store_id"),
                rs.getString("store_key"),
                rs.getString("object_key"),
                rs.getString("table_name"),
                rs.getString("display_name"),
                rs.getString("description"),
                rs.getString("schema_json"),
                rs.getInt("status")), id);
        if (rows.isEmpty()) {
            throw ApiException.notFound("business object not found");
        }
        return rows.get(0);
    }

    private ObjectDefinitionResponse definitionFromRow(Long id, Long storeId, String storeKey,
                                                       String objectKey, String tableName,
                                                       String displayName, String description,
                                                       String schemaJson, Integer status) {
        ObjectSchema schema = readSchema(schemaJson);
        return new ObjectDefinitionResponse(id, storeId, storeKey, objectKey, tableName,
                displayName, description, schema.fields(), schema.indexes(), status);
    }

    private StoreResponse getStoreById(Long id) {
        List<StoreResponse> rows = jdbcTemplate.query("""
                SELECT id, store_key, name, description, jdbc_url, schema_name, username, status
                FROM bo_stores
                WHERE id = ? AND deleted = 0
                """, (rs, rowNum) -> new StoreResponse(
                rs.getLong("id"),
                rs.getString("store_key"),
                rs.getString("name"),
                rs.getString("description"),
                rs.getString("jdbc_url"),
                rs.getString("schema_name"),
                rs.getString("username"),
                rs.getInt("status")
        ), id);
        if (rows.isEmpty()) {
            throw ApiException.notFound("business object store not found");
        }
        return rows.get(0);
    }

    private StoreApiKeyResponse getStoreApiKey(StoreConnection store, Long keyId) {
        List<StoreApiKeyResponse> rows = jdbcTemplate.query("""
                SELECT id, store_id, key_name, permissions_json, status, created_at, updated_at
                FROM bo_store_api_keys
                WHERE id = ? AND store_id = ? AND deleted = 0
                """, (rs, rowNum) -> new StoreApiKeyResponse(
                rs.getLong("id"),
                rs.getLong("store_id"),
                store.storeKey(),
                rs.getString("key_name"),
                permissions(rs.getString("permissions_json")).stream().sorted().toList(),
                rs.getInt("status"),
                rs.getTimestamp("created_at").toLocalDateTime(),
                rs.getTimestamp("updated_at").toLocalDateTime(),
                null
        ), keyId, store.storeId());
        if (rows.isEmpty()) {
            throw ApiException.notFound("store API key not found: " + keyId);
        }
        return rows.get(0);
    }

    private StoreConnection loadStoreConnection(String storeKey) {
        String normalized = BusinessObjectSqlSupport.requireIdentifier(storeKey, "storeKey");
        List<StoreConnection> rows = jdbcTemplate.query("""
                SELECT id, store_key, jdbc_url, schema_name, username, password
                FROM bo_stores
                WHERE store_key = ? AND deleted = 0 AND status = 1
                """, (rs, rowNum) -> new StoreConnection(
                rs.getLong("id"),
                rs.getString("store_key"),
                rs.getString("jdbc_url"),
                rs.getString("schema_name"),
                rs.getString("username"),
                rs.getString("password")
        ), normalized);
        if (rows.isEmpty()) {
            throw ApiException.notFound("business object store not found: " + storeKey);
        }
        return rows.get(0);
    }

    private StoreConnection loadStoreConnection(Long storeId) {
        List<StoreConnection> rows = jdbcTemplate.query("""
                SELECT id, store_key, jdbc_url, schema_name, username, password
                FROM bo_stores
                WHERE id = ? AND deleted = 0 AND status = 1
                """, (rs, rowNum) -> new StoreConnection(
                rs.getLong("id"),
                rs.getString("store_key"),
                rs.getString("jdbc_url"),
                rs.getString("schema_name"),
                rs.getString("username"),
                rs.getString("password")
        ), storeId);
        if (rows.isEmpty()) {
            throw ApiException.notFound("business object store not found: " + storeId);
        }
        return rows.get(0);
    }

    private StoreRuntime storeRuntime(Long storeId) {
        return storeRuntimes.computeIfAbsent("store:" + storeId, key -> {
            StoreConnection connection = loadStoreConnection(storeId);
            HikariConfig config = new HikariConfig();
            config.setPoolName("bo-store-" + connection.storeKey());
            config.setJdbcUrl(connection.jdbcUrl());
            if (!isBlank(connection.schemaName())) {
                config.setSchema(connection.schemaName());
            }
            config.setUsername(connection.username());
            config.setPassword(connection.password());
            config.setMaximumPoolSize(5);
            config.setMinimumIdle(0);
            DataSource dataSource = new HikariDataSource(config);
            return new StoreRuntime(dataSource, new JdbcTemplate(dataSource), DSL.using(dataSource, SQLDialect.POSTGRES));
        });
    }

    private void requireGrant(BoAuthContext auth, Long storeId, Grant grant) {
        boolean allowed = auth.grants().stream()
                .filter(row -> storeId.equals(row.storeId()))
                .anyMatch(row -> switch (grant) {
                    case READ -> row.canRead();
                    case WRITE -> row.canWrite();
                    case MANAGE -> row.canManage();
                });
        if (!allowed) {
            throw new ApiException(403, "BO_STORE_FORBIDDEN", "business object store is not granted: " + storeId);
        }
    }

    private BoGrant singleGrant(BoAuthContext auth) {
        if (auth.grants().isEmpty()) {
            throw new ApiException(403, "BO_STORE_FORBIDDEN", "business object store is not granted");
        }
        return auth.grants().get(0);
    }

    private BoGrant resolveTargetGrant(BoAuthContext auth, String requestedStoreKey) {
        if (isBlank(requestedStoreKey)) {
            return singleGrant(auth);
        }
        String normalized = BusinessObjectSqlSupport.requireIdentifier(requestedStoreKey, "storeKey");
        return auth.grants().stream()
                .filter(grant -> normalized.equals(grant.storeKey()))
                .findFirst()
                .orElseThrow(() -> new ApiException(403, "BO_STORE_FORBIDDEN",
                        "BO store key is not authorized for storeKey=" + normalized));
    }

    private void executeDdl(Long storeId, String objectKey, JdbcTemplate store, List<String> ddl) {
        for (String sql : ddl) {
            try {
                store.execute(sql);
                jdbcTemplate.update("""
                        INSERT INTO bo_object_ddl_history (store_id, object_key, ddl_sql, status)
                        VALUES (?, ?, ?, 'SUCCESS')
                        """, storeId, objectKey, sql);
            } catch (Exception e) {
                jdbcTemplate.update("""
                        INSERT INTO bo_object_ddl_history (store_id, object_key, ddl_sql, status, message)
                        VALUES (?, ?, ?, 'FAILED', ?)
                        """, storeId, objectKey, sql, e.getMessage());
                throw e;
            }
        }
    }

    private String writeJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            throw ApiException.badRequest("invalid JSON value: " + e.getMessage());
        }
    }

    private Set<String> permissions(String permissionsJson) {
        try {
            return objectMapper.readValue(permissionsJson, new TypeReference<List<String>>() {
                    }).stream()
                    .filter(permission -> !isBlank(permission))
                    .map(permission -> permission.trim().toUpperCase())
                    .collect(Collectors.toSet());
        } catch (Exception e) {
            throw new ApiException(500, "BO_KEY_PERMISSIONS_INVALID",
                    "stored BO key permissions are invalid: " + e.getMessage());
        }
    }

    private List<String> normalizePermissions(List<String> requested) {
        Set<String> allowed = Arrays.stream(Grant.values())
                .map(Enum::name)
                .collect(Collectors.toSet());
        List<String> normalized = requested == null || requested.isEmpty()
                ? Arrays.stream(Grant.values()).map(Enum::name).toList()
                : requested.stream()
                .filter(permission -> !isBlank(permission))
                .map(permission -> permission.trim().toUpperCase())
                .distinct()
                .toList();
        for (String permission : normalized) {
            if (!allowed.contains(permission)) {
                throw ApiException.badRequest("unsupported BO store key permission: " + permission);
            }
        }
        return normalized;
    }

    private String generateKey() {
        byte[] bytes = new byte[24];
        RANDOM.nextBytes(bytes);
        return "bos_" + HexFormat.of().formatHex(bytes);
    }

    static String hash(String rawKey) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(rawKey.getBytes(StandardCharsets.UTF_8));
            StringBuilder out = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                out.append(String.format("%02x", b));
            }
            return out.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }

    private ObjectSchema readSchema(String json) {
        try {
            return objectMapper.readValue(json, new TypeReference<>() {
            });
        } catch (Exception e) {
            throw new ApiException(500, "BO_SCHEMA_INVALID", "stored schema is invalid: " + e.getMessage());
        }
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private String headerIgnoreCase(HttpServletRequest request, String name) {
        String exact = request.getHeader(name);
        if (exact != null) {
            return exact;
        }
        java.util.Enumeration<String> names = request.getHeaderNames();
        while (names.hasMoreElements()) {
            String candidate = names.nextElement();
            if (name.equalsIgnoreCase(candidate)) {
                return request.getHeader(candidate);
            }
        }
        return null;
    }

    private enum Grant {
        READ, WRITE, MANAGE
    }

    private record StoreConnection(Long storeId, String storeKey, String jdbcUrl,
                                   String schemaName, String username, String password) {
    }

    private record StoreRuntime(DataSource dataSource, JdbcTemplate jdbc, DSLContext dsl) {
    }

    public record BoAuthContext(String connectionKey, List<BoGrant> grants) {
    }

    public record BoGrant(Long storeId, String storeKey, String keyName,
                          boolean canRead, boolean canWrite, boolean canManage) {
    }

    private record ObjectSchema(List<FieldDefinition> fields, List<IndexDefinition> indexes) {
        private ObjectSchema {
            fields = fields == null ? List.of() : List.copyOf(fields);
            indexes = indexes == null ? List.of() : List.copyOf(indexes);
        }
    }

    private record ObjectDefinition(Long id, Long storeId, String storeKey, String objectKey,
                                    String tableName, String displayName, String description,
                                    ObjectSchema schema, Integer status) {
        ObjectDefinitionResponse toResponse() {
            return new ObjectDefinitionResponse(id, storeId, storeKey, objectKey, tableName,
                    displayName, description, schema.fields(), schema.indexes(), status);
        }
    }

    private record RuntimeQuery(Condition condition, List<SortField<?>> sortFields,
                                Integer page, Integer pageSize) {
    }
}
