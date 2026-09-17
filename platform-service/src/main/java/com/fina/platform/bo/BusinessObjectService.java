package com.fina.platform.bo;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import com.fina.platform.bo.BusinessObjectDtos.StoreGrantRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreGrantResponse;
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
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class BusinessObjectService {
    private static final int DEFAULT_PAGE_SIZE = 50;
    private static final int MAX_PAGE_SIZE = 500;

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
            throw new ApiException(401, "BO_CONNECTION_KEY_REQUIRED", "X-BO-Connection-Key header is required");
        }
        String granteeKey = BusinessObjectSqlSupport.requireIdentifier(connectionKey.trim(), "connectionKey");
        List<BoGrant> grants = jdbcTemplate.query("""
                SELECT s.id AS store_id, s.store_key, g.grantee_key,
                       g.can_read, g.can_write, g.can_manage
                  FROM bo_store_grants g
                  JOIN bo_stores s
                    ON s.id = g.store_id
                   AND s.status = 1
                   AND s.deleted = 0
                 WHERE g.grantee_key = ?
                   AND g.status = 1
                   AND g.deleted = 0
                """, (rs, rowNum) -> new BoGrant(
                rs.getLong("store_id"),
                rs.getString("store_key"),
                rs.getString("grantee_key"),
                rs.getBoolean("can_read"),
                rs.getBoolean("can_write"),
                rs.getBoolean("can_manage")
        ), granteeKey);
        if (grants.isEmpty()) {
            throw new ApiException(403, "BO_CONNECTION_FORBIDDEN", "BO connection key is not granted");
        }
        return new BoAuthContext(granteeKey, List.copyOf(grants));
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
        jdbcTemplate.update("""
                INSERT INTO bo_store_grants (store_id, grantee_key, can_read, can_write, can_manage, status)
                VALUES (?, 'tenant', true, true, true, 1)
                ON CONFLICT (store_id, grantee_key)
                WHERE deleted = 0
                DO UPDATE SET can_read = true,
                              can_write = true,
                              can_manage = true,
                              status = 1,
                              updated_at = now()
                """, id);
        return getStoreById(id);
    }

    public Map<String, Object> testStore(String storeKey) {
        StoreConnection connection = loadStoreConnection(storeKey);
        JdbcTemplate template = storeRuntime(connection.storeId()).jdbc();
        Integer ok = template.queryForObject("SELECT 1", Integer.class);
        return Map.of("ok", ok != null && ok == 1, "storeKey", storeKey);
    }

    public List<StoreGrantResponse> listStoreGrants(String storeKey) {
        StoreConnection store = loadStoreConnection(storeKey);
        return jdbcTemplate.query("""
                SELECT id, store_id, grantee_key, can_read, can_write, can_manage, status
                FROM bo_store_grants
                WHERE store_id = ? AND deleted = 0
                ORDER BY grantee_key
                """, (rs, rowNum) -> new StoreGrantResponse(
                rs.getLong("id"),
                rs.getLong("store_id"),
                store.storeKey(),
                rs.getString("grantee_key"),
                rs.getBoolean("can_read"),
                rs.getBoolean("can_write"),
                rs.getBoolean("can_manage"),
                rs.getInt("status")
        ), store.storeId());
    }

    @Transactional
    public StoreGrantResponse createOrUpdateStoreGrant(String storeKey, StoreGrantRequest request) {
        StoreConnection store = loadStoreConnection(storeKey);
        String granteeKey = isBlank(request.granteeKey()) ? "tenant"
                : BusinessObjectSqlSupport.requireIdentifier(request.granteeKey(), "granteeKey");
        boolean canRead = request.canRead() == null || request.canRead();
        boolean canWrite = Boolean.TRUE.equals(request.canWrite());
        boolean canManage = Boolean.TRUE.equals(request.canManage());
        int status = request.status() == null ? 1 : request.status();
        Long id = jdbcTemplate.queryForObject("""
                INSERT INTO bo_store_grants (store_id, grantee_key, can_read, can_write, can_manage, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (store_id, grantee_key)
                WHERE deleted = 0
                DO UPDATE SET can_read = EXCLUDED.can_read,
                              can_write = EXCLUDED.can_write,
                              can_manage = EXCLUDED.can_manage,
                              status = EXCLUDED.status,
                              updated_at = now()
                RETURNING id
                """, Long.class, store.storeId(), granteeKey, canRead, canWrite, canManage, status);
        return listStoreGrants(storeKey).stream()
                .filter(grant -> grant.id().equals(id))
                .findFirst()
                .orElseThrow(() -> ApiException.notFound("store grant not found"));
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
        String storeKey = BusinessObjectSqlSupport.requireIdentifier(request.storeKey(), "storeKey");
        StoreConnection connection = loadStoreConnection(storeKey);
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

    public record BoGrant(Long storeId, String storeKey, String granteeKey,
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
