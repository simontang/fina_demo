package com.fina.platform.bo;

import com.fina.platform.bo.BusinessObjectDtos.FieldDefinition;
import com.fina.platform.bo.BusinessObjectDtos.IndexDefinition;
import com.fina.platform.exception.ApiException;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

final class BusinessObjectSqlSupport {
    private static final Pattern IDENTIFIER = Pattern.compile("^[a-z][a-z0-9_]{0,62}$");
    private static final Set<String> BASE_COLUMNS = Set.of("id", "created_at", "updated_at", "deleted");
    private static final Set<String> FIELD_TYPES = Set.of(
            "string", "text", "integer", "long", "decimal", "boolean", "date", "datetime", "json");

    private BusinessObjectSqlSupport() {
    }

    static String requireIdentifier(String value, String label) {
        if (value == null || !IDENTIFIER.matcher(value.trim()).matches()) {
            throw ApiException.badRequest(label + " must match " + IDENTIFIER.pattern());
        }
        return value.trim();
    }

    static String tableNameFor(String objectKey) {
        String key = requireIdentifier(objectKey, "objectKey");
        String table = "bo_" + key;
        if (table.length() > 63) {
            throw ApiException.badRequest("objectKey is too long for a PostgreSQL table name");
        }
        return table;
    }

    static String quote(String identifier) {
        return "\"" + requireIdentifier(identifier, "identifier") + "\"";
    }

    static List<FieldDefinition> normalizeFields(List<FieldDefinition> fields) {
        if (fields == null || fields.isEmpty()) {
            throw ApiException.badRequest("fields is required");
        }
        Map<String, FieldDefinition> normalized = new LinkedHashMap<>();
        for (FieldDefinition field : fields) {
            if (field == null) {
                throw ApiException.badRequest("field cannot be null");
            }
            String key = requireIdentifier(field.key(), "field.key");
            if (BASE_COLUMNS.contains(key)) {
                throw ApiException.badRequest("field.key is reserved: " + key);
            }
            String type = normalizeType(field.type());
            if (normalized.containsKey(key)) {
                throw ApiException.badRequest("duplicate field.key: " + key);
            }
            normalized.put(key, new FieldDefinition(
                    key,
                    type,
                    Boolean.TRUE.equals(field.required()),
                    field.maxLength(),
                    field.precision(),
                    field.scale(),
                    field.description()
            ));
        }
        return List.copyOf(normalized.values());
    }

    static List<IndexDefinition> normalizeIndexes(List<IndexDefinition> indexes, List<FieldDefinition> fields) {
        if (indexes == null) {
            return List.of();
        }
        Set<String> fieldKeys = fields.stream().map(FieldDefinition::key).collect(java.util.stream.Collectors.toSet());
        List<IndexDefinition> normalized = new ArrayList<>();
        for (IndexDefinition index : indexes) {
            if (index == null || index.fields() == null || index.fields().isEmpty()) {
                throw ApiException.badRequest("index.fields is required");
            }
            String name = index.name() == null || index.name().isBlank()
                    ? "idx_" + String.join("_", index.fields())
                    : requireIdentifier(index.name(), "index.name");
            List<String> indexFields = index.fields().stream()
                    .map(field -> requireIdentifier(field, "index.fields"))
                    .peek(field -> {
                        if (!fieldKeys.contains(field)) {
                            throw ApiException.badRequest("index field is not defined: " + field);
                        }
                    })
                    .toList();
            normalized.add(new IndexDefinition(name, indexFields, Boolean.TRUE.equals(index.unique())));
        }
        return List.copyOf(normalized);
    }

    static String createTableSql(String tableName, List<FieldDefinition> fields) {
        StringBuilder sql = new StringBuilder();
        sql.append("CREATE TABLE IF NOT EXISTS ").append(quote(tableName)).append(" (");
        sql.append("\"id\" VARCHAR(64) PRIMARY KEY, ");
        sql.append("\"created_at\" TIMESTAMP NOT NULL DEFAULT now(), ");
        sql.append("\"updated_at\" TIMESTAMP NOT NULL DEFAULT now(), ");
        sql.append("\"deleted\" INT NOT NULL DEFAULT 0");
        for (FieldDefinition field : fields) {
            sql.append(", ").append(quote(field.key())).append(" ").append(sqlType(field));
        }
        sql.append(")");
        return sql.toString();
    }

    static List<String> createIndexSql(String tableName, String objectKey, List<IndexDefinition> indexes) {
        List<String> sql = new ArrayList<>();
        for (IndexDefinition index : indexes) {
            String indexName = "idx_" + objectKey + "_" + index.name();
            if (indexName.length() > 63) {
                indexName = indexName.substring(0, 63);
            }
            String unique = Boolean.TRUE.equals(index.unique()) ? "UNIQUE " : "";
            String columns = index.fields().stream().map(BusinessObjectSqlSupport::quote)
                    .collect(java.util.stream.Collectors.joining(", "));
            sql.add("CREATE " + unique + "INDEX IF NOT EXISTS " + quote(indexName)
                    + " ON " + quote(tableName) + " (" + columns + ")");
        }
        return sql;
    }

    static List<String> alterTableSql(String tableName, List<FieldDefinition> oldFields, List<FieldDefinition> newFields) {
        Map<String, FieldDefinition> oldByKey = byKey(oldFields);
        List<String> sql = new ArrayList<>();
        for (FieldDefinition field : newFields) {
            FieldDefinition old = oldByKey.get(field.key());
            if (old == null) {
                sql.add("ALTER TABLE " + quote(tableName) + " ADD COLUMN IF NOT EXISTS "
                        + quote(field.key()) + " " + sqlType(field));
                continue;
            }
            if (!normalizeType(old.type()).equals(normalizeType(field.type()))
                    || !String.valueOf(old.maxLength()).equals(String.valueOf(field.maxLength()))
                    || !String.valueOf(old.precision()).equals(String.valueOf(field.precision()))
                    || !String.valueOf(old.scale()).equals(String.valueOf(field.scale()))) {
                throw ApiException.badRequest("field type changes are not supported in v1: " + field.key());
            }
        }
        for (String oldKey : oldByKey.keySet()) {
            boolean stillExists = newFields.stream().anyMatch(field -> oldKey.equals(field.key()));
            if (!stillExists) {
                throw ApiException.badRequest("field removal is not supported in v1: " + oldKey);
            }
        }
        return sql;
    }

    static boolean isJsonField(FieldDefinition field) {
        return "json".equals(normalizeType(field.type()));
    }

    static Map<String, FieldDefinition> byKey(List<FieldDefinition> fields) {
        Map<String, FieldDefinition> map = new LinkedHashMap<>();
        for (FieldDefinition field : fields) {
            map.put(field.key(), field);
        }
        return map;
    }

    private static String normalizeType(String type) {
        String normalized = type == null ? "" : type.trim().toLowerCase(Locale.ROOT);
        if (!FIELD_TYPES.contains(normalized)) {
            throw ApiException.badRequest("unsupported field type: " + type);
        }
        return normalized;
    }

    private static String sqlType(FieldDefinition field) {
        return switch (normalizeType(field.type())) {
            case "string" -> "VARCHAR(" + bounded(field.maxLength(), 1, 2000, 255, "maxLength") + ")";
            case "text" -> "TEXT";
            case "integer" -> "INTEGER";
            case "long" -> "BIGINT";
            case "decimal" -> "NUMERIC("
                    + bounded(field.precision(), 1, 38, 18, "precision")
                    + ","
                    + bounded(field.scale(), 0, 18, 2, "scale")
                    + ")";
            case "boolean" -> "BOOLEAN";
            case "date" -> "DATE";
            case "datetime" -> "TIMESTAMP";
            case "json" -> "JSONB";
            default -> throw ApiException.badRequest("unsupported field type: " + field.type());
        };
    }

    private static int bounded(Integer value, int min, int max, int defaultValue, String label) {
        int actual = value == null ? defaultValue : value;
        if (actual < min || actual > max) {
            throw ApiException.badRequest(label + " must be between " + min + " and " + max);
        }
        return actual;
    }
}
