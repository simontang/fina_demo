package com.fina.metrics.service;

import com.fina.metrics.dto.DataSourceTableGrantVO;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.util.SqlIdentifierUtils;
import org.springframework.util.StringUtils;

import java.util.List;
import java.util.Locale;
import java.util.Set;

/** A datasource's base visibility boundary, independent of caller identity or meta publication. */
public record DataSourceVisibleScope(String mode, String defaultSchema, List<DataSourceTableGrantVO> rules) {
    public static final String LEGACY_TENANT_MARKER = "__datasource__";
    private static final Set<String> SYSTEM_SCHEMAS = Set.of(
            "information_schema", "pg_catalog", "sys", "sysibm", "syscat", "_sys_statistics",
            "_sys_bi", "_sys_repo");

    public DataSourceVisibleScope {
        // Migration initializes old configurations explicitly; missing mode must never imply ALL.
        mode = mode == null ? "RESTRICTED" : mode.toUpperCase(Locale.ROOT);
        if (!Set.of("ALL", "RESTRICTED").contains(mode)) {
            throw new IllegalArgumentException("visibleScopeMode must be ALL or RESTRICTED");
        }
        rules = List.copyOf(rules);
    }

    public static DataSourceVisibleScope from(DataSourceConfig config, List<DataSourceTableGrantVO> rules) {
        return new DataSourceVisibleScope(config == null ? null : config.getVisibleScopeMode(),
                config == null ? null : config.getSchemaName(), rules);
    }

    public boolean isRestricted() {
        return "RESTRICTED".equals(mode);
    }

    public boolean allows(String schema, String table) {
        if (!StringUtils.hasText(table)) return false;
        SqlIdentifierUtils.TableIdentifier identifier = SqlIdentifierUtils.parseTableIdentifier(schema, table);
        if (!isRestricted()) return true;
        String effectiveSchema = identifier.schemaName();
        if (!StringUtils.hasText(effectiveSchema)) effectiveSchema = defaultSchema;
        if (isSystemObject(effectiveSchema, identifier.tableName())) return false;
        for (DataSourceTableGrantVO rule : rules) {
            if (!Integer.valueOf(1).equals(rule.getStatus())) continue;
            boolean sensitive = Boolean.TRUE.equals(rule.getCaseSensitive());
            if (StringUtils.hasText(rule.getSchemaName())
                    && !equal(rule.getSchemaName(), effectiveSchema, sensitive)) continue;
            String candidate = SqlIdentifierUtils.normalizeForComparison(identifier.tableName(), sensitive);
            String pattern = SqlIdentifierUtils.normalizeForComparison(rule.getTablePattern(), sensitive);
            if (pattern == null) continue;
            if ("EXACT".equals(rule.getPatternType()) && candidate.equals(pattern)
                    || "PREFIX".equals(rule.getPatternType()) && candidate.startsWith(pattern)) return true;
        }
        return false;
    }

    private static boolean equal(String left, String right, boolean sensitive) {
        return right != null && (sensitive ? left.equals(right) : left.equalsIgnoreCase(right));
    }

    private static boolean isSystemObject(String schema, String table) {
        String normalizedSchema = schema == null ? "" : schema.toLowerCase(Locale.ROOT);
        String normalizedTable = table.toLowerCase(Locale.ROOT);
        return SYSTEM_SCHEMAS.contains(normalizedSchema)
                || normalizedSchema.startsWith("pg_temp") || normalizedSchema.startsWith("pg_toast")
                || normalizedTable.startsWith("pg_") || normalizedTable.startsWith("sys.");
    }
}
