package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.dto.*;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.entity.DataSourceTableGrant;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.mapper.DataSourceTableGrantMapper;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.util.ReadOnlySqlValidator;
import com.fina.metrics.util.SqlTableReferenceExtractor;
import com.fina.metrics.util.TenantHeaderResolver;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.jdbc.core.ResultSetExtractor;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class DataSourceTableAccessServiceImpl implements DataSourceTableAccessService {

    private static final String PATTERN_PREFIX = "PREFIX";
    private static final String PATTERN_EXACT = "EXACT";
    private static final int DEFAULT_PROBE_MAX_ROWS = 100;
    private static final int MAX_PROBE_ROWS = 1000;

    private final DataSourceTableGrantMapper grantMapper;
    private final DataSourceConfigMapper datasourceMapper;
    private final DynamicDataSourceManager dsManager;

    @Override
    public List<DataSourceTableGrantVO> listGrants(String tenantId, Long datasourceId) {
        return selectGrants(tenantId, datasourceId, null).stream()
                .map(this::toVO)
                .collect(Collectors.toList());
    }

    @Override
    public List<DataSourceTableGrantVO> listActiveGrants(String tenantId, Long datasourceId) {
        return selectActiveGrants(tenantId, datasourceId).stream()
                .map(this::toVO)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public DataSourceTableGrantVO createGrant(
            String tenantId,
            Long datasourceId,
            DataSourceTableGrantRequest request) {
        requireDatasource(datasourceId);
        DataSourceTableGrant grant = new DataSourceTableGrant();
        grant.setTenantId(TenantHeaderResolver.resolve(tenantId));
        grant.setDatasourceId(datasourceId);
        applyRequest(grant, request);
        grant.setDeleted(0);
        grantMapper.insert(grant);
        log.info("Created datasource table grant id={} tenant={} datasource={} pattern={}",
                grant.getId(), grant.getTenantId(), datasourceId, grant.getTablePattern());
        return toVO(grant);
    }

    @Override
    @Transactional
    public DataSourceTableGrantVO updateGrant(
            String tenantId,
            Long datasourceId,
            Long grantId,
            DataSourceTableGrantRequest request) {
        DataSourceTableGrant grant = requireGrant(tenantId, datasourceId, grantId);
        applyRequest(grant, request);
        grantMapper.updateById(grant);
        log.info("Updated datasource table grant id={} tenant={} datasource={}",
                grantId, grant.getTenantId(), datasourceId);
        return toVO(grant);
    }

    @Override
    @Transactional
    public void deleteGrant(String tenantId, Long datasourceId, Long grantId) {
        DataSourceTableGrant grant = requireGrant(tenantId, datasourceId, grantId);
        grantMapper.deleteById(grant.getId());
        log.info("Deleted datasource table grant id={} tenant={} datasource={}",
                grantId, grant.getTenantId(), datasourceId);
    }

    @Override
    public boolean hasActiveGrants(String tenantId, Long datasourceId) {
        return !selectActiveGrants(tenantId, datasourceId).isEmpty();
    }

    @Override
    public boolean isTableAuthorized(
            String tenantId,
            Long datasourceId,
            String schemaName,
            String tableName) {
        List<DataSourceTableGrant> grants = selectActiveGrants(tenantId, datasourceId);
        DataSourceConfig datasource = resolveDatasource(datasourceId);
        String effectiveSchema = resolveEffectiveSchema(schemaName, datasource, grants);
        return grants.stream().anyMatch(grant -> matchesGrant(grant, effectiveSchema, tableName));
    }

    @Override
    public boolean isTableAuthorizedIfGrantsConfigured(
            String tenantId,
            Long datasourceId,
            String schemaName,
            String tableName) {
        List<DataSourceTableGrant> grants = selectActiveGrants(tenantId, datasourceId);
        if (grants.isEmpty()) {
            return true;
        }
        DataSourceConfig datasource = resolveDatasource(datasourceId);
        String effectiveSchema = resolveEffectiveSchema(schemaName, datasource, grants);
        return grants.stream().anyMatch(grant -> matchesGrant(grant, effectiveSchema, tableName));
    }

    @Override
    public void assertSqlAuthorized(String tenantId, Long datasourceId, String sql) {
        ReadOnlySqlValidator.validate(sql);
        List<DataSourceTableGrant> grants = selectActiveGrants(tenantId, datasourceId);
        if (grants.isEmpty()) {
            throw new ForbiddenException("No table grants configured for tenant="
                    + TenantHeaderResolver.resolve(tenantId) + " datasourceId=" + datasourceId);
        }
        DataSourceConfig datasource = resolveDatasource(datasourceId);
        Set<SqlTableReferenceExtractor.TableReference> references = SqlTableReferenceExtractor.extract(sql);
        for (SqlTableReferenceExtractor.TableReference reference : references) {
            String effectiveSchema = resolveEffectiveSchema(reference.schemaName(), datasource, grants);
            boolean allowed = grants.stream()
                    .anyMatch(grant -> matchesGrant(grant, effectiveSchema, reference.tableName()));
            if (!allowed) {
                throw new ForbiddenException("SQL references unauthorized table: " + reference.original());
            }
        }
    }

    @Override
    public List<DataSourceTableVO> listPhysicalTables(Long datasourceId, String schemaName) {
        DataSourceConfig datasource = requireDatasource(datasourceId);
        String effectiveSchema = StringUtils.hasText(schemaName) ? schemaName : datasource.getSchemaName();
        return withConnection(datasourceId, connection -> {
            DatabaseMetaData meta = connection.getMetaData();
            String catalog = connection.getCatalog();
            String escape = meta.getSearchStringEscape();
            Set<String> schemaPatterns = candidateMetadataPatterns(effectiveSchema, escape, false, false);
            Map<String, DataSourceTableVO> rows = new LinkedHashMap<>();
            for (String schemaPattern : schemaPatterns) {
                try (ResultSet rs = meta.getTables(catalog, schemaPattern, "%", new String[]{"TABLE", "VIEW"})) {
                    while (rs.next()) {
                        String rowSchema = rs.getString("TABLE_SCHEM");
                        String tableName = rs.getString("TABLE_NAME");
                        String key = normalizeKey(rowSchema, tableName);
                        rows.putIfAbsent(key, DataSourceTableVO.builder()
                                .schemaName(rowSchema)
                                .tableName(tableName)
                                .tableType(rs.getString("TABLE_TYPE"))
                                .remarks(rs.getString("REMARKS"))
                                .build());
                    }
                }
            }
            return new ArrayList<>(rows.values());
        });
    }

    @Override
    public List<DataSourceTableVO> listAuthorizedTables(String tenantId, Long datasourceId) {
        requireDatasource(datasourceId);
        List<DataSourceTableGrant> grants = selectActiveGrants(tenantId, datasourceId);
        if (grants.isEmpty()) {
            return List.of();
        }
        return withConnection(datasourceId, connection -> {
            DatabaseMetaData meta = connection.getMetaData();
            String catalog = connection.getCatalog();
            String escape = meta.getSearchStringEscape();
            Map<String, DataSourceTableVO> rows = new LinkedHashMap<>();
            for (DataSourceTableGrant grant : grants) {
                for (String schemaPattern : candidateMetadataPatterns(grant.getSchemaName(), escape, grant.getCaseSensitive(), false)) {
                    for (String tablePattern : candidateMetadataPatterns(
                            grant.getTablePattern(), escape, grant.getCaseSensitive(), PATTERN_PREFIX.equals(grant.getPatternType()))) {
                        try (ResultSet rs = meta.getTables(catalog, schemaPattern, tablePattern, new String[]{"TABLE", "VIEW"})) {
                            while (rs.next()) {
                                String schemaName = rs.getString("TABLE_SCHEM");
                                String tableName = rs.getString("TABLE_NAME");
                                if (!matchesAnyGrant(grants, schemaName, tableName)) {
                                    continue;
                                }
                                String key = normalizeKey(schemaName, tableName);
                                rows.putIfAbsent(key, DataSourceTableVO.builder()
                                        .schemaName(schemaName)
                                        .tableName(tableName)
                                        .tableType(rs.getString("TABLE_TYPE"))
                                        .remarks(rs.getString("REMARKS"))
                                        .build());
                            }
                        }
                    }
                }
            }
            return new ArrayList<>(rows.values());
        });
    }

    @Override
    public List<DataSourceColumnVO> listAuthorizedColumns(
            String tenantId,
            Long datasourceId,
            String schemaName,
            String tableName) {
        if (!isTableAuthorized(tenantId, datasourceId, schemaName, tableName)) {
            throw new ForbiddenException("Table is not authorized for this tenant: " + tableName);
        }
        List<DataSourceTableGrant> grants = selectActiveGrants(tenantId, datasourceId);
        DataSourceConfig datasource = resolveDatasource(datasourceId);
        String effectiveSchema = resolveEffectiveSchema(schemaName, datasource, grants);

        return withConnection(datasourceId, connection -> {
            DatabaseMetaData meta = connection.getMetaData();
            String catalog = connection.getCatalog();
            String escape = meta.getSearchStringEscape();
            Set<String> schemaPatterns = candidateMetadataPatterns(effectiveSchema, escape, false, false);
            Set<String> tablePatterns = candidateMetadataPatterns(tableName, escape, false, false);
            Map<String, DataSourceColumnVO> rows = new LinkedHashMap<>();
            for (String schemaPattern : schemaPatterns) {
                for (String tablePattern : tablePatterns) {
                    try (ResultSet rs = meta.getColumns(catalog, schemaPattern, tablePattern, "%")) {
                        while (rs.next()) {
                            String rowSchema = rs.getString("TABLE_SCHEM");
                            String rowTable = rs.getString("TABLE_NAME");
                            String columnName = rs.getString("COLUMN_NAME");
                            if (!matchesAnyGrant(grants, rowSchema, rowTable)) {
                                continue;
                            }
                            String key = normalizeKey(rowSchema, rowTable) + "." + columnName;
                            rows.putIfAbsent(key, DataSourceColumnVO.builder()
                                    .schemaName(rowSchema)
                                    .tableName(rowTable)
                                    .columnName(columnName)
                                    .ordinalPosition(rs.getInt("ORDINAL_POSITION"))
                                    .dataType(rs.getInt("DATA_TYPE"))
                                    .typeName(rs.getString("TYPE_NAME"))
                                    .columnSize(rs.getInt("COLUMN_SIZE"))
                                    .nullable(rs.getInt("NULLABLE") == DatabaseMetaData.columnNullable)
                                    .remarks(rs.getString("REMARKS"))
                                    .build());
                        }
                    }
                }
            }
            return new ArrayList<>(rows.values());
        });
    }

    @Override
    public MetricsQueryData queryDatasource(Long datasourceId, SqlProbeRequest request) {
        ReadOnlySqlValidator.validate(request.getSql());
        return executeSql(datasourceId, request, "datasource_query", null);
    }

    @Override
    public MetricsQueryData probeSql(String tenantId, Long datasourceId, SqlProbeRequest request) {
        String sql = request.getSql();
        assertSqlAuthorized(tenantId, datasourceId, sql);
        Set<SqlTableReferenceExtractor.TableReference> references = SqlTableReferenceExtractor.extract(sql);
        Map<String, Object> extraDebug = Boolean.TRUE.equals(request.getDebug())
                ? Map.of("referencedTables", references.stream()
                        .map(SqlTableReferenceExtractor.TableReference::original)
                        .toList())
                : null;
        return executeSql(datasourceId, request, "probe", extraDebug);
    }

    private MetricsQueryData executeSql(
            Long datasourceId,
            SqlProbeRequest request,
            String semanticModel,
            Map<String, Object> extraDebug) {
        int maxRows = resolveProbeMaxRows(request.getMaxRows());
        NamedParameterJdbcTemplate jdbc = dsManager.getNamedJdbcTemplate(datasourceId);
        int previousMaxRows = jdbc.getJdbcTemplate().getMaxRows();
        jdbc.getJdbcTemplate().setMaxRows(maxRows);
        try {
            List<ColumnMeta> columns = new ArrayList<>();
            ResultSetExtractor<List<List<Object>>> extractor = rs -> readRows(rs, columns);
            List<List<Object>> rows = jdbc.query(
                    request.getSql(),
                    new MapSqlParameterSource(request.getParams() != null ? request.getParams() : Map.of()),
                    extractor);
            Map<String, Object> debug = null;
            if (Boolean.TRUE.equals(request.getDebug())) {
                debug = new LinkedHashMap<>();
                debug.put("sql", request.getSql());
                debug.put("params", request.getParams() != null ? request.getParams() : Map.of());
                debug.put("maxRows", maxRows);
                if (extraDebug != null) {
                    debug.putAll(extraDebug);
                }
            }
            return MetricsQueryData.builder()
                    .semanticModel(semanticModel)
                    .columns(columns)
                    .rows(rows != null ? rows : List.of())
                    .rowCount(rows != null ? rows.size() : 0)
                    .debug(debug)
                    .build();
        } finally {
            jdbc.getJdbcTemplate().setMaxRows(previousMaxRows);
        }
    }

    private List<List<Object>> readRows(ResultSet rs, List<ColumnMeta> columns) throws java.sql.SQLException {
        ResultSetMetaData meta = rs.getMetaData();
        int n = meta.getColumnCount();
        for (int i = 1; i <= n; i++) {
            columns.add(ColumnMeta.builder()
                    .name(meta.getColumnLabel(i))
                    .type(mapJdbcTypeToDocType(meta.getColumnTypeName(i)))
                    .build());
        }
        List<List<Object>> rows = new ArrayList<>();
        while (rs.next()) {
            List<Object> row = new ArrayList<>(n);
            for (int i = 1; i <= n; i++) {
                row.add(rs.getObject(i));
            }
            rows.add(row);
        }
        return rows;
    }

    private List<DataSourceTableGrant> selectActiveGrants(String tenantId, Long datasourceId) {
        return selectGrants(tenantId, datasourceId, 1);
    }

    private List<DataSourceTableGrant> selectGrants(String tenantId, Long datasourceId, Integer status) {
        LambdaQueryWrapper<DataSourceTableGrant> wrapper = new LambdaQueryWrapper<DataSourceTableGrant>()
                .eq(DataSourceTableGrant::getTenantId, TenantHeaderResolver.resolve(tenantId))
                .eq(DataSourceTableGrant::getDatasourceId, datasourceId)
                .eq(DataSourceTableGrant::getDeleted, 0)
                .orderByAsc(DataSourceTableGrant::getId);
        if (status != null) {
            wrapper.eq(DataSourceTableGrant::getStatus, status);
        }
        return grantMapper.selectList(wrapper);
    }

    private DataSourceTableGrant requireGrant(String tenantId, Long datasourceId, Long grantId) {
        DataSourceTableGrant grant = grantMapper.selectOne(
                new LambdaQueryWrapper<DataSourceTableGrant>()
                        .eq(DataSourceTableGrant::getId, grantId)
                        .eq(DataSourceTableGrant::getTenantId, TenantHeaderResolver.resolve(tenantId))
                        .eq(DataSourceTableGrant::getDatasourceId, datasourceId)
                        .eq(DataSourceTableGrant::getDeleted, 0)
        );
        if (grant == null) {
            throw new IllegalArgumentException("DataSource table grant not found: id=" + grantId);
        }
        return grant;
    }

    private DataSourceConfig requireDatasource(Long datasourceId) {
        DataSourceConfig datasource = resolveDatasource(datasourceId);
        if (datasource == null) {
            throw new IllegalArgumentException("DataSource not found: id=" + datasourceId);
        }
        return datasource;
    }

    private DataSourceConfig resolveDatasource(Long datasourceId) {
        return datasourceMapper.selectOne(
                new LambdaQueryWrapper<DataSourceConfig>()
                        .eq(DataSourceConfig::getId, datasourceId)
                        .eq(DataSourceConfig::getDeleted, 0)
        );
    }

    private void applyRequest(DataSourceTableGrant grant, DataSourceTableGrantRequest request) {
        String patternType = StringUtils.hasText(request.getPatternType())
                ? request.getPatternType().trim().toUpperCase(Locale.ROOT)
                : PATTERN_PREFIX;
        if (!PATTERN_PREFIX.equals(patternType) && !PATTERN_EXACT.equals(patternType)) {
            throw new IllegalArgumentException("patternType must be PREFIX or EXACT");
        }
        Integer status = request.getStatus() != null ? request.getStatus() : 1;
        if (status != 0 && status != 1) {
            throw new IllegalArgumentException("status must be 0 or 1");
        }
        grant.setSchemaName(trimToNull(request.getSchemaName()));
        grant.setTablePattern(request.getTablePattern().trim());
        grant.setPatternType(patternType);
        grant.setCaseSensitive(Boolean.TRUE.equals(request.getCaseSensitive()));
        grant.setStatus(status);
    }

    private boolean matchesAnyGrant(List<DataSourceTableGrant> grants, String schemaName, String tableName) {
        return grants.stream().anyMatch(grant -> matchesGrant(grant, schemaName, tableName));
    }

    private boolean matchesGrant(DataSourceTableGrant grant, String schemaName, String tableName) {
        if (!StringUtils.hasText(tableName)) {
            return false;
        }
        boolean caseSensitive = Boolean.TRUE.equals(grant.getCaseSensitive());
        if (StringUtils.hasText(grant.getSchemaName())
                && !equalsMaybeCaseSensitive(grant.getSchemaName(), schemaName, caseSensitive)) {
            return false;
        }
        String candidate = caseSensitive ? tableName : tableName.toLowerCase(Locale.ROOT);
        String pattern = caseSensitive ? grant.getTablePattern() : grant.getTablePattern().toLowerCase(Locale.ROOT);
        return PATTERN_EXACT.equals(grant.getPatternType())
                ? candidate.equals(pattern)
                : candidate.startsWith(pattern);
    }

    private boolean equalsMaybeCaseSensitive(String expected, String actual, boolean caseSensitive) {
        if (!StringUtils.hasText(actual)) {
            return false;
        }
        return caseSensitive ? expected.equals(actual) : expected.equalsIgnoreCase(actual);
    }

    private String resolveEffectiveSchema(
            String schemaName,
            DataSourceConfig datasource,
            List<DataSourceTableGrant> grants) {
        if (StringUtils.hasText(schemaName)) {
            return schemaName;
        }
        if (datasource != null && StringUtils.hasText(datasource.getSchemaName())) {
            return datasource.getSchemaName();
        }
        Set<String> schemas = grants.stream()
                .map(DataSourceTableGrant::getSchemaName)
                .filter(StringUtils::hasText)
                .collect(Collectors.toCollection(LinkedHashSet::new));
        return schemas.size() == 1 ? schemas.iterator().next() : null;
    }

    private <T> T withConnection(Long datasourceId, SqlConnectionCallback<T> callback) {
        try {
            DataSource dataSource = Objects.requireNonNull(
                    dsManager.getNamedJdbcTemplate(datasourceId).getJdbcTemplate().getDataSource(),
                    "Datasource is not available");
            try (Connection connection = dataSource.getConnection()) {
                return callback.doWithConnection(connection);
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to inspect datasource metadata: " + e.getMessage(), e);
        }
    }

    private static Set<String> candidateMetadataPatterns(
            String value,
            String escape,
            Boolean caseSensitive,
            boolean prefix) {
        if (!StringUtils.hasText(value)) {
            Set<String> wildcard = new LinkedHashSet<>();
            wildcard.add(null);
            return wildcard;
        }
        Set<String> values = new LinkedHashSet<>();
        values.add(value);
        if (!Boolean.TRUE.equals(caseSensitive)) {
            values.add(value.toLowerCase(Locale.ROOT));
            values.add(value.toUpperCase(Locale.ROOT));
        }
        return values.stream()
                .map(candidate -> metadataPattern(candidate, escape, prefix))
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    private static String metadataPattern(String raw, String escape, boolean prefix) {
        String escaped = escapeMetadataLike(raw, escape);
        return prefix ? escaped + "%" : escaped;
    }

    private static String escapeMetadataLike(String raw, String escape) {
        if (!StringUtils.hasText(escape)) {
            return raw;
        }
        return raw
                .replace(escape, escape + escape)
                .replace("%", escape + "%")
                .replace("_", escape + "_");
    }

    private static int resolveProbeMaxRows(Integer maxRows) {
        if (maxRows == null || maxRows <= 0) {
            return DEFAULT_PROBE_MAX_ROWS;
        }
        return Math.min(maxRows, MAX_PROBE_ROWS);
    }

    private static String normalizeKey(String schemaName, String tableName) {
        return (schemaName == null ? "" : schemaName.toLowerCase(Locale.ROOT))
                + "."
                + (tableName == null ? "" : tableName.toLowerCase(Locale.ROOT));
    }

    private static String trimToNull(String value) {
        return StringUtils.hasText(value) ? value.trim() : null;
    }

    private static String mapJdbcTypeToDocType(String jdbcTypeName) {
        if (jdbcTypeName == null) {
            return "varchar";
        }
        String u = jdbcTypeName.toUpperCase(Locale.ROOT);
        if (u.contains("CHAR") || u.contains("TEXT") || u.contains("STRING")) {
            return "varchar";
        }
        if (u.contains("DECIMAL") || u.contains("NUMERIC") || u.contains("DOUBLE")
                || u.contains("FLOAT") || u.contains("REAL")) {
            return "numeric";
        }
        if (u.contains("INT") || u.contains("LONG") || u.contains("SMALLINT")) {
            return "numeric";
        }
        if (u.contains("DATE") || u.contains("TIME") || u.contains("STAMP")) {
            return "date";
        }
        return jdbcTypeName.toLowerCase(Locale.ROOT);
    }

    private DataSourceTableGrantVO toVO(DataSourceTableGrant grant) {
        DataSourceTableGrantVO vo = new DataSourceTableGrantVO();
        BeanUtils.copyProperties(grant, vo);
        return vo;
    }

    @FunctionalInterface
    private interface SqlConnectionCallback<T> {
        T doWithConnection(Connection connection) throws Exception;
    }
}
