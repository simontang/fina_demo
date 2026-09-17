package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.config.DataSourceType;
import com.fina.metrics.dto.*;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.entity.DataSourceTableGrant;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.mapper.DataSourceTableGrantMapper;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.DataSourceVisibleScope;
import com.fina.metrics.service.RuntimeMetaCache;
import com.fina.metrics.util.ReadOnlySqlValidator;
import com.fina.metrics.util.JdbcValueNormalizer;
import com.fina.metrics.util.SqlTableReferenceExtractor;
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
    private final RuntimeMetaCache runtimeMetaCache;

    @Override
    public List<DataSourceTableGrantVO> listGrants(String tenantId, Long datasourceId) {
        return selectEffectiveGrants(tenantId, datasourceId, null).stream()
                .map(this::toVO)
                .collect(Collectors.toList());
    }

    @Override
    public List<DataSourceTableGrantVO> listActiveGrants(String tenantId, Long datasourceId) {
        return selectEffectiveActiveGrants(tenantId, datasourceId).stream()
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
        grant.setTenantId(DataSourceVisibleScope.LEGACY_TENANT_MARKER);
        grant.setDatasourceId(datasourceId);
        applyRequest(grant, request);
        grant.setDeleted(0);
        grantMapper.insert(grant);
        runtimeMetaCache.invalidateDatasourceAfterCommit(datasourceId, "visible scope created");
        log.info("Created datasource visible scope id={} datasource={} pattern={}",
                grant.getId(), datasourceId, grant.getTablePattern());
        return toVO(grant);
    }

    @Override
    @Transactional
    public DataSourceTableGrantVO updateGrant(
            String tenantId,
            Long datasourceId,
            Long grantId,
            DataSourceTableGrantRequest request) {
        DataSourceTableGrant grant = requireGrant(datasourceId, grantId);
        applyRequest(grant, request);
        grantMapper.updateById(grant);
        runtimeMetaCache.invalidateDatasourceAfterCommit(datasourceId, "visible scope updated");
        log.info("Updated datasource visible scope id={} datasource={}", grantId, datasourceId);
        return toVO(grant);
    }

    @Override
    @Transactional
    public void deleteGrant(String tenantId, Long datasourceId, Long grantId) {
        DataSourceTableGrant grant = requireGrant(datasourceId, grantId);
        grantMapper.deleteById(grant.getId());
        runtimeMetaCache.invalidateDatasourceAfterCommit(datasourceId, "visible scope deleted");
        log.info("Deleted datasource visible scope id={} datasource={}", grantId, datasourceId);
    }

    @Override
    public boolean hasActiveGrants(String tenantId, Long datasourceId) {
        return !selectEffectiveActiveGrants(tenantId, datasourceId).isEmpty();
    }

    @Override
    public boolean isTableAuthorized(
            String tenantId,
            Long datasourceId,
            String schemaName,
            String tableName) {
        return visibleScope(datasourceId).allows(schemaName, tableName);
    }

    @Override
    public boolean isTableAuthorizedIfGrantsConfigured(
            String tenantId,
            Long datasourceId,
            String schemaName,
            String tableName) {
        return isTableAuthorized(null, datasourceId, schemaName, tableName);
    }

    @Override
    public void assertSqlAuthorized(String tenantId, Long datasourceId, String sql) {
        ReadOnlySqlValidator.validate(sql);
        DataSourceConfig datasource = requireDatasource(datasourceId);
        DataSourceVisibleScope scope = visibleScope(datasource);
        Set<SqlTableReferenceExtractor.TableReference> references = SqlTableReferenceExtractor.extract(sql);
        if (!scope.isRestricted()) return;
        if (scope.rules().isEmpty()) {
            throw new ForbiddenException("No active datasource visible scopes for datasourceId=" + datasourceId);
        }
        for (SqlTableReferenceExtractor.TableReference reference : references) {
            if (!StringUtils.hasText(reference.schemaName())
                    && (!StringUtils.hasText(datasource.getSchemaName())
                        || DataSourceType.resolve(datasource.getSourceType(), datasource.getUrl()) == DataSourceType.SAP_B1_SQLSERVER)) {
                throw new ForbiddenException("Schema-qualified table required in restricted SQL: " + reference.original());
            }
            if (!scope.allows(reference.schemaName(), reference.tableName())) {
                throw new ForbiddenException("SQL references unauthorized table: " + reference.original());
            }
        }
    }

    @Override
    public List<DataSourceTableVO> listPhysicalTables(Long datasourceId, String schemaName) {
        DataSourceConfig datasource = requireDatasource(datasourceId);
        DataSourceVisibleScope scope = visibleScope(datasource);
        if (scope.isRestricted() && scope.rules().isEmpty()) return List.of();
        String effectiveSchema = trimToNull(schemaName);
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
                        if (!scope.allows(rowSchema, tableName)) continue;
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
        return listPhysicalTables(datasourceId, null);
    }

    @Override
    public List<DataSourceColumnVO> listAuthorizedColumns(
            String tenantId,
            Long datasourceId,
            String schemaName,
            String tableName) {
        if (!isTableAuthorized(tenantId, datasourceId, schemaName, tableName)) {
            throw new ForbiddenException("Table is not authorized by datasource visible scope: " + tableName);
        }
        DataSourceVisibleScope scope = visibleScope(datasourceId);
        List<DataSourceTableGrant> grants = selectEffectiveActiveGrants(tenantId, datasourceId);
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
                            if (!scope.allows(rowSchema, rowTable)) {
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
        assertSqlAuthorized(null, datasourceId, request.getSql());
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
            Map<String, Object> params = request.getParams() != null ? request.getParams() : Map.of();
            List<List<Object>> rows = params.isEmpty()
                    ? jdbc.getJdbcTemplate().query(request.getSql(), extractor)
                    : jdbc.query(request.getSql(), new MapSqlParameterSource(params), extractor);
            Map<String, Object> debug = null;
            if (Boolean.TRUE.equals(request.getDebug())) {
                debug = new LinkedHashMap<>();
                debug.put("sql", request.getSql());
                debug.put("params", params);
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
                row.add(JdbcValueNormalizer.normalize(rs.getObject(i)));
            }
            rows.add(row);
        }
        return rows;
    }

    private List<DataSourceTableGrant> selectEffectiveActiveGrants(String tenantId, Long datasourceId) {
        return selectEffectiveGrants(tenantId, datasourceId, 1);
    }

    private List<DataSourceTableGrant> selectEffectiveGrants(String tenantId, Long datasourceId, Integer status) {
        return selectDatasourceGrants(datasourceId, status);
    }

    private DataSourceVisibleScope visibleScope(Long datasourceId) {
        return visibleScope(requireDatasource(datasourceId));
    }

    private DataSourceVisibleScope visibleScope(DataSourceConfig datasource) {
        return DataSourceVisibleScope.from(datasource, listActiveGrants(null, datasource.getId()));
    }

    private List<DataSourceTableGrant> selectDatasourceGrants(Long datasourceId, Integer status) {
        LambdaQueryWrapper<DataSourceTableGrant> wrapper = new LambdaQueryWrapper<DataSourceTableGrant>()
                .eq(DataSourceTableGrant::getDatasourceId, datasourceId)
                .eq(DataSourceTableGrant::getDeleted, 0)
                .orderByAsc(DataSourceTableGrant::getId);
        if (status != null) {
            wrapper.eq(DataSourceTableGrant::getStatus, status);
        }
        return grantMapper.selectList(wrapper);
    }

    private DataSourceTableGrant requireGrant(Long datasourceId, Long grantId) {
        DataSourceTableGrant grant = grantMapper.selectOne(
                new LambdaQueryWrapper<DataSourceTableGrant>()
                        .eq(DataSourceTableGrant::getId, grantId)
                        .eq(DataSourceTableGrant::getDatasourceId, datasourceId)
                        .eq(DataSourceTableGrant::getDeleted, 0)
        );
        if (grant == null) {
            throw new IllegalArgumentException("Datasource visible scope not found: id=" + grantId);
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
