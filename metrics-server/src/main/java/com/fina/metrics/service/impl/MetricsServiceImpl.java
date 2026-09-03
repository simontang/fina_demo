package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.config.DataSourceType;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.dto.*;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.entity.MetricsMeta;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.mapper.MetricsMetaMapper;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.MetaCatalogService;
import com.fina.metrics.service.MetricsService;
import com.fina.metrics.service.SemanticQueryBuilder;
import com.fina.metrics.service.TableViewMetaService;
import com.fina.metrics.util.ReadOnlySqlValidator;
import com.fina.metrics.util.SqlIdentifierUtils;
import com.fina.metrics.util.TenantHeaderResolver;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.sql.ResultSetMetaData;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class MetricsServiceImpl implements MetricsService {

    private final MetricsMetaMapper        metaMapper;
    private final DataSourceConfigMapper   dsConfigMapper;
    private final DynamicDataSourceManager dsManager;
    private final MetaCatalogService       catalog;
    private final SemanticQueryBuilder     queryBuilder;
    private final TableViewMetaService     tableViewMetaService;
    private final DataSourceTableAccessService tableAccessService;

    private static final int MAX_LIMIT     = 10000;
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String CATERPILLAR_PREFIX = "caterpillar_";

    // ── Discovery / meta ──────────────────────────────────────────────────────

    @Override
    public MetricsIndexResponse getMetricsIndex(Long datasourceId) {
        return getMetricsIndex(datasourceId, TenantHeaderResolver.DEFAULT_TENANT_ID);
    }

    @Override
    public MetricsIndexResponse getMetricsIndex(Long datasourceId, String tenantId) {
        log.debug("getMetricsIndex datasource={}", datasourceId);
        String resolvedTenant = TenantHeaderResolver.resolve(tenantId);
        DataSourceConfig datasource = resolveDatasourceConfig(datasourceId);
        DataSourceType sourceType = resolveDatasourceType(datasource);
        CdpCatalogScope cdpScope = resolveCdpCatalogScope(datasource, sourceType);
        List<DataSourceTableGrantVO> tableGrants = tableAccessService.listActiveGrants(resolvedTenant, datasourceId);
        boolean tableGrantFiltering = !tableGrants.isEmpty();
        Map<String, JsonNode> detailLookup = tableGrantFiltering ? detailLookup(datasourceId) : Map.of();
        boolean legacyRegistrationRequired = requiresLegacyCdpRegistration(sourceType, tableGrantFiltering, cdpScope);
        Set<String> registered = metaMapper.selectList(
                        new LambdaQueryWrapper<MetricsMeta>()
                                .eq(MetricsMeta::getDatasourceId, datasourceId)
                                .eq(MetricsMeta::getStatus, 1)
                                .eq(MetricsMeta::getDeleted, 0)
                                .select(MetricsMeta::getMetricCode))
                .stream()
                .map(MetricsMeta::getMetricCode)
                .collect(Collectors.toSet());

        String dsName = datasource != null ? datasource.getName() : null;

        List<MetricsIndexResponse.MetricIndexItem> items = catalog.getIndexItems(datasourceId).stream()
                .filter(node -> tableGrantFiltering
                        ? isMetricPublishedAndAuthorizedForRuntime(
                                datasourceId, node, sourceType, tableGrants, detailLookup)
                        : isMetricVisibleForDatasource(node, sourceType, cdpScope))
                .filter(node -> isMetricAuthorizedForTenant(
                        datasourceId, node, tableGrants, detailLookup))
                .filter(node -> !legacyRegistrationRequired
                        || isRegisteredMetricOnPublishedTable(node, registered, datasourceId, detailLookup))
                .map(node -> {
                    List<String> keywords = new ArrayList<>();
                    node.path("search_keywords").forEach(k -> keywords.add(k.asText()));
                    String metricName = node.path("metric_name").asText("");
                    boolean runtimeQueryable = registered.contains(metricName)
                            || (tableGrantFiltering && isMetricPublishedAndAuthorizedForRuntime(
                                    datasourceId, node, sourceType, tableGrants, detailLookup));
                    return MetricsIndexResponse.MetricIndexItem.builder()
                            .metricName(metricName)
                            .displayName(node.path("display_name").asText(""))
                            .domain(node.path("domain").asText(""))
                            .shortDesc(node.path("short_desc").asText(""))
                            .searchKeywords(keywords)
                            .registered(runtimeQueryable)
                            .build();
                })
                .collect(Collectors.toList());

        List<TableViewIndexItem> tables = tableViewMetaService.getTableViewsIndex(datasourceId).stream()
                .filter(item -> tableGrantFiltering
                        || isTableVisibleForDatasource(item.getTableName(), sourceType, cdpScope))
                .filter(item -> !tableGrantFiltering
                        || isTableAuthorizedByGrantList(tableGrants, null, item.getTableName()))
                .collect(Collectors.toList());

        return MetricsIndexResponse.builder()
                .datasourceId(datasourceId)
                .datasourceName(dsName)
                .catalogVersion(catalog.getCatalogVersion(datasourceId))
                .domainCategories(catalog.getDomainCategories(datasourceId))
                .metrics(items)
                .tables(tables)
                .build();
    }

    @Override
    public MetricsDetailResponse getMetricDetail(Long datasourceId, String metricName) {
        return getMetricDetail(datasourceId, metricName, TenantHeaderResolver.DEFAULT_TENANT_ID);
    }

    @Override
    public MetricsDetailResponse getMetricDetail(Long datasourceId, String metricName, String tenantId) {
        log.debug("getMetricDetail datasource={} metric={}", datasourceId, metricName);
        String resolvedTenant = TenantHeaderResolver.resolve(tenantId);
        DataSourceConfig datasource = resolveDatasourceConfig(datasourceId);
        DataSourceType sourceType = resolveDatasourceType(datasource);
        CdpCatalogScope cdpScope = resolveCdpCatalogScope(datasource, sourceType);
        JsonNode catalogDetail = catalog.findDetailItem(metricName, datasourceId)
                .orElseThrow(() -> new IllegalArgumentException(
                        "Metric not found in catalog: " + metricName));
        List<DataSourceTableGrantVO> tableGrants = tableAccessService.listActiveGrants(resolvedTenant, datasourceId);
        boolean tableGrantFiltering = !tableGrants.isEmpty();
        if (tableGrantFiltering) {
            if (!isMetricPublishedAndAuthorizedForRuntime(
                    datasourceId, catalogDetail, sourceType, tableGrants, Map.of())) {
                throw new ForbiddenException("Metric is not authorized for this tenant: " + metricName);
            }
        } else if (!isMetricVisibleForDatasource(catalogDetail, sourceType, cdpScope)) {
            throw new IllegalArgumentException(
                    "Metric " + metricName + " is not available for datasource type " + sourceType.getCode());
        }
        if (!isMetricAuthorizedForTenant(
                datasourceId,
                catalogDetail,
                tableGrants,
                Map.of())) {
            throw new ForbiddenException("Metric is not authorized for this tenant: " + metricName);
        }

        MetricsMeta dbMeta = metaMapper.selectOne(
                new LambdaQueryWrapper<MetricsMeta>()
                        .eq(MetricsMeta::getDatasourceId, datasourceId)
                        .eq(MetricsMeta::getMetricCode, metricName)
                        .eq(MetricsMeta::getStatus, 1)
                        .eq(MetricsMeta::getDeleted, 0)
        );
        if (requiresLegacyCdpRegistration(sourceType, tableGrantFiltering, cdpScope)
                && (dbMeta == null || !isTablePublishedForDatasource(
                        datasourceId, catalogDetail.path("source").path("table_view").asText(null)))) {
            throw new IllegalArgumentException(
                    "Metric " + metricName + " is not registered for this datasource");
        }

        JsonNode aiCtxNode = catalogDetail.path("ai_agent_context");
        List<Map<String, Object>> thresholds = parseJsonArray(aiCtxNode.path("thresholds"));
        List<String> synonyms = new ArrayList<>();
        aiCtxNode.path("synonyms").forEach(s -> synonyms.add(s.asText()));

        MetricsDetailResponse.AiAgentContext aiCtx = MetricsDetailResponse.AiAgentContext.builder()
                .polarity(aiCtxNode.path("polarity").asText("positive"))
                .synonyms(synonyms)
                .thresholds(thresholds)
                .diagnosticWorkflow(aiCtxNode.path("diagnostic_workflow").isMissingNode()
                        ? null : aiCtxNode.path("diagnostic_workflow"))
                .humanReadableExplanation(
                        aiCtxNode.path("human_readable_explanation").asText(null))
                .build();

        List<Map<String, Object>> dimensions =
                parseJsonArray(catalogDetail.path("supported_dimensions"));

        // Build TimeContext from the merged default_time_context catalog node
        MetricsDetailResponse.TimeContext timeCtx = null;
        JsonNode tcNode = catalogDetail.path("default_time_context");
        if (!tcNode.isMissingNode()) {
            List<String> grains = new java.util.ArrayList<>();
            tcNode.path("supported_grains").forEach(g -> grains.add(g.asText()));

            MetricsDetailResponse.TimeContext.QueryUsage.FilterUsage filterUsage = null;
            JsonNode filterNode = tcNode.path("query_usage").path("filter");
            if (!filterNode.isMissingNode()) {
                List<String> ops = new java.util.ArrayList<>();
                filterNode.path("supported_operators").forEach(o -> ops.add(o.asText()));
                Map<String, Object> example = MAPPER.convertValue(
                        filterNode.path("example"), new com.fasterxml.jackson.core.type.TypeReference<>() {});
                filterUsage = MetricsDetailResponse.TimeContext.QueryUsage.FilterUsage.builder()
                        .dimensionKey(filterNode.path("dimension_key").asText(null))
                        .supportedOperators(ops)
                        .valueFormat(filterNode.path("value_format").asText(null))
                        .example(example)
                        .build();
            }

            MetricsDetailResponse.TimeContext.QueryUsage.GroupByUsage gbUsage = null;
            JsonNode gbNode = tcNode.path("query_usage").path("group_by");
            if (!gbNode.isMissingNode()) {
                List<String> examples = new java.util.ArrayList<>();
                gbNode.path("examples").forEach(e -> examples.add(e.asText()));
                gbUsage = MetricsDetailResponse.TimeContext.QueryUsage.GroupByUsage.builder()
                        .pattern(gbNode.path("pattern").asText(null))
                        .examples(examples)
                        .build();
            }

            MetricsDetailResponse.TimeContext.QueryUsage queryUsage =
                    (filterUsage != null || gbUsage != null)
                            ? MetricsDetailResponse.TimeContext.QueryUsage.builder()
                                    .filter(filterUsage).groupBy(gbUsage).build()
                            : null;

            timeCtx = MetricsDetailResponse.TimeContext.builder()
                    .timeDimension(tcNode.path("time_dimension").asText(null))
                    .label(tcNode.path("label").asText(null))
                    .granularity(tcNode.path("granularity").asText(null))
                    .window(tcNode.path("window").asText(null))
                    .supportedGrains(grains)
                    .queryUsage(queryUsage)
                    .build();
        }

        MetricsDetailResponse.QueryInfo queryInfo = null;
        if (dbMeta != null) {
            queryInfo = MetricsDetailResponse.QueryInfo.builder()
                    .registered(true)
                    .metricCode(dbMeta.getMetricCode())
                    .querySql(dbMeta.getQuerySql())
                    .parameters(parseParametersJson(dbMeta.getParameters()))
                    .valueColumn(dbMeta.getValueColumn())
                    .build();
        }

        return MetricsDetailResponse.builder()
                .datasourceId(datasourceId)
                .metricName(metricName)
                .displayName(catalogDetail.path("display_name").asText(""))
                .domain(catalogDetail.path("domain").asText(""))
                .description(catalogDetail.path("description").asText(""))
                .dataType(catalogDetail.path("data_type").asText(""))
                .format(catalogDetail.path("format").asText(""))
                .defaultTimeContext(timeCtx)
                .supportedDimensions(dimensions)
                .aiAgentContext(aiCtx)
                .queryInfo(queryInfo)
                .build();
    }

    @Override
    public MetricsMetaFullResponse getMetricsMeta(Long datasourceId) {
        return getMetricsMeta(datasourceId, TenantHeaderResolver.DEFAULT_TENANT_ID);
    }

    @Override
    public MetricsMetaFullResponse getMetricsMeta(Long datasourceId, String tenantId) {
        // index already carries tables[] via getMetricsIndex
        String resolvedTenant = TenantHeaderResolver.resolve(tenantId);
        DataSourceConfig datasource = resolveDatasourceConfig(datasourceId);
        DataSourceType sourceType = resolveDatasourceType(datasource);
        CdpCatalogScope cdpScope = resolveCdpCatalogScope(datasource, sourceType);
        List<DataSourceTableGrantVO> tableGrants = tableAccessService.listActiveGrants(resolvedTenant, datasourceId);
        boolean tableGrantFiltering = !tableGrants.isEmpty();
        MetricsIndexResponse index = getMetricsIndex(datasourceId, resolvedTenant);
        List<MetricsDetailResponse> metricsDetails = index.getMetrics().stream()
                .map(item -> getMetricDetail(datasourceId, item.getMetricName(), resolvedTenant))
                .collect(Collectors.toList());
        List<TableViewDetailResponse> tableDetails = tableGrantFiltering
                && (index.getTables() == null || index.getTables().isEmpty())
                ? List.of()
                : tableViewMetaService.getTableViewsDetails(datasourceId).stream()
                        .filter(item -> tableGrantFiltering
                                || isTableVisibleForDatasource(item.getTableName(), sourceType, cdpScope))
                        .filter(item -> !tableGrantFiltering
                                || isTableAuthorizedByGrantList(tableGrants, null, item.getTableName()))
                        .collect(Collectors.toList());
        return MetricsMetaFullResponse.builder()
                .index(index)
                .metricsDetails(metricsDetails)
                .tablesDetails(tableDetails)
                .build();
    }

    // ── Metric definition CRUD ────────────────────────────────────────────────

    @Override
    public List<MetricsMetaVO> listByDatasource(Long datasourceId) {
        List<MetricsMeta> list = metaMapper.selectList(
                new LambdaQueryWrapper<MetricsMeta>()
                        .eq(MetricsMeta::getDatasourceId, datasourceId)
                        .eq(MetricsMeta::getDeleted, 0)
                        .orderByAsc(MetricsMeta::getMetricCode)
        );
        return list.stream().map(this::toVO).collect(Collectors.toList());
    }

    @Override
    public MetricsMetaVO getMetricMeta(Long datasourceId, String metricCode) {
        return toVO(requireMeta(datasourceId, metricCode));
    }

    @Override
    @Transactional
    public MetricsMetaVO createMetricMeta(MetricsMetaRequest request) {
        MetricsMeta meta = new MetricsMeta();
        BeanUtils.copyProperties(request, meta);
        meta.setDeleted(0);
        metaMapper.insert(meta);
        log.info("Created metric id={} code={}", meta.getId(), meta.getMetricCode());
        return toVO(meta);
    }

    @Override
    @Transactional
    public MetricsMetaVO updateMetricMeta(Long id, MetricsMetaRequest request) {
        MetricsMeta meta = metaMapper.selectOne(
                new LambdaQueryWrapper<MetricsMeta>()
                        .eq(MetricsMeta::getId, id)
                        .eq(MetricsMeta::getDeleted, 0)
        );
        if (meta == null) {
            throw new IllegalArgumentException("MetricsMeta not found: " + id);
        }
        BeanUtils.copyProperties(request, meta, "id", "createdAt", "deleted");
        metaMapper.updateById(meta);
        log.info("Updated metric id={} code={}", id, meta.getMetricCode());
        return toVO(meta);
    }

    @Override
    @Transactional
    public void deleteMetricMeta(Long id) {
        metaMapper.deleteById(id);
        log.info("Deleted metric id={}", id);
    }

    // ── Query execution ───────────────────────────────────────────────────────

    @Override
    public MetricsQueryData query(SemanticQueryRequest request) {
        return query(request, TenantHeaderResolver.DEFAULT_TENANT_ID);
    }

    @Override
    public MetricsQueryData query(SemanticQueryRequest request, String tenantId) {
        boolean debug = Boolean.TRUE.equals(request.getDebug());
        String resolvedTenant = TenantHeaderResolver.resolve(tenantId);

        // ── Ad-hoc SQL ───────────────────────────────────────────────────────
        if (StringUtils.hasText(request.getCustomSql())) {
            log.info("Ad-hoc query datasource={}", request.getDatasourceId());
            // For customSql, limit is enforced through JDBC maxRows so SQL remains dialect-neutral.
            String sqlToRun = request.getCustomSql();
            ReadOnlySqlValidator.validate(sqlToRun);
            SqlProbeRequest sqlRequest = new SqlProbeRequest();
            sqlRequest.setSql(sqlToRun);
            sqlRequest.setParams(request.getParams());
            sqlRequest.setMaxRows(request.getLimit());
            sqlRequest.setDebug(request.getDebug());
            MetricsQueryData result;
            if (tableAccessService.hasActiveGrants(resolvedTenant, request.getDatasourceId())) {
                result = tableAccessService.probeSql(resolvedTenant, request.getDatasourceId(), sqlRequest);
            } else {
                result = tableAccessService.queryDatasource(request.getDatasourceId(), sqlRequest);
            }
            result.setSemanticModel("adhoc");
            return result;
        }

        // ── Semantic mode: one SQL for all metrics ────────────────────────────
        List<String> metrics = request.getMetrics() != null ? request.getMetrics() : List.of();
        if (metrics.isEmpty()) {
            throw new IllegalArgumentException(
                    "Either 'metrics' (semantic mode) or 'custom_sql' must be provided");
        }
        DataSourceConfig datasource = resolveDatasourceConfig(request.getDatasourceId());
        DataSourceType sourceType = resolveDatasourceType(datasource);
        CdpCatalogScope cdpScope = resolveCdpCatalogScope(datasource, sourceType);
        List<DataSourceTableGrantVO> tableGrants = tableAccessService.listActiveGrants(
                resolvedTenant, request.getDatasourceId());
        boolean tableGrantFiltering = !tableGrants.isEmpty();
        boolean legacyCdpSemanticAllowed = isLegacyCdpSemanticAllowed(sourceType, tableGrantFiltering, cdpScope);
        if (sourceType.isCdp() && !tableGrantFiltering && !legacyCdpSemanticAllowed) {
            throw new IllegalArgumentException(
                    "Semantic metrics are not enabled for cdp_postgres yet; use custom_sql for CDP datasource queries");
        }

        List<JsonNode> catalogDetails = new ArrayList<>();
        String semanticTableView = null;
        for (String metricName : metrics) {
            JsonNode detail = catalog.findDetailItem(metricName, request.getDatasourceId())
                    .orElseThrow(() -> new IllegalArgumentException(
                            "Metric '" + metricName + "' not found in catalog"));
            if (tableGrantFiltering) {
                if (!isMetricPublishedAndAuthorizedForRuntime(
                        request.getDatasourceId(), detail, sourceType, tableGrants, Map.of())) {
                    throw new ForbiddenException("Metric is not authorized for this tenant: " + metricName);
                }
            } else if (!isMetricVisibleForDatasource(detail, sourceType, cdpScope)) {
                throw new IllegalArgumentException(
                        "Metric '" + metricName + "' is not available for datasource type " + sourceType.getCode());
            }
            if (requiresLegacyCdpRegistration(sourceType, tableGrantFiltering, cdpScope)) {
                requireMeta(request.getDatasourceId(), metricName);
                if (!isTablePublishedForDatasource(
                        request.getDatasourceId(), detail.path("source").path("table_view").asText(null))) {
                    throw new IllegalArgumentException(
                            "Metric '" + metricName + "' references an unpublished table");
                }
            }
            String metricTableView = detail.path("source").path("table_view").asText("");
            if (!StringUtils.hasText(metricTableView)) {
                throw new IllegalArgumentException(
                        "Metric '" + metricName + "' has no source.table_view in catalog");
            }
            if (semanticTableView == null) {
                semanticTableView = metricTableView;
            } else if (!SqlIdentifierUtils.sameTableName(semanticTableView, metricTableView, false)) {
                throw new IllegalArgumentException(
                        "All requested metrics must share the same source table/view");
            }
            catalogDetails.add(detail);
        }

        SemanticQueryBuilder.BuildResult built = queryBuilder.buildMulti(
                metrics, request, catalogDetails, sourceType.getCode());
        QueryResult qr = executeQuery(request.getDatasourceId(), built.sql(), built.params());

        String semanticModel = catalogDetails.get(0).path("source").path("table_view").asText("");

        Map<String, Object> debugObj = null;
        if (debug) {
            debugObj = new LinkedHashMap<>();
            debugObj.put("sql", built.sql());
            debugObj.put("params", built.params());
        }

        return MetricsQueryData.builder()
                .semanticModel(semanticModel)
                .columns(qr.columns())
                .rows(qr.rows())
                .debug(debugObj)
                .build();
    }

    /**
     * Execute SQL and return columns (name+type) and rows as value arrays.
     * Column order in rows matches columns list.
     */
    private QueryResult executeQuery(Long datasourceId, String sql, Map<String, Object> params) {
        NamedParameterJdbcTemplate jdbc = dsManager.getNamedJdbcTemplate(datasourceId);
        Map<String, Object> safeParams = params != null ? params : Map.of();

        List<ColumnMeta> columns = new ArrayList<>();
        List<List<Object>> rows = jdbc.query(sql, new MapSqlParameterSource(safeParams), rs -> {
            ResultSetMetaData meta = rs.getMetaData();
            int n = meta.getColumnCount();
            for (int i = 1; i <= n; i++) {
                columns.add(ColumnMeta.builder()
                        .name(meta.getColumnLabel(i))
                        .type(mapJdbcTypeToDocType(meta.getColumnTypeName(i)))
                        .build());
            }
            List<List<Object>> out = new ArrayList<>();
            while (rs.next()) {
                List<Object> row = new ArrayList<>(n);
                for (int i = 1; i <= n; i++) {
                    row.add(rs.getObject(i));
                }
                out.add(row);
            }
            return out;
        });
        if (rows == null) rows = List.of();

        log.info("Query executed datasource={} rows={}", datasourceId, rows.size());
        return new QueryResult(columns, rows);
    }

    /** Map JDBC type name to doc-friendly type (varchar, numeric, date). */
    private static String mapJdbcTypeToDocType(String jdbcTypeName) {
        if (jdbcTypeName == null) return "varchar";
        String u = jdbcTypeName.toUpperCase();
        if (u.contains("CHAR") || u.contains("TEXT") || u.contains("STRING")) return "varchar";
        if (u.contains("DECIMAL") || u.contains("NUMERIC") || u.contains("DOUBLE")
                || u.contains("FLOAT") || u.contains("REAL")) return "numeric";
        if (u.contains("INT") || u.contains("LONG") || u.contains("SMALLINT")) return "numeric";
        if (u.contains("DATE") || u.contains("TIME") || u.contains("STAMP")) return "date";
        return jdbcTypeName.toLowerCase();
    }

    private record QueryResult(List<ColumnMeta> columns, List<List<Object>> rows) {}

    private static int resolveLimit(Integer requested) {
        if (requested == null || requested <= 0) return 1000;
        return Math.min(requested, 10000);
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    private MetricsMeta requireMeta(Long datasourceId, String metricCode) {
        MetricsMeta meta = metaMapper.selectOne(
                new LambdaQueryWrapper<MetricsMeta>()
                        .eq(MetricsMeta::getDatasourceId, datasourceId)
                        .eq(MetricsMeta::getMetricCode, metricCode)
                        .eq(MetricsMeta::getStatus, 1)
                        .eq(MetricsMeta::getDeleted, 0)
        );
        if (meta == null) {
            throw new IllegalArgumentException(
                    "MetricsMeta not found: datasourceId=" + datasourceId
                            + " metricCode=" + metricCode);
        }
        return meta;
    }

    private MetricsMetaVO toVO(MetricsMeta meta) {
        MetricsMetaVO vo = new MetricsMetaVO();
        BeanUtils.copyProperties(meta, vo, "parameters");
        vo.setParametersJson(meta.getParameters());
        return vo;
    }

    private List<Map<String, Object>> parseJsonArray(JsonNode node) {
        if (node == null || node.isMissingNode() || !node.isArray()) return List.of();
        try {
            return MAPPER.convertValue(node, new TypeReference<>() {});
        } catch (Exception e) {
            return List.of();
        }
    }

    private List<Map<String, Object>> parseParametersJson(String json) {
        if (!StringUtils.hasText(json)) return List.of();
        try {
            return MAPPER.readValue(json, new TypeReference<>() {});
        } catch (Exception e) {
            log.warn("Failed to parse parameters JSON: {}", json);
            return List.of();
        }
    }

    private DataSourceConfig resolveDatasourceConfig(Long datasourceId) {
        try {
            return dsConfigMapper.selectById(datasourceId);
        } catch (Exception e) {
            return null;
        }
    }

    private DataSourceType resolveDatasourceType(DataSourceConfig config) {
        if (config == null) {
            return DataSourceType.SAP_B1_HANA;
        }
        return DataSourceType.resolve(config.getSourceType(), config.getUrl());
    }

    private boolean isMetricVisibleForDatasource(
            JsonNode node,
            DataSourceType sourceType,
            CdpCatalogScope cdpScope) {
        return isMetricVisibleForSource(node, sourceType, cdpScope);
    }

    private boolean isTableVisibleForDatasource(
            String tableName,
            DataSourceType sourceType,
            CdpCatalogScope cdpScope) {
        return isTableVisibleForSource(tableName, sourceType, cdpScope);
    }

    private boolean isMetricPublishedAndAuthorizedForRuntime(
            Long datasourceId,
            JsonNode node,
            DataSourceType sourceType,
            List<DataSourceTableGrantVO> tableGrants,
            Map<String, JsonNode> detailLookup) {
        if (!isMetricSourceTypeCompatible(datasourceId, node, sourceType, detailLookup)) {
            return false;
        }
        String tableView = resolveMetricTableView(datasourceId, node, detailLookup).orElse(null);
        return StringUtils.hasText(tableView)
                && isTableAuthorizedByGrantList(tableGrants, null, tableView)
                && isTablePublishedForDatasource(datasourceId, tableView);
    }

    private boolean isMetricAuthorizedForTenant(
            Long datasourceId,
            JsonNode node,
            List<DataSourceTableGrantVO> tableGrants,
            Map<String, JsonNode> detailLookup) {
        if (tableGrants.isEmpty()) {
            return true;
        }
        String tableView = resolveMetricTableView(datasourceId, node, detailLookup).orElse(null);
        return StringUtils.hasText(tableView)
                && isTableAuthorizedByGrantList(tableGrants, null, tableView);
    }

    private boolean isTableAuthorizedByGrantList(
            List<DataSourceTableGrantVO> tableGrants,
            String schemaName,
            String tableName) {
        if (!StringUtils.hasText(tableName)) {
            return false;
        }
        SqlIdentifierUtils.TableIdentifier tableIdentifier =
                SqlIdentifierUtils.parseTableIdentifier(schemaName, tableName);
        for (DataSourceTableGrantVO grant : tableGrants) {
            boolean caseSensitive = Boolean.TRUE.equals(grant.getCaseSensitive());
            String effectiveSchema = tableIdentifier.schemaName();
            if (StringUtils.hasText(effectiveSchema)
                    && StringUtils.hasText(grant.getSchemaName())) {
                boolean schemaMatches = caseSensitive
                        ? grant.getSchemaName().equals(effectiveSchema)
                        : grant.getSchemaName().equalsIgnoreCase(effectiveSchema);
                if (!schemaMatches) {
                    continue;
                }
            }
            String candidate = SqlIdentifierUtils.normalizeForComparison(
                    tableIdentifier.tableName(), caseSensitive);
            String pattern = caseSensitive
                    ? grant.getTablePattern()
                    : grant.getTablePattern().toLowerCase(Locale.ROOT);
            boolean allowed = "EXACT".equals(grant.getPatternType())
                    ? candidate.equals(pattern)
                    : candidate.startsWith(pattern);
            if (allowed) {
                return true;
            }
        }
        return false;
    }

    private Optional<String> resolveMetricTableView(
            Long datasourceId,
            JsonNode node,
            Map<String, JsonNode> detailLookup) {
        String tableView = node.path("source").path("table_view").asText(null);
        if (StringUtils.hasText(tableView)) {
            return Optional.of(tableView);
        }
        String metricName = node.path("metric_name").asText(null);
        if (!StringUtils.hasText(metricName)) {
            return Optional.empty();
        }
        return Optional.ofNullable(detailLookup.get(metricName))
                .or(() -> catalog.findDetailItem(metricName, datasourceId))
                .map(detail -> detail.path("source").path("table_view").asText(null))
                .filter(StringUtils::hasText);
    }

    private boolean isTablePublishedForDatasource(Long datasourceId, String tableName) {
        if (!StringUtils.hasText(tableName)) {
            return false;
        }
        return tableViewMetaService.getTableViewsIndex(datasourceId).stream()
                .anyMatch(table -> SqlIdentifierUtils.sameTableName(
                        table.getTableName(), tableName, false));
    }

    private boolean isRegisteredMetricOnPublishedTable(
            JsonNode indexNode,
            Set<String> registered,
            Long datasourceId,
            Map<String, JsonNode> detailLookup) {
        String metricName = indexNode.path("metric_name").asText("");
        if (!registered.contains(metricName)) {
            return false;
        }
        return resolveMetricTableView(datasourceId, indexNode, detailLookup)
                .filter(tableView -> isTablePublishedForDatasource(datasourceId, tableView))
                .isPresent();
    }

    private boolean isMetricSourceTypeCompatible(
            Long datasourceId,
            JsonNode node,
            DataSourceType sourceType,
            Map<String, JsonNode> detailLookup) {
        if (isMetricSourceTypeCompatible(node, sourceType)) {
            return true;
        }
        String metricName = node.path("metric_name").asText(null);
        if (!StringUtils.hasText(metricName)) {
            return false;
        }
        JsonNode detail = Optional.ofNullable(detailLookup.get(metricName))
                .or(() -> catalog.findDetailItem(metricName, datasourceId))
                .orElse(null);
        return detail != null && isMetricSourceTypeCompatible(detail, sourceType);
    }

    private boolean isMetricSourceTypeCompatible(JsonNode node, DataSourceType sourceType) {
        String itemSourceType = node.path("source_type").asText(null);
        if (sourceType.isCdp()) {
            return "cdp_postgres".equals(itemSourceType);
        }
        return itemSourceType == null || itemSourceType.isBlank() || "sap_b1_hana".equals(itemSourceType);
    }

    private boolean isLegacyCdpSemanticAllowed(
            DataSourceType sourceType,
            boolean tableGrantFiltering,
            CdpCatalogScope cdpScope) {
        return sourceType.isCdp()
                && !tableGrantFiltering
                && cdpScope == CdpCatalogScope.CATERPILLAR;
    }

    private boolean requiresLegacyCdpRegistration(
            DataSourceType sourceType,
            boolean tableGrantFiltering,
            CdpCatalogScope cdpScope) {
        return isLegacyCdpSemanticAllowed(sourceType, tableGrantFiltering, cdpScope);
    }

    private Map<String, JsonNode> detailLookup(Long datasourceId) {
        return catalog.getDetailItems(datasourceId).stream()
                .filter(item -> StringUtils.hasText(item.path("metric_name").asText(null)))
                .collect(Collectors.toMap(
                        item -> item.path("metric_name").asText(),
                        item -> item,
                        (left, right) -> right,
                        LinkedHashMap::new));
    }

    private boolean hasCaterpillarPrefix(String name) {
        return StringUtils.hasText(name) && name.startsWith(CATERPILLAR_PREFIX);
    }

    private boolean isMetricVisibleForSource(JsonNode node, DataSourceType sourceType, CdpCatalogScope cdpScope) {
        String itemSourceType = node.path("source_type").asText(null);
        if (sourceType.isCdp()) {
            return "cdp_postgres".equals(itemSourceType)
                    && cdpScope.matches(metricCatalogName(node));
        }
        return itemSourceType == null || itemSourceType.isBlank() || "sap_b1_hana".equals(itemSourceType);
    }

    private boolean isTableVisibleForSource(String tableName, DataSourceType sourceType, CdpCatalogScope cdpScope) {
        boolean isCdpTable = tableName != null
                && (tableName.startsWith("demo_")
                || tableName.startsWith("retailcdp_")
                || hasCaterpillarPrefix(tableName));
        return sourceType.isCdp() ? isCdpTable && cdpScope.matches(tableName) : !isCdpTable;
    }

    private String metricCatalogName(JsonNode node) {
        String tableView = node.path("source").path("table_view").asText("");
        if (StringUtils.hasText(tableView)) {
            return tableView;
        }
        return node.path("metric_name").asText("");
    }

    private CdpCatalogScope resolveCdpCatalogScope(DataSourceConfig datasource, DataSourceType sourceType) {
        if (!sourceType.isCdp()) {
            return CdpCatalogScope.ALL;
        }
        if (datasource == null) {
            return CdpCatalogScope.ALL;
        }
        String descriptor = String.join(" ",
                String.valueOf(datasource.getId()),
                nullToEmpty(datasource.getName()),
                nullToEmpty(datasource.getDescription()),
                nullToEmpty(datasource.getUrl()),
                nullToEmpty(datasource.getSchemaName()))
                .toLowerCase(Locale.ROOT);
        if (descriptor.contains("retailcdp") || descriptor.contains("retail cdp")) {
            return CdpCatalogScope.RETAIL_CDP;
        }
        if (descriptor.contains("caterpillar")) {
            return CdpCatalogScope.CATERPILLAR;
        }
        if (descriptor.contains("demo_") || descriptor.contains("cdp demo") || descriptor.contains("crm agent cdp")) {
            return CdpCatalogScope.DEMO_CDP;
        }
        return CdpCatalogScope.NONE;
    }

    private String nullToEmpty(String value) {
        return value == null ? "" : value;
    }

    private enum CdpCatalogScope {
        DEMO_CDP("demo_"),
        RETAIL_CDP("retailcdp_"),
        CATERPILLAR(CATERPILLAR_PREFIX),
        ALL(""),
        NONE(null);

        private final String prefix;

        CdpCatalogScope(String prefix) {
            this.prefix = prefix;
        }

        private boolean matches(String name) {
            if (!StringUtils.hasText(name)) {
                return false;
            }
            if (this == ALL) {
                return true;
            }
            if (this == NONE) {
                return false;
            }
            return name.startsWith(prefix);
        }
    }

}
