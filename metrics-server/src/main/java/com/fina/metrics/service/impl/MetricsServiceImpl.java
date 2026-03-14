package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.dto.*;
import com.fina.metrics.entity.MetricsMeta;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.mapper.MetricsMetaMapper;
import com.fina.metrics.service.MetaCatalogService;
import com.fina.metrics.service.MetricsService;
import com.fina.metrics.service.SemanticQueryBuilder;
import com.fina.metrics.service.TableViewMetaService;
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

    private static final int MAX_LIMIT     = 10000;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    // ── Discovery / meta ──────────────────────────────────────────────────────

    @Override
    public MetricsIndexResponse getMetricsIndex(Long datasourceId) {
        log.debug("getMetricsIndex datasource={}", datasourceId);
        Set<String> registered = metaMapper.selectList(
                        new LambdaQueryWrapper<MetricsMeta>()
                                .eq(MetricsMeta::getDatasourceId, datasourceId)
                                .eq(MetricsMeta::getStatus, 1)
                                .eq(MetricsMeta::getDeleted, 0)
                                .select(MetricsMeta::getMetricCode))
                .stream()
                .map(MetricsMeta::getMetricCode)
                .collect(Collectors.toSet());

        String dsName = resolveDatasourceName(datasourceId);

        List<MetricsIndexResponse.MetricIndexItem> items = catalog.getIndexItems().stream()
                .map(node -> {
                    List<String> keywords = new ArrayList<>();
                    node.path("search_keywords").forEach(k -> keywords.add(k.asText()));
                    String metricName = node.path("metric_name").asText("");
                    return MetricsIndexResponse.MetricIndexItem.builder()
                            .metricName(metricName)
                            .displayName(node.path("display_name").asText(""))
                            .domain(node.path("domain").asText(""))
                            .shortDesc(node.path("short_desc").asText(""))
                            .searchKeywords(keywords)
                            .registered(registered.contains(metricName))
                            .build();
                })
                .collect(Collectors.toList());

        return MetricsIndexResponse.builder()
                .datasourceId(datasourceId)
                .datasourceName(dsName)
                .catalogVersion(catalog.getCatalogVersion())
                .domainCategories(catalog.getDomainCategories())
                .metrics(items)
                .tables(tableViewMetaService.getTableViewsIndex())
                .build();
    }

    @Override
    public MetricsDetailResponse getMetricDetail(Long datasourceId, String metricName) {
        log.debug("getMetricDetail datasource={} metric={}", datasourceId, metricName);
        JsonNode catalogDetail = catalog.findDetailItem(metricName)
                .orElseThrow(() -> new IllegalArgumentException(
                        "Metric not found in catalog: " + metricName));

        MetricsMeta dbMeta = metaMapper.selectOne(
                new LambdaQueryWrapper<MetricsMeta>()
                        .eq(MetricsMeta::getDatasourceId, datasourceId)
                        .eq(MetricsMeta::getMetricCode, metricName)
                        .eq(MetricsMeta::getStatus, 1)
                        .eq(MetricsMeta::getDeleted, 0)
        );

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
        // index already carries tables[] via getMetricsIndex
        MetricsIndexResponse index = getMetricsIndex(datasourceId);
        List<MetricsDetailResponse> metricsDetails = index.getMetrics().stream()
                .map(item -> getMetricDetail(datasourceId, item.getMetricName()))
                .collect(Collectors.toList());
        return MetricsMetaFullResponse.builder()
                .index(index)
                .metricsDetails(metricsDetails)
                .tablesDetails(tableViewMetaService.getTableViewsDetails())
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
        boolean debug = Boolean.TRUE.equals(request.getDebug());

        // ── Ad-hoc SQL ───────────────────────────────────────────────────────
        if (StringUtils.hasText(request.getCustomSql())) {
            log.info("Ad-hoc query datasource={}", request.getDatasourceId());
            int resolvedLimit = resolveLimit(request.getLimit());
            String sqlToRun = request.getCustomSql().trim();
            if (resolvedLimit > 0 && !sqlToRun.toUpperCase().contains(" LIMIT ")) {
                sqlToRun = sqlToRun + "\nLIMIT " + resolvedLimit;
            }
            Map<String, Object> params = request.getParams() != null ? request.getParams() : Map.of();
            QueryResult qr = executeQuery(request.getDatasourceId(), sqlToRun, params);
            Map<String, Object> debugObj = debug
                    ? Map.of("sql", sqlToRun, "params", params)
                    : null;
            return MetricsQueryData.builder()
                    .semanticModel("adhoc")
                    .columns(qr.columns())
                    .rows(qr.rows())
                    .debug(debugObj)
                    .build();
        }

        // ── Semantic mode: one SQL for all metrics ────────────────────────────
        List<String> metrics = request.getMetrics() != null ? request.getMetrics() : List.of();
        if (metrics.isEmpty()) {
            throw new IllegalArgumentException(
                    "Either 'metrics' (semantic mode) or 'custom_sql' must be provided");
        }

        List<JsonNode> catalogDetails = new ArrayList<>();
        for (String metricName : metrics) {
            JsonNode detail = catalog.findDetailItem(metricName)
                    .orElseThrow(() -> new IllegalArgumentException(
                            "Metric '" + metricName + "' not found in catalog"));
            catalogDetails.add(detail);
        }

        SemanticQueryBuilder.BuildResult built = queryBuilder.buildMulti(metrics, request, catalogDetails);
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

    private String resolveDatasourceName(Long datasourceId) {
        try {
            var config = dsConfigMapper.selectById(datasourceId);
            return config != null ? config.getName() : null;
        } catch (Exception e) {
            return null;
        }
    }

}
