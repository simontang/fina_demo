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
        MetricsIndexResponse index = getMetricsIndex(datasourceId);
        List<MetricsDetailResponse> details = index.getMetrics().stream()
                .map(item -> getMetricDetail(datasourceId, item.getMetricName()))
                .collect(Collectors.toList());
        return MetricsMetaFullResponse.builder()
                .index(index)
                .details(details)
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
    public SemanticQueryResponse query(SemanticQueryRequest request) {
        long wallStart = System.currentTimeMillis();
        String dsName  = resolveDatasourceName(request.getDatasourceId());
        boolean debug  = Boolean.TRUE.equals(request.getDebug());

        // ── Ad-hoc SQL escape hatch ───────────────────────────────────────────
        if (StringUtils.hasText(request.getCustomSql())) {
            log.info("Ad-hoc query datasource={}", request.getDatasourceId());
            SemanticQueryResponse.MetricResult result =
                    executeAdHocSql(request.getDatasourceId(), request.getCustomSql(),
                            request.getParams());
            List<String> sqls = debug ? List.of(request.getCustomSql()) : null;
            return SemanticQueryResponse.builder()
                    .datasourceId(request.getDatasourceId())
                    .datasourceName(dsName)
                    .results(List.of(result))
                    .totalExecutionTimeMs(System.currentTimeMillis() - wallStart)
                    .executedSqls(sqls)
                    .build();
        }

        // ── Semantic multi-metric mode ────────────────────────────────────────
        List<String> metrics = request.getMetrics() != null ? request.getMetrics() : List.of();
        if (metrics.isEmpty()) {
            throw new IllegalArgumentException(
                    "Either 'metrics' (semantic mode) or 'custom_sql' must be provided");
        }

        // Execute each metric in parallel; a single failure is captured per-result
        List<ExecutionBundle> bundles = metrics.parallelStream()
                .map(metricName -> executeMetric(metricName, request))
                .collect(Collectors.toList());

        List<SemanticQueryResponse.MetricResult> results = bundles.stream()
                .map(ExecutionBundle::result)
                .collect(Collectors.toList());

        List<String> executedSqls = debug
                ? bundles.stream().map(ExecutionBundle::sql).collect(Collectors.toList())
                : null;

        return SemanticQueryResponse.builder()
                .datasourceId(request.getDatasourceId())
                .datasourceName(dsName)
                .results(results)
                .totalExecutionTimeMs(System.currentTimeMillis() - wallStart)
                .executedSqls(executedSqls)
                .build();
    }

    // ── Private execution helpers ─────────────────────────────────────────────

    /**
     * Build SQL and execute a single metric. All exceptions are caught and
     * recorded in MetricResult.error so other metrics still succeed.
     */
    private ExecutionBundle executeMetric(String metricName, SemanticQueryRequest request) {
        long start = System.currentTimeMillis();
        String generatedSql = null;

        try {
            // Catalog detail is required for SQL generation
            JsonNode catalogDetail = catalog.findDetailItem(metricName)
                    .orElseThrow(() -> new IllegalArgumentException(
                            "Metric '" + metricName + "' not found in catalog"));

            // Build SQL
            SemanticQueryBuilder.BuildResult built =
                    queryBuilder.build(metricName, request, catalogDetail);
            generatedSql = built.sql();

            // Execute
            NamedParameterJdbcTemplate jdbc =
                    dsManager.getNamedJdbcTemplate(request.getDatasourceId());

            List<String> columns = new ArrayList<>();
            List<Map<String, Object>> rows = jdbc.query(
                    built.sql(),
                    new MapSqlParameterSource(built.params()),
                    rs -> {
                        List<Map<String, Object>> out = new ArrayList<>();
                        ResultSetMetaData rsMeta = rs.getMetaData();
                        int colCount = rsMeta.getColumnCount();
                        if (columns.isEmpty()) {
                            for (int i = 1; i <= colCount; i++) {
                                columns.add(rsMeta.getColumnLabel(i));
                            }
                        }
                        while (rs.next()) {
                            Map<String, Object> row = new LinkedHashMap<>();
                            for (int i = 1; i <= colCount; i++) {
                                row.put(rsMeta.getColumnLabel(i), rs.getObject(i));
                            }
                            out.add(row);
                        }
                        return out;
                    }
            );
            if (rows == null) rows = List.of();

            long elapsed = System.currentTimeMillis() - start;
            log.info("Executed metric={} datasource={} rows={} time={}ms",
                    metricName, request.getDatasourceId(), rows.size(), elapsed);

            SemanticQueryResponse.MetricResult metricResult =
                    SemanticQueryResponse.MetricResult.builder()
                            .metricName(metricName)
                            .displayName(catalogDetail.path("display_name").asText(null))
                            .dataType(catalogDetail.path("data_type").asText(null))
                            .format(catalogDetail.path("format").asText(null))
                            .polarity(catalogDetail.path("ai_agent_context")
                                    .path("polarity").asText(null))
                            .columns(columns)
                            .rows(rows)
                            .rowCount(rows.size())
                            .executionTimeMs(elapsed)
                            .error(null)
                            .aiHints(buildAiHints(catalogDetail))
                            .build();

            return new ExecutionBundle(metricResult, generatedSql);

        } catch (Exception e) {
            log.error("Failed to execute metric={}: {}", metricName, e.getMessage(), e);
            SemanticQueryResponse.MetricResult errorResult =
                    SemanticQueryResponse.MetricResult.builder()
                            .metricName(metricName)
                            .columns(List.of())
                            .rows(List.of())
                            .rowCount(0)
                            .executionTimeMs(System.currentTimeMillis() - start)
                            .error(e.getMessage())
                            .build();
            return new ExecutionBundle(errorResult, generatedSql);
        }
    }

    /** Execute an ad-hoc SQL string directly on the datasource */
    private SemanticQueryResponse.MetricResult executeAdHocSql(
            Long datasourceId, String sql, Map<String, Object> params) {
        long start = System.currentTimeMillis();
        try {
            NamedParameterJdbcTemplate jdbc = dsManager.getNamedJdbcTemplate(datasourceId);
            Map<String, Object> safeParams = params != null ? params : Map.of();
            log.debug("Executing ad-hoc SQL datasource={} sql={}", datasourceId, sql);

            List<String> columns = new ArrayList<>();
            List<Map<String, Object>> rows = jdbc.query(
                    sql,
                    new MapSqlParameterSource(safeParams),
                    rs -> {
                        List<Map<String, Object>> out = new ArrayList<>();
                        ResultSetMetaData rsMeta = rs.getMetaData();
                        int colCount = rsMeta.getColumnCount();
                        if (columns.isEmpty()) {
                            for (int i = 1; i <= colCount; i++) {
                                columns.add(rsMeta.getColumnLabel(i));
                            }
                        }
                        while (rs.next()) {
                            Map<String, Object> row = new LinkedHashMap<>();
                            for (int i = 1; i <= colCount; i++) {
                                row.put(rsMeta.getColumnLabel(i), rs.getObject(i));
                            }
                            out.add(row);
                        }
                        return out;
                    }
            );
            if (rows == null) rows = List.of();

            long elapsed = System.currentTimeMillis() - start;
            log.info("Ad-hoc SQL executed datasource={} rows={} time={}ms",
                    datasourceId, rows.size(), elapsed);

            return SemanticQueryResponse.MetricResult.builder()
                    .columns(columns)
                    .rows(rows)
                    .rowCount(rows.size())
                    .executionTimeMs(elapsed)
                    .build();
        } catch (Exception e) {
            log.error("Ad-hoc SQL failed datasource={}: {}", datasourceId, e.getMessage(), e);
            return SemanticQueryResponse.MetricResult.builder()
                    .columns(List.of())
                    .rows(List.of())
                    .rowCount(0)
                    .executionTimeMs(System.currentTimeMillis() - start)
                    .error(e.getMessage())
                    .build();
        }
    }

    private SemanticQueryResponse.AiHints buildAiHints(JsonNode catalogDetail) {
        JsonNode aiCtx  = catalogDetail.path("ai_agent_context");
        String polarity = aiCtx.path("polarity").asText("positive");
        String role     = catalogDetail.path("behavior_profile").path("role").asText("");

        String interpretation = "positive".equals(polarity)
                ? "Higher is better." + (role.isBlank() ? "" : " This metric is a " + role.replace("_", " ") + ".")
                : "Lower is better."  + (role.isBlank() ? "" : " This metric is a " + role.replace("_", " ") + ".");

        List<Map<String, Object>> thresholds = parseJsonArray(aiCtx.path("thresholds"));
        List<String> followup = buildFollowupHints(aiCtx.path("diagnostic_workflow"));

        return SemanticQueryResponse.AiHints.builder()
                .polarity(polarity)
                .valueInterpretation(interpretation)
                .thresholds(thresholds)
                .suggestedFollowup(followup)
                .build();
    }

    private List<String> buildFollowupHints(JsonNode workflowNode) {
        if (workflowNode == null || workflowNode.isMissingNode()) return List.of();
        List<String> hints = new ArrayList<>();
        JsonNode actions = workflowNode.path("actions");
        if (actions.isArray()) {
            actions.forEach(action -> {
                String type   = action.path("type").asText("");
                String intent = action.path("intent").asText("").replace("_", " ");
                switch (type) {
                    case "compare_metric" -> {
                        String target = action.path("metric").asText("");
                        hints.add("compare_metric: " + target + " — " + intent);
                    }
                    case "drill_down" -> {
                        List<String> dims = new ArrayList<>();
                        action.path("dimensions").forEach(d -> dims.add(d.asText()));
                        hints.add("drill_down by " + String.join(", ", dims) + " — " + intent);
                    }
                    default -> hints.add(type + " — " + intent);
                }
            });
        }
        return hints;
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

    /** Carrier record: bundles the MetricResult with the SQL used (for debug mode) */
    private record ExecutionBundle(SemanticQueryResponse.MetricResult result, String sql) {}
}
