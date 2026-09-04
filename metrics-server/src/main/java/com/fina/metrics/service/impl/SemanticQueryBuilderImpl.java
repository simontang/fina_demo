package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fina.metrics.config.DataSourceType;
import com.fina.metrics.dto.SemanticQueryRequest;
import com.fina.metrics.service.SemanticQueryBuilder;
import com.fina.metrics.util.SqlIdentifierUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Generates datasource-specific SQL from a semantic query.
 *
 * Generation steps:
 *   1. Resolve dim_ids in group_by → actual HANA field names (via supported_dimensions)
 *   2. Build SELECT clause: group_by columns + metric expression AS "value"
 *   3. Build FROM clause from catalog source.table_view
 *   4. Build WHERE clause from filters (+ catalog base_filters)
 *   5. Build GROUP BY from resolved group_by fields
 *   6. Build ORDER BY from request orderBy
 *   7. Append LIMIT
 *
 * Metric calculation supports two contracts:
 *   1. Legacy catalog: calculation.sql_expression is already an executable SQL expression.
 *   2. DB-backed meta: SQL-free calculation DSL, e.g. aggregate measure or ratio
 *      derived from other metrics. This keeps new datasource meta semantic instead
 *      of forcing agents to author raw SQL snippets.
 *
 * Double-underscore time granularity:
 *   "DocDate__month" → TO_NVARCHAR("DocDate", 'YYYY-MM') AS "DocDate__month"
 *   Supported: year, month, week, day
 *
 * Filter operators:
 *   BETWEEN, IN, EQ, NEQ, GT, GTE, LT, LTE, LIKE, NOT_NULL
 *
 * Dimension resolution is intentionally strict: request dimensions must be
 * published in supported_dimensions or default_time_context.
 */
@Slf4j
@Service
public class SemanticQueryBuilderImpl implements SemanticQueryBuilder {

    private static final Pattern FORMULA_TOKEN = Pattern.compile(
            "[A-Za-z_][A-Za-z0-9_]*|\\d+(?:\\.\\d+)?|[()+\\-*/]");

    private static final Set<String> AGGREGATIONS = Set.of(
            "sum", "avg", "min", "max", "count", "count_distinct");

    private static final Map<String, String> HANA_GRAIN_FORMAT = Map.of(
            "year",  "'YYYY'",
            "month", "'YYYY-MM'",
            "week",  "'YYYY-IW'",
            "day",   "'YYYY-MM-DD'"
    );

    private static final Map<String, String> POSTGRES_GRAIN_FORMAT = Map.of(
            "year",  "'YYYY'",
            "month", "'YYYY-MM'",
            "week",  "'IYYY-IW'",
            "day",   "'YYYY-MM-DD'"
    );

    @Override
    public BuildResult build(String metricName,
                             SemanticQueryRequest request,
                             JsonNode catalogDetail,
                             String sourceType) {
        DataSourceType dataSourceType = DataSourceType.resolve(sourceType, null);

        String tableView     = catalogDetail.path("source").path("table_view").asText("");
        JsonNode dimNodes    = catalogDetail.path("supported_dimensions");
        JsonNode timeDimNode = catalogDetail.path("default_time_context");
        JsonNode baseFilters = catalogDetail.path("source").path("base_filters");

        if (!StringUtils.hasText(tableView)) {
            log.error("Catalog misconfiguration: metric={} missing source.table_view", metricName);
            throw new IllegalStateException(
                    "Metric " + metricName + " has no source.table_view in catalog");
        }
        Map<String, JsonNode> metricDetailsByName = metricDetailLookup(
                List.of(catalogDetail), List.of(metricName));
        String sqlExpr = resolveMetricExpression(
                metricName,
                catalogDetail,
                metricDetailsByName,
                new LinkedHashSet<>());

        // Build dim_id → field_name lookup map for this metric
        Map<String, String> dimMap = buildDimMap(dimNodes, timeDimNode);

        List<String> groupByItems = request.getGroupBy() != null
                ? request.getGroupBy() : List.of();
        validateRequiredGroupBy(List.of(catalogDetail), groupByItems);

        Map<String, Object> params = new LinkedHashMap<>();
        List<String> selectCols   = new ArrayList<>();
        List<String> selectExprs  = new ArrayList<>();
        List<String> groupByClauses = new ArrayList<>();
        List<String> columnLabels = new ArrayList<>();

        // ── SELECT + GROUP BY for each group_by item ─────────────────────────
        for (String gbItem : groupByItems) {
            DimRef ref = resolveGroupByItem(gbItem, dimMap, dataSourceType);
            selectExprs.add(ref.selectExpr + " AS " + quoteAlias(gbItem));
            groupByClauses.add(ref.groupByExpr);
            columnLabels.add(gbItem);
            selectCols.add(gbItem);
        }

        // ── SELECT: metric expression AS metric name (doc: column = metric name) ──
        selectExprs.add(sqlExpr + " AS " + quoteAlias(metricName));
        columnLabels.add(metricName);

        // ── WHERE clause ──────────────────────────────────────────────────────
        List<String> whereParts = new ArrayList<>();

        // Catalog base_filters (e.g. DiscPrcnt IS NOT NULL)
        if (baseFilters != null && baseFilters.isArray()) {
            baseFilters.forEach(bf -> {
                String field    = bf.path("field").asText("");
                String operator = bf.path("operator").asText("").toUpperCase();
                if (StringUtils.hasText(field) && "NOT_NULL".equals(operator)) {
                    whereParts.add(quoteColumn(field) + " IS NOT NULL");
                }
            });
        }

        // Request filters
        List<SemanticQueryRequest.FilterItem> filters = request.getFilters() != null
                ? request.getFilters() : List.of();

        for (int i = 0; i < filters.size(); i++) {
            SemanticQueryRequest.FilterItem f = filters.get(i);
            String fieldName = resolveFieldName(f.getDimension(), dimMap);
            String quotedField = quoteColumn(fieldName);
            String op = f.getOperator().toUpperCase();
            List<Object> values = f.getValues() != null ? f.getValues() : List.of();

            switch (op) {
                case "BETWEEN" -> {
                    String p0 = "f" + i + "_v0", p1 = "f" + i + "_v1";
                    whereParts.add(quotedField + " BETWEEN :" + p0 + " AND :" + p1);
                    params.put(p0, values.get(0));
                    params.put(p1, values.get(1));
                }
                case "IN" -> {
                    String pKey = "f" + i + "_vals";
                    whereParts.add(quotedField + " IN (:" + pKey + ")");
                    params.put(pKey, values);
                }
                case "EQ" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " = :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "NEQ" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " != :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "GT" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " > :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "GTE" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " >= :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "LT" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " < :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "LTE" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " <= :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "LIKE" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " LIKE :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "NOT_NULL" ->
                    whereParts.add(quotedField + " IS NOT NULL");
                default ->
                    log.warn("Unknown filter operator '{}' — skipped", op);
            }
        }

        // ── ORDER BY ──────────────────────────────────────────────────────────
        List<String> orderByParts = new ArrayList<>();
        List<SemanticQueryRequest.OrderByItem> orderBys = request.getOrderBy() != null
                ? request.getOrderBy() : List.of();

        for (SemanticQueryRequest.OrderByItem ob : orderBys) {
            String dir = "DESC".equalsIgnoreCase(ob.getDirection()) ? "DESC" : "ASC";
            validateOrderByField(ob.getField(), groupByItems, List.of(metricName));
            orderByParts.add(quoteAlias(ob.getField()) + " " + dir);
        }

        // ── Assemble SQL ──────────────────────────────────────────────────────
        int limit = resolveLimit(request.getLimit());
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT\n  ");
        sql.append(String.join(",\n  ", selectExprs));
        sql.append("\nFROM ").append(SqlIdentifierUtils.quoteQualified(tableView));

        if (!whereParts.isEmpty()) {
            sql.append("\nWHERE ").append(String.join("\n  AND ", whereParts));
        }
        if (!groupByClauses.isEmpty()) {
            sql.append("\nGROUP BY ").append(String.join(", ", groupByClauses));
        }
        if (!orderByParts.isEmpty()) {
            sql.append("\nORDER BY ").append(String.join(", ", orderByParts));
        }
        sql.append("\nLIMIT ").append(limit);

        log.debug("Built SQL for metric={}: {}", metricName, sql);
        return new BuildResult(sql.toString(), params, columnLabels);
    }

    @Override
    public BuildResult buildMulti(List<String> metricNames,
                                  SemanticQueryRequest request,
                                  List<JsonNode> catalogDetails,
                                  String sourceType) {
        DataSourceType dataSourceType = DataSourceType.resolve(sourceType, null);
        if (metricNames == null || metricNames.isEmpty() || catalogDetails == null
                || catalogDetails.isEmpty()) {
            throw new IllegalArgumentException("metricNames and catalogDetails must be non-empty");
        }
        Map<String, JsonNode> metricDetailsByName = metricDetailLookup(catalogDetails, metricNames);
        JsonNode first = requireMetricDetail(metricNames.get(0), metricDetailsByName);
        String tableView = first.path("source").path("table_view").asText("");
        if (!StringUtils.hasText(tableView)) {
            throw new IllegalStateException("First metric has no source.table_view in catalog");
        }
        for (Map.Entry<String, JsonNode> entry : metricDetailsByName.entrySet()) {
            String tv = entry.getValue().path("source").path("table_view").asText("");
            if (!tableView.equals(tv)) {
                throw new IllegalArgumentException(
                        "All requested metrics must share the same source table/view. "
                                + "First uses " + tableView + ", metric " + entry.getKey() + " uses " + tv);
            }
        }

        JsonNode dimNodes = first.path("supported_dimensions");
        JsonNode timeDimNode = first.path("default_time_context");
        JsonNode baseFilters = first.path("source").path("base_filters");

        Map<String, String> dimMap = buildDimMap(dimNodes, timeDimNode);
        List<String> groupByItems = request.getGroupBy() != null ? request.getGroupBy() : List.of();
        validateRequiredGroupBy(
                metricNames.stream().map(metricDetailsByName::get).filter(Objects::nonNull).toList(),
                groupByItems);

        Map<String, Object> params = new LinkedHashMap<>();
        List<String> selectExprs = new ArrayList<>();
        List<String> groupByClauses = new ArrayList<>();
        List<String> columnLabels = new ArrayList<>();

        for (String gbItem : groupByItems) {
            DimRef ref = resolveGroupByItem(gbItem, dimMap, dataSourceType);
            selectExprs.add(ref.selectExpr + " AS " + quoteAlias(gbItem));
            groupByClauses.add(ref.groupByExpr);
            columnLabels.add(gbItem);
        }

        for (int i = 0; i < metricNames.size(); i++) {
            String metricName = metricNames.get(i);
            JsonNode detail = requireMetricDetail(metricName, metricDetailsByName);
            String sqlExpr = resolveMetricExpression(
                    metricName,
                    detail,
                    metricDetailsByName,
                    new LinkedHashSet<>());
            selectExprs.add(sqlExpr + " AS " + quoteAlias(metricName));
            columnLabels.add(metricName);
        }

        List<String> whereParts = new ArrayList<>();
        if (baseFilters != null && baseFilters.isArray()) {
            baseFilters.forEach(bf -> {
                String field = bf.path("field").asText("");
                String operator = bf.path("operator").asText("").toUpperCase();
                if (StringUtils.hasText(field) && "NOT_NULL".equals(operator)) {
                    whereParts.add(quoteColumn(field) + " IS NOT NULL");
                }
            });
        }

        List<SemanticQueryRequest.FilterItem> filters = request.getFilters() != null
                ? request.getFilters() : List.of();
        for (int i = 0; i < filters.size(); i++) {
            SemanticQueryRequest.FilterItem f = filters.get(i);
            String fieldName = resolveFieldName(f.getDimension(), dimMap);
            String quotedField = quoteColumn(fieldName);
            String op = f.getOperator().toUpperCase();
            List<Object> values = f.getValues() != null ? f.getValues() : List.of();

            switch (op) {
                case "BETWEEN" -> {
                    String p0 = "f" + i + "_v0", p1 = "f" + i + "_v1";
                    whereParts.add(quotedField + " BETWEEN :" + p0 + " AND :" + p1);
                    params.put(p0, values.get(0));
                    params.put(p1, values.get(1));
                }
                case "IN" -> {
                    String pKey = "f" + i + "_vals";
                    whereParts.add(quotedField + " IN (:" + pKey + ")");
                    params.put(pKey, values);
                }
                case "EQ" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " = :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "NEQ" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " != :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "GT" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " > :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "GTE" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " >= :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "LT" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " < :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "LTE" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " <= :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "LIKE" -> {
                    String pKey = "f" + i + "_v0";
                    whereParts.add(quotedField + " LIKE :" + pKey);
                    params.put(pKey, values.get(0));
                }
                case "NOT_NULL" -> whereParts.add(quotedField + " IS NOT NULL");
                default -> log.warn("Unknown filter operator '{}' — skipped", op);
            }
        }

        List<String> orderByParts = new ArrayList<>();
        List<SemanticQueryRequest.OrderByItem> orderBys = request.getOrderBy() != null
                ? request.getOrderBy() : List.of();
        for (SemanticQueryRequest.OrderByItem ob : orderBys) {
            String dir = "DESC".equalsIgnoreCase(ob.getDirection()) ? "DESC" : "ASC";
            validateOrderByField(ob.getField(), groupByItems, metricNames);
            orderByParts.add(quoteAlias(ob.getField()) + " " + dir);
        }

        int limit = resolveLimit(request.getLimit());
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT\n  ");
        sql.append(String.join(",\n  ", selectExprs));
        sql.append("\nFROM ").append(SqlIdentifierUtils.quoteQualified(tableView));
        if (!whereParts.isEmpty()) {
            sql.append("\nWHERE ").append(String.join("\n  AND ", whereParts));
        }
        if (!groupByClauses.isEmpty()) {
            sql.append("\nGROUP BY ").append(String.join(", ", groupByClauses));
        }
        if (!orderByParts.isEmpty()) {
            sql.append("\nORDER BY ").append(String.join(", ", orderByParts));
        }
        sql.append("\nLIMIT ").append(limit);

        log.debug("Built multi-metric SQL for metrics={}: {}", metricNames, sql);
        return new BuildResult(sql.toString(), params, columnLabels);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Build dim_id → field_name map from supported_dimensions plus the
     * default_time_dimension (keyed by its field_name so that e.g.
     * "DocDate" and "DocDate__month" are both resolvable).
     */
    private Map<String, String> buildDimMap(JsonNode dimNodes, JsonNode timeDimNode) {
        Map<String, String> map = new LinkedHashMap<>();

        // Register default_time_dimension first so it has lowest override priority
        if (timeDimNode != null && !timeDimNode.isMissingNode()) {
            String fieldName = timeDimNode.path("time_dimension").asText(null);
            if (fieldName != null) {
                // Key by field_name so "DocDate" resolves directly;
                // grain variants ("DocDate__month") are handled in resolveGroupByItem.
                map.put(fieldName, fieldName);
            }
        }

        if (dimNodes != null && dimNodes.isArray()) {
            dimNodes.forEach(d -> {
                String dimId     = d.path("dim_id").asText(null);
                String fieldName = d.path("field_name").asText(null);
                if (dimId != null && fieldName != null) {
                    map.put(dimId, fieldName);
                }
            });
        }
        return map;
    }

    /**
     * Resolve a group_by item to SELECT and GROUP BY expressions.
     * Handles "field__grain" notation for time truncation.
     */
    private DimRef resolveGroupByItem(String gbItem, Map<String, String> dimMap, DataSourceType sourceType) {
        // Check for time granularity suffix: "DocDate__month"
        int dunder = gbItem.lastIndexOf("__");
        if (dunder > 0) {
            String dimPart   = gbItem.substring(0, dunder);
            String grainPart = gbItem.substring(dunder + 2).toLowerCase();
            String fmt = sourceType == DataSourceType.CDP_POSTGRES
                    ? POSTGRES_GRAIN_FORMAT.get(grainPart)
                    : HANA_GRAIN_FORMAT.get(grainPart);
            if (fmt != null) {
                // Resolve the dim part to a field name
                String fieldName = requireDimensionField(dimPart, dimMap);
                String expr = sourceType == DataSourceType.CDP_POSTGRES
                        ? "to_char(" + quoteColumn(fieldName) + ", " + fmt + ")"
                        : "TO_NVARCHAR(" + quoteColumn(fieldName) + ", " + fmt + ")";
                return new DimRef(expr, expr);
            }
        }
        String fieldName = requireDimensionField(gbItem, dimMap);
        String quoted = quoteColumn(fieldName);
        return new DimRef(quoted, quoted);
    }

    /**
     * Resolve a published filter dimension to the physical field name.
     */
    private String resolveFieldName(String dimension, Map<String, String> dimMap) {
        // Handle "field__grain" in filters (use base field only)
        int dunder = dimension.lastIndexOf("__");
        if (dunder > 0) {
            String dimPart = dimension.substring(0, dunder);
            return requireDimensionField(dimPart, dimMap);
        }
        return requireDimensionField(dimension, dimMap);
    }

    private String requireDimensionField(String dimension, Map<String, String> dimMap) {
        if (!StringUtils.hasText(dimension) || !dimMap.containsKey(dimension)) {
            throw new IllegalArgumentException(
                    "Dimension '" + dimension + "' is not published in supported_dimensions");
        }
        return dimMap.get(dimension);
    }

    private void validateOrderByField(
            String field,
            List<String> groupByItems,
            List<String> metricNames) {
        if (!StringUtils.hasText(field)
                || (!groupByItems.contains(field) && !metricNames.contains(field))) {
            throw new IllegalArgumentException(
                    "Order-by field '" + field + "' must be a selected dimension or metric");
        }
    }

    private void validateRequiredGroupBy(List<JsonNode> details, List<String> groupByItems) {
        Set<String> requested = groupByItems.stream()
                .filter(StringUtils::hasText)
                .map(item -> {
                    int dunder = item.lastIndexOf("__");
                    return dunder > 0 ? item.substring(0, dunder) : item;
                })
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
        for (JsonNode detail : details) {
            JsonNode required = detail.path("query_constraints").path("required_group_by");
            if (required == null || !required.isArray()) {
                continue;
            }
            List<String> missing = new ArrayList<>();
            required.forEach(node -> {
                String dimension = node.asText(null);
                if (StringUtils.hasText(dimension) && !requested.contains(dimension)) {
                    missing.add(dimension);
                }
            });
            if (!missing.isEmpty()) {
                throw new IllegalArgumentException(
                        "Metric '" + detail.path("metric_name").asText("")
                                + "' requires groupBy dimensions: " + String.join(", ", missing));
            }
        }
    }

    private String quoteAlias(String alias) {
        return "\"" + alias.replace("\"", "\"\"") + "\"";
    }

    private int resolveLimit(Integer requested) {
        if (requested == null || requested <= 0) return 1000;
        return Math.min(requested, 10000);
    }

    /** Holds the SELECT expression and GROUP BY expression for one dimension */
    private record DimRef(String selectExpr, String groupByExpr) {}

    private Map<String, JsonNode> metricDetailLookup(
            List<JsonNode> catalogDetails,
            List<String> fallbackMetricNames) {
        Map<String, JsonNode> lookup = new LinkedHashMap<>();
        for (int i = 0; i < catalogDetails.size(); i++) {
            JsonNode detail = catalogDetails.get(i);
            String name = detail.path("metric_name").asText(null);
            if (!StringUtils.hasText(name) && i < fallbackMetricNames.size()) {
                name = fallbackMetricNames.get(i);
            }
            if (StringUtils.hasText(name)) {
                lookup.put(name, detail);
            }
        }
        return lookup;
    }

    private JsonNode requireMetricDetail(String metricName, Map<String, JsonNode> metricDetailsByName) {
        JsonNode detail = metricDetailsByName.get(metricName);
        if (detail == null) {
            throw new IllegalStateException("Metric " + metricName + " has no metric_detail in catalog");
        }
        return detail;
    }

    private String resolveMetricExpression(
            String metricName,
            JsonNode catalogDetail,
            Map<String, JsonNode> metricDetailsByName,
            Set<String> visiting) {
        if (!visiting.add(metricName)) {
            throw new IllegalStateException("Circular derived metric reference: " + String.join(" -> ", visiting)
                    + " -> " + metricName);
        }
        try {
            JsonNode calculation = catalogDetail.path("calculation");
            Optional<String> semanticExpression = compileSemanticCalculation(
                    metricName,
                    calculation,
                    metricDetailsByName,
                    visiting);
            if (semanticExpression.isPresent()) {
                return semanticExpression.get();
            }

            String sqlExpr = calculation.path("sql_expression").asText("");
            if (StringUtils.hasText(sqlExpr)) {
                return sqlExpr;
            }

            log.error("Catalog misconfiguration: metric={} missing supported calculation", metricName);
            throw new IllegalStateException(
                    "Metric " + metricName
                            + " has no supported calculation; provide SQL-free aggregate/derived metadata "
                            + "or legacy calculation.sql_expression");
        } finally {
            visiting.remove(metricName);
        }
    }

    private Optional<String> compileSemanticCalculation(
            String metricName,
            JsonNode calculation,
            Map<String, JsonNode> metricDetailsByName,
            Set<String> visiting) {
        if (calculation == null || calculation.isMissingNode() || !calculation.isObject()) {
            return Optional.empty();
        }

        if (isRatioCalculation(calculation)) {
            return Optional.of(compileRatioExpression(metricName, calculation, metricDetailsByName, visiting));
        }
        if (StringUtils.hasText(calculation.path("formula").asText(null))) {
            return Optional.of(compileFormulaExpression(
                    metricName,
                    calculation.path("formula").asText(),
                    metricDetailsByName,
                    visiting));
        }
        if (isAggregateCalculation(calculation)) {
            return Optional.of(compileAggregateExpression(metricName, calculation));
        }
        return Optional.empty();
    }

    private boolean isAggregateCalculation(JsonNode calculation) {
        String type = normalizeCalculationToken(calculation.path("type").asText(null));
        return "aggregate".equals(type)
                || AGGREGATIONS.contains(type)
                || StringUtils.hasText(calculation.path("aggregation").asText(null))
                || StringUtils.hasText(calculation.path("aggregate").asText(null))
                || StringUtils.hasText(calculation.path("function").asText(null));
    }

    private boolean isRatioCalculation(JsonNode calculation) {
        String type = normalizeCalculationToken(calculation.path("type").asText(null));
        String operator = normalizeCalculationToken(calculation.path("operator").asText(null));
        return "ratio".equals(type)
                || "ratio".equals(operator)
                || (StringUtils.hasText(calculation.path("numerator").asText(null))
                && StringUtils.hasText(calculation.path("denominator").asText(null)));
    }

    private String compileAggregateExpression(String metricName, JsonNode calculation) {
        String aggregation = firstNonBlank(
                calculation.path("aggregation").asText(null),
                calculation.path("aggregate").asText(null),
                calculation.path("function").asText(null));
        String type = normalizeCalculationToken(calculation.path("type").asText(null));
        if (!StringUtils.hasText(aggregation) && AGGREGATIONS.contains(type)) {
            aggregation = type;
        }
        aggregation = normalizeCalculationToken(aggregation);

        String measure = firstNonBlank(
                calculation.path("measure").asText(null),
                calculation.path("field").asText(null),
                calculation.path("column").asText(null));

        return switch (aggregation) {
            case "sum" -> aggregateWithRequiredMeasure(metricName, "SUM", measure);
            case "avg" -> aggregateWithRequiredMeasure(metricName, "AVG", measure);
            case "min" -> aggregateWithRequiredMeasure(metricName, "MIN", measure);
            case "max" -> aggregateWithRequiredMeasure(metricName, "MAX", measure);
            case "count" -> StringUtils.hasText(measure)
                    ? "COUNT(" + quoteColumn(measure) + ")"
                    : "COUNT(*)";
            case "count_distinct" -> {
                if (!StringUtils.hasText(measure)) {
                    throw new IllegalStateException(
                            "Metric " + metricName + " count_distinct calculation requires measure");
                }
                yield "COUNT(DISTINCT " + quoteColumn(measure) + ")";
            }
            default -> throw new IllegalStateException(
                    "Metric " + metricName + " has unsupported aggregation: " + aggregation);
        };
    }

    private String aggregateWithRequiredMeasure(String metricName, String functionName, String measure) {
        if (!StringUtils.hasText(measure)) {
            throw new IllegalStateException(
                    "Metric " + metricName + " " + functionName.toLowerCase(Locale.ROOT)
                            + " calculation requires measure");
        }
        return functionName + "(" + quoteColumn(measure) + ")";
    }

    private String compileRatioExpression(
            String metricName,
            JsonNode calculation,
            Map<String, JsonNode> metricDetailsByName,
            Set<String> visiting) {
        String numerator = calculation.path("numerator").asText(null);
        String denominator = calculation.path("denominator").asText(null);
        if (!StringUtils.hasText(numerator) || !StringUtils.hasText(denominator)) {
            throw new IllegalStateException(
                    "Metric " + metricName + " ratio calculation requires numerator and denominator");
        }
        String numeratorExpr = resolveMetricReferenceExpression(
                metricName, numerator, metricDetailsByName, visiting);
        String denominatorExpr = resolveMetricReferenceExpression(
                metricName, denominator, metricDetailsByName, visiting);
        return "(" + numeratorExpr + ") / NULLIF((" + denominatorExpr + "), 0)";
    }

    private String compileFormulaExpression(
            String metricName,
            String formula,
            Map<String, JsonNode> metricDetailsByName,
            Set<String> visiting) {
        StringBuilder sql = new StringBuilder();
        Matcher matcher = FORMULA_TOKEN.matcher(formula);
        int position = 0;
        while (matcher.find()) {
            String skipped = formula.substring(position, matcher.start());
            if (!skipped.isBlank()) {
                throw new IllegalStateException(
                        "Metric " + metricName + " formula contains unsupported token near: " + skipped.trim());
            }
            String token = matcher.group();
            if (isIdentifierToken(token)) {
                sql.append("(").append(resolveMetricReferenceExpression(
                        metricName, token, metricDetailsByName, visiting)).append(")");
            } else {
                sql.append(token);
            }
            position = matcher.end();
        }
        String tail = formula.substring(position);
        if (!tail.isBlank()) {
            throw new IllegalStateException(
                    "Metric " + metricName + " formula contains unsupported token near: " + tail.trim());
        }
        if (sql.isEmpty()) {
            throw new IllegalStateException("Metric " + metricName + " formula is empty");
        }
        return sql.toString();
    }

    private String resolveMetricReferenceExpression(
            String parentMetricName,
            String referencedMetricName,
            Map<String, JsonNode> metricDetailsByName,
            Set<String> visiting) {
        JsonNode referenced = metricDetailsByName.get(referencedMetricName);
        if (referenced == null) {
            throw new IllegalStateException(
                    "Metric " + parentMetricName + " references unknown metric: " + referencedMetricName);
        }
        return resolveMetricExpression(referencedMetricName, referenced, metricDetailsByName, visiting);
    }

    private boolean isIdentifierToken(String token) {
        return token != null && !token.isBlank()
                && (Character.isLetter(token.charAt(0)) || token.charAt(0) == '_');
    }

    private String quoteColumn(String column) {
        return SqlIdentifierUtils.quoteQualified(column);
    }

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (StringUtils.hasText(value)) {
                return value.trim();
            }
        }
        return "";
    }

    private String normalizeCalculationToken(String value) {
        return StringUtils.hasText(value)
                ? value.trim().toLowerCase(Locale.ROOT).replace('-', '_')
                : "";
    }
}
