package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fina.metrics.dto.SemanticQueryRequest;
import com.fina.metrics.service.SemanticQueryBuilder;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.*;

/**
 * Generates HANA SQL from a semantic query for a single metric.
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
 * Double-underscore time granularity:
 *   "DocDate__month" → TO_NVARCHAR("DocDate", 'YYYY-MM') AS "DocDate__month"
 *   Supported: year, month, week, day
 *
 * Filter operators:
 *   BETWEEN, IN, EQ, NEQ, GT, GTE, LT, LTE, LIKE, NOT_NULL
 *
 * Dimension resolution:
 *   1. Look up dim_id in the metric's supported_dimensions → take field_name
 *   2. If not found, treat the input as the raw field name directly
 */
@Slf4j
@Service
public class SemanticQueryBuilderImpl implements SemanticQueryBuilder {

    private static final Map<String, String> GRAIN_FORMAT = Map.of(
            "year",  "'YYYY'",
            "month", "'YYYY-MM'",
            "week",  "'YYYY-IW'",
            "day",   "'YYYY-MM-DD'"
    );

    @Override
    public BuildResult build(String metricName,
                             SemanticQueryRequest request,
                             JsonNode catalogDetail) {

        String tableView     = catalogDetail.path("source").path("table_view").asText("");
        String sqlExpr       = catalogDetail.path("calculation").path("sql_expression").asText("");
        JsonNode dimNodes    = catalogDetail.path("supported_dimensions");
        JsonNode timeDimNode = catalogDetail.path("default_time_context");
        JsonNode baseFilters = catalogDetail.path("source").path("base_filters");

        if (!StringUtils.hasText(tableView)) {
            log.error("Catalog misconfiguration: metric={} missing source.table_view", metricName);
            throw new IllegalStateException(
                    "Metric " + metricName + " has no source.table_view in catalog");
        }
        if (!StringUtils.hasText(sqlExpr)) {
            log.error("Catalog misconfiguration: metric={} missing calculation.sql_expression", metricName);
            throw new IllegalStateException(
                    "Metric " + metricName + " has no calculation.sql_expression in catalog");
        }

        // Build dim_id → field_name lookup map for this metric
        Map<String, String> dimMap = buildDimMap(dimNodes, timeDimNode);

        List<String> groupByItems = request.getGroupBy() != null
                ? request.getGroupBy() : List.of();

        Map<String, Object> params = new LinkedHashMap<>();
        List<String> selectCols   = new ArrayList<>();
        List<String> selectExprs  = new ArrayList<>();
        List<String> groupByClauses = new ArrayList<>();
        List<String> columnLabels = new ArrayList<>();

        // ── SELECT + GROUP BY for each group_by item ─────────────────────────
        for (String gbItem : groupByItems) {
            DimRef ref = resolveGroupByItem(gbItem, dimMap);
            selectExprs.add(ref.selectExpr + " AS \"" + gbItem + "\"");
            groupByClauses.add(ref.groupByExpr);
            columnLabels.add(gbItem);
            selectCols.add(gbItem);
        }

        // ── SELECT: metric expression AS metric name (doc: column = metric name) ──
        selectExprs.add(sqlExpr + " AS \"" + metricName + "\"");
        columnLabels.add(metricName);

        // ── WHERE clause ──────────────────────────────────────────────────────
        List<String> whereParts = new ArrayList<>();

        // Catalog base_filters (e.g. DiscPrcnt IS NOT NULL)
        if (baseFilters != null && baseFilters.isArray()) {
            baseFilters.forEach(bf -> {
                String field    = bf.path("field").asText("");
                String operator = bf.path("operator").asText("").toUpperCase();
                if (StringUtils.hasText(field) && "NOT_NULL".equals(operator)) {
                    whereParts.add("\"" + field + "\" IS NOT NULL");
                }
            });
        }

        // Request filters
        List<SemanticQueryRequest.FilterItem> filters = request.getFilters() != null
                ? request.getFilters() : List.of();

        for (int i = 0; i < filters.size(); i++) {
            SemanticQueryRequest.FilterItem f = filters.get(i);
            String fieldName = resolveFieldName(f.getDimension(), dimMap);
            String quotedField = "\"" + fieldName + "\"";
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
            // Use the alias name (quoted) so HANA can resolve computed columns
            orderByParts.add("\"" + ob.getField() + "\" " + dir);
        }

        // ── Assemble SQL ──────────────────────────────────────────────────────
        int limit = resolveLimit(request.getLimit());
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT\n  ");
        sql.append(String.join(",\n  ", selectExprs));
        sql.append("\nFROM \"").append(tableView).append("\"");

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
                                  List<JsonNode> catalogDetails) {
        if (metricNames == null || metricNames.isEmpty() || catalogDetails == null
                || catalogDetails.size() != metricNames.size()) {
            throw new IllegalArgumentException("metricNames and catalogDetails must be non-empty and same size");
        }
        String tableView = catalogDetails.get(0).path("source").path("table_view").asText("");
        if (!StringUtils.hasText(tableView)) {
            throw new IllegalStateException("First metric has no source.table_view in catalog");
        }
        for (int i = 1; i < catalogDetails.size(); i++) {
            String tv = catalogDetails.get(i).path("source").path("table_view").asText("");
            if (!tableView.equals(tv)) {
                throw new IllegalArgumentException(
                        "All requested metrics must share the same source table/view. "
                                + "First uses " + tableView + ", metric " + metricNames.get(i) + " uses " + tv);
            }
        }

        JsonNode first = catalogDetails.get(0);
        JsonNode dimNodes = first.path("supported_dimensions");
        JsonNode timeDimNode = first.path("default_time_context");
        JsonNode baseFilters = first.path("source").path("base_filters");

        Map<String, String> dimMap = buildDimMap(dimNodes, timeDimNode);
        List<String> groupByItems = request.getGroupBy() != null ? request.getGroupBy() : List.of();

        Map<String, Object> params = new LinkedHashMap<>();
        List<String> selectExprs = new ArrayList<>();
        List<String> groupByClauses = new ArrayList<>();
        List<String> columnLabels = new ArrayList<>();

        for (String gbItem : groupByItems) {
            DimRef ref = resolveGroupByItem(gbItem, dimMap);
            selectExprs.add(ref.selectExpr + " AS \"" + gbItem + "\"");
            groupByClauses.add(ref.groupByExpr);
            columnLabels.add(gbItem);
        }

        for (int i = 0; i < metricNames.size(); i++) {
            String metricName = metricNames.get(i);
            String sqlExpr = catalogDetails.get(i).path("calculation").path("sql_expression").asText("");
            if (!StringUtils.hasText(sqlExpr)) {
                throw new IllegalStateException(
                        "Metric " + metricName + " has no calculation.sql_expression in catalog");
            }
            selectExprs.add(sqlExpr + " AS \"" + metricName + "\"");
            columnLabels.add(metricName);
        }

        List<String> whereParts = new ArrayList<>();
        if (baseFilters != null && baseFilters.isArray()) {
            baseFilters.forEach(bf -> {
                String field = bf.path("field").asText("");
                String operator = bf.path("operator").asText("").toUpperCase();
                if (StringUtils.hasText(field) && "NOT_NULL".equals(operator)) {
                    whereParts.add("\"" + field + "\" IS NOT NULL");
                }
            });
        }

        List<SemanticQueryRequest.FilterItem> filters = request.getFilters() != null
                ? request.getFilters() : List.of();
        for (int i = 0; i < filters.size(); i++) {
            SemanticQueryRequest.FilterItem f = filters.get(i);
            String fieldName = resolveFieldName(f.getDimension(), dimMap);
            String quotedField = "\"" + fieldName + "\"";
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
            orderByParts.add("\"" + ob.getField() + "\" " + dir);
        }

        int limit = resolveLimit(request.getLimit());
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT\n  ");
        sql.append(String.join(",\n  ", selectExprs));
        sql.append("\nFROM \"").append(tableView).append("\"");
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
    private DimRef resolveGroupByItem(String gbItem, Map<String, String> dimMap) {
        // Check for time granularity suffix: "DocDate__month"
        int dunder = gbItem.lastIndexOf("__");
        if (dunder > 0) {
            String dimPart   = gbItem.substring(0, dunder);
            String grainPart = gbItem.substring(dunder + 2).toLowerCase();
            String fmt = GRAIN_FORMAT.get(grainPart);
            if (fmt != null) {
                // Resolve the dim part to a field name
                String fieldName = dimMap.getOrDefault(dimPart, dimPart);
                String expr = "TO_NVARCHAR(\"" + fieldName + "\", " + fmt + ")";
                return new DimRef(expr, expr);
            }
        }
        // Plain dim_id or raw field name
        String fieldName = dimMap.getOrDefault(gbItem, gbItem);
        String quoted = "\"" + fieldName + "\"";
        return new DimRef(quoted, quoted);
    }

    /**
     * Resolve a filter dimension string to the actual HANA field name.
     * Falls back to treating the input as a raw field name.
     */
    private String resolveFieldName(String dimension, Map<String, String> dimMap) {
        // Handle "field__grain" in filters (use base field only)
        int dunder = dimension.lastIndexOf("__");
        if (dunder > 0) {
            String dimPart = dimension.substring(0, dunder);
            return dimMap.getOrDefault(dimPart, dimPart);
        }
        return dimMap.getOrDefault(dimension, dimension);
    }

    private int resolveLimit(Integer requested) {
        if (requested == null || requested <= 0) return 1000;
        return Math.min(requested, 10000);
    }

    /** Holds the SELECT expression and GROUP BY expression for one dimension */
    private record DimRef(String selectExpr, String groupByExpr) {}
}
