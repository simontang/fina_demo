package com.fina.metrics.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fina.metrics.dto.SemanticQueryRequest;

import java.util.List;
import java.util.Map;

/**
 * Converts a semantic query request into executable HANA SQL.
 *
 * Single metric: one SELECT with groupBy + one metric expression.
 * Multiple metrics: one SELECT with groupBy + multiple metric expressions (same FROM/WHERE),
 * provided all metrics share the same source.table_view.
 */
public interface SemanticQueryBuilder {

    /**
     * Build a HANA SQL statement for one metric.
     *
     * @param metricName   catalog metric_name (used as the metric value column alias)
     * @param request      the full semantic query request
     * @param catalogDetail the full detail JsonNode from metrics-detail-meta.json
     * @return BuildResult containing the SQL string, named params, and SELECT column list
     */
    BuildResult build(String metricName,
                      SemanticQueryRequest request,
                      JsonNode catalogDetail);

    /**
     * Build a single HANA SQL for multiple metrics (one result set).
     * All metrics must share the same source.table_view; otherwise throws.
     *
     * @param metricNames   list of catalog metric_name (order = SELECT column order after groupBy)
     * @param request       the full semantic query request
     * @param catalogDetails list of detail JsonNodes, same order as metricNames
     * @return BuildResult with columns = groupBy labels + metric names
     */
    BuildResult buildMulti(List<String> metricNames,
                           SemanticQueryRequest request,
                           List<JsonNode> catalogDetails);

    /**
     * Holds the generated SQL, its named parameters, and the ordered column name list
     * that the caller should use when mapping result set columns.
     */
    record BuildResult(
            /** Parameterised SQL ready for NamedParameterJdbcTemplate */
            String sql,
            /** Named parameter map: param_key → value (String or List<Object>) */
            Map<String, Object> params,
            /** Column labels in SELECT order (group_by cols first, then metric names) */
            List<String> columns
    ) {}
}
