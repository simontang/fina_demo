package com.fina.metrics.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fina.metrics.dto.SemanticQueryRequest;

import java.util.List;
import java.util.Map;

/**
 * Converts a semantic query request into executable HANA SQL for a single metric.
 *
 * The builder reads the metric's catalog definition (sql_expression, source.table_view,
 * supported_dimensions) and the request (groupBy, filters, orderBy, limit) to produce
 * a parameterised SQL string and its named parameter map.
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
     * Holds the generated SQL, its named parameters, and the ordered column name list
     * that the caller should use when mapping result set columns.
     */
    record BuildResult(
            /** Parameterised SQL ready for NamedParameterJdbcTemplate */
            String sql,
            /** Named parameter map: param_key → value (String or List<Object>) */
            Map<String, Object> params,
            /** Column labels in SELECT order (group_by cols first, then "value") */
            List<String> columns
    ) {}
}
