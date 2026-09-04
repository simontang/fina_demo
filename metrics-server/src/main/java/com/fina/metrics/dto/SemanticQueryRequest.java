package com.fina.metrics.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * Semantic query request for POST /api/v1/metrics/query
 *
 * The agent does not need to know SQL. It specifies:
 *   - which metrics to compute (by catalog metric_name)
 *   - how to group results (by dimension id or "field__granularity")
 *   - how to filter (structured dimension + operator + values)
 *   - how to sort and paginate
 *
 * The server generates one HANA SQL per metric, executes them in parallel,
 * and returns per-metric result sets in the response.
 *
 * Escape hatch: set custom_sql + params to bypass semantic mode entirely.
 *
 * Supported operators for filters:
 *   BETWEEN, IN, EQ, NEQ, GT, GTE, LT, LTE, LIKE, NOT_NULL
 *
 * Time granularity in group_by uses double-underscore notation:
 *   "DocDate__month"  →  field=DocDate, grain=month
 *   Supported grains: year, month, week, day
 */
@Data
public class SemanticQueryRequest {

    @NotNull(message = "datasource_id is required")
    private Long datasourceId;

    // ── Semantic mode ─────────────────────────────────────────────────────────

    /**
     * One or more metric names from the catalog.
     * e.g. ["order_amt_tax_inc", "avg_discount_rate"]
     * Legacy catalog metrics may resolve to calculation.sql_expression.
     * New DB-backed metrics can resolve from SQL-free calculation metadata
     * such as aggregate measure or derived ratio definitions.
     */
    private List<String> metrics;

    /**
     * Published dimensions to group by.
     * Use dim_id from supported_dimensions (e.g. "org_region")
     * or "fieldName__granularity" for time dims (e.g. "DocDate__month").
     */
    private List<String> groupBy;

    /** Structured filter conditions applied to every metric query */
    @Valid
    private List<FilterItem> filters;

    /** Sort order applied to every metric result set */
    @Valid
    private List<OrderByItem> orderBy;

    /** Max rows per metric result (default 1000, max 10000) */
    private Integer limit;

    /** When true, includes executed_sqls in the response for debugging */
    private Boolean debug;

    // ── Ad-hoc escape hatch ───────────────────────────────────────────────────

    /**
     * Ad-hoc SQL executed directly against the datasource.
     * When set, metrics / groupBy / filters / orderBy are ignored.
     * Use :paramName placeholders; supply values in params.
     */
    private String customSql;

    /** Named parameter values for customSql, e.g. {"startDate": "2025-01-01"} */
    private Map<String, Object> params;

    // ── Nested types ──────────────────────────────────────────────────────────

    /**
     * A single filter condition.
     *
     * dimension: dim_id published in supported_dimensions (e.g. "org_region")
     * operator : BETWEEN | IN | EQ | NEQ | GT | GTE | LT | LTE | LIKE | NOT_NULL
     * values   : list of operand values (empty for NOT_NULL; two for BETWEEN; any count for IN)
     */
    @Data
    public static class FilterItem {

        @NotNull(message = "filter dimension is required")
        private String dimension;

        @NotNull(message = "filter operator is required")
        private String operator;

        private List<Object> values;
    }

    /**
     * A single sort specification.
     * field     : dim_id or "field__granularity" (same notation as group_by)
     * direction : ASC | DESC (default ASC)
     */
    @Data
    public static class OrderByItem {

        @NotNull(message = "orderBy field is required")
        private String field;

        private String direction = "ASC";
    }
}
