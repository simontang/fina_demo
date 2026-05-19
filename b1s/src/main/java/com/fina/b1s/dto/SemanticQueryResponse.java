package com.fina.b1s.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * Agent-friendly response for POST /api/v1/metrics/query (semantic mode).
 *
 * Top-level fields:
 *   datasource_id / datasource_name  — identity
 *   results                          — one MetricResult per requested metric
 *   total_execution_time_ms          — wall-clock time for all parallel queries
 *   executed_sqls                    — only present when request.debug=true
 *
 * Each MetricResult is self-contained: the agent can read columns + rows and
 * understand the result without cross-referencing the catalog.
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SemanticQueryResponse {

    private Long datasourceId;
    private String datasourceName;

    /** One entry per metric in the request, in request order */
    private List<MetricResult> results;

    /**
     * When there is exactly one result, same as results[0].
     * Lets clients use result instead of results[0] for custom_sql or single-metric queries.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public MetricResult getResult() {
        return (results != null && results.size() == 1) ? results.get(0) : null;
    }

    /** Total wall-clock time including all parallel SQL Server executions (ms) */
    private Long totalExecutionTimeMs;

    /**
     * Executed SQL strings, one per metric.
     * Only populated when SemanticQueryRequest.debug == true.
     */
    private List<String> executedSqls;

    // ─── Nested types ──────────────────────────────────────────────────────────

    /**
     * Result set for a single metric.
     *
     * On success: error is null, columns/rows/row_count are populated.
     * On failure: error contains the exception message; columns/rows are empty.
     * This allows partial success when querying multiple metrics.
     */
    @Data
    @Builder
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class MetricResult {

        private String metricName;
        private String displayName;
        /** "currency" | "number" | "percentage" */
        private String dataType;
        /** e.g. "¥0,0.00" */
        private String format;
        /** "positive" | "negative" */
        private String polarity;

        private List<String> columns;
        private List<Map<String, Object>> rows;
        private Integer rowCount;
        private Long executionTimeMs;

        /**
         * For ad-hoc custom_sql: the actual SQL executed (including appended LIMIT if any).
         * Null for semantic metric results.
         */
        private String executedSql;

        /**
         * Null on success.
         * Contains a user-readable error description when the query failed.
         * Other results in the response are still valid.
         */
        private String error;

        /**
         * Agent interpretation hints.
         * Null when the metric is not found in the catalog
         * (e.g. custom_sql mode or unregistered metric).
         */
        private AiHints aiHints;
    }

    /**
     * AI interpretation hints derived from metrics-detail-meta.json.
     * Helps the agent understand the result and decide what to do next.
     */
    @Data
    @Builder
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class AiHints {
        /** "positive" (higher=better) | "negative" (lower=better) */
        private String polarity;
        /**
         * One-line rule, e.g.
         * "Higher is better. This metric is a leading indicator for revenue."
         */
        private String valueInterpretation;
        /** Alert thresholds from the catalog */
        private List<Map<String, Object>> thresholds;
        /**
         * Plain-language follow-up suggestions derived from diagnostic_workflow.actions.
         * e.g. "compare_metric: net_sales_amt — check revenue conversion rate"
         *      "drill_down by org_region, sales_person — identify underperforming segments"
         */
        private List<String> suggestedFollowup;
    }
}
