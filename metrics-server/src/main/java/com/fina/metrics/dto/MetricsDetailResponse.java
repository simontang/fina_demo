package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * Response for GET /api/v1/datasources/{dsId}/metrics/{metricName}/detail
 *
 * Full agent context for a single metric: static catalog semantics merged with
 * the DB-stored query definition for the specified datasource.
 *
 * The agent reads this before calling /query to understand:
 *   - what parameters are required and their types
 *   - how to interpret the result (polarity, thresholds)
 *   - what follow-up analyses are suggested
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class MetricsDetailResponse {

    // ── identity ─────────────────────────────────────────────────────────────
    private Long datasourceId;
    private String metricName;
    private String displayName;
    private String domain;
    private String description;

    // ── display ───────────────────────────────────────────────────────────────
    /** "currency" | "number" | "percentage" */
    private String dataType;
    /** e.g. "¥0,0.00" or "0.00%" */
    private String format;

    // ── time context ──────────────────────────────────────────────────────────
    /**
     * Primary time axis for this metric.
     * Contains the time field name, default analysis window, supported grains,
     * and ready-to-use query examples so an agent can construct filters and
     * group_by expressions without any inference.
     */
    private TimeContext defaultTimeContext;

    // ── dimensions ────────────────────────────────────────────────────────────
    /** Categorical GROUP BY / filter dimensions: [{dim_id, field_name, type}] */
    private List<Map<String, Object>> supportedDimensions;

    /** Runtime constraints such as required_group_by for non-additive metrics. */
    private Map<String, Object> queryConstraints;

    // ── AI agent context (from catalog JSON) ─────────────────────────────────
    private AiAgentContext aiAgentContext;

    // ── query info (from t_metrics_meta, may be null if not registered) ───────
    private QueryInfo queryInfo;

    /**
     * Unified time context: field identity + default window + agent query hints.
     *
     * An agent should:
     *   1. Use queryUsage.filter.example as the template for date-range filters.
     *   2. Pick a value from queryUsage.groupBy.examples for time-grain grouping.
     *   3. Apply granularity / window as defaults when the user gives no date range.
     */
    @Data
    @Builder
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class TimeContext {
        /** HANA field name used as the time axis, e.g. "DocDate" */
        private String timeDimension;
        private String label;
        /** Default aggregation grain: year | month | week | day */
        private String granularity;
        /** Default time window hint: current_month | current_quarter | last_7_days … */
        private String window;
        private List<String> supportedGrains;
        private QueryUsage queryUsage;

        @Data
        @Builder
        @JsonInclude(JsonInclude.Include.NON_NULL)
        public static class QueryUsage {
            private FilterUsage filter;
            private GroupByUsage groupBy;

            @Data
            @Builder
            @JsonInclude(JsonInclude.Include.NON_NULL)
            public static class FilterUsage {
                /** Use this as the `dimension` key in a filter item */
                private String dimensionKey;
                private List<String> supportedOperators;
                /** ISO-8601 date string: YYYY-MM-DD */
                private String valueFormat;
                /** Copy-paste-ready example filter item */
                private Map<String, Object> example;
            }

            @Data
            @Builder
            @JsonInclude(JsonInclude.Include.NON_NULL)
            public static class GroupByUsage {
                /** Naming convention: {timeDimension}__{grain} */
                private String pattern;
                /** Ready-to-use group_by strings */
                private List<String> examples;
            }
        }
    }

    /**
     * Full AI agent context from metrics-detail-meta.json.
     * Used by the agent to understand polarity, trigger conditions, and
     * how to react to anomalies.
     */
    @Data
    @Builder
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class AiAgentContext {
        private String polarity;           // "positive" | "negative"
        private List<String> synonyms;
        private List<Map<String, Object>> thresholds;
        private JsonNode diagnosticWorkflow;
        private String humanReadableExplanation;
    }

    /**
     * SQL-level query information from t_metrics_meta.
     * null when registered=false (metric not yet configured for this datasource).
     */
    @Data
    @Builder
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class QueryInfo {
        /** Always true when this object is present */
        private boolean registered;
        private String metricCode;
        private String querySql;
        /** Parsed parameter definitions: [{name, type, required, description}] */
        private List<Map<String, Object>> parameters;
        private String valueColumn;
    }
}
