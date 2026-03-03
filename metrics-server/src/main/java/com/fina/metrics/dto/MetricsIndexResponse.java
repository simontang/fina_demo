package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Response for GET /api/v1/datasources/{dsId}/metrics/index
 *
 * Lightweight discovery: the agent calls this first to know which metrics
 * exist in the catalog and which are actually registered (queryable) for
 * the given datasource.
 *
 * Also includes a tables index so the agent knows what source tables/views
 * are available for ad-hoc queries.
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class MetricsIndexResponse {

    private Long datasourceId;
    private String datasourceName;
    private String catalogVersion;
    private List<String> domainCategories;
    /** Metric catalog index items. */
    private List<MetricIndexItem> metrics;
    /** Table/View index items — same level as metrics index, for agent discovery. */
    private List<TableViewIndexItem> tables;

    /**
     * One entry per metric in the catalog.
     * registered=true means there is a configured SQL in t_metrics_meta for
     * this datasource and the metric can be queried immediately.
     */
    @Data
    @Builder
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class MetricIndexItem {
        private String metricName;
        private String displayName;
        private String domain;
        private String shortDesc;
        private List<String> searchKeywords;
        /** true = SQL configured in t_metrics_meta for this datasource */
        private boolean registered;
    }
}
