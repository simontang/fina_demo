package com.fina.b1s.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Response for GET /api/v1/datasources/{dsId}/meta
 *
 * One-shot response for agent discovery:
 *   - index.metrics   : lightweight metric index (same as GET .../metrics/index)
 *   - index.tables    : lightweight Table/View index (same level as metrics index)
 *   - metricsDetails  : full detail per metric  (same as GET .../metrics/{name}/detail)
 *   - tablesDetails   : full column meta per table/view
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class MetricsMetaFullResponse {

    /**
     * Unified index: contains both metrics[] and tables[] so the agent can
     * discover all queryable entities in one place.
     */
    private MetricsIndexResponse index;

    /** Full detail for each metric, same as GET .../metrics/{metricName}/detail */
    private List<MetricsDetailResponse> metricsDetails;

    /** Table/View detail: full column meta per table/view */
    private List<TableViewDetailResponse> tablesDetails;
}
