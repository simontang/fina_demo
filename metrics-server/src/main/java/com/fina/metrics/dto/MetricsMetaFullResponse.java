package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Response for GET /api/v1/datasources/{dsId}/meta
 *
 * One-shot response containing both the metrics index and full detail
 * for every metric in the catalog (index + details).
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class MetricsMetaFullResponse {

    /** Same as GET .../metrics/index */
    private MetricsIndexResponse index;

    /** Full detail for each metric, same as GET .../metrics/{metricName}/detail */
    private List<MetricsDetailResponse> details;
}
