package com.fina.b1s.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * Request body for creating or updating a metric definition.
 */
@Data
public class MetricsMetaRequest {

    @NotNull(message = "datasourceId is required")
    private Long datasourceId;

    @NotBlank(message = "metricCode is required")
    private String metricCode;

    @NotBlank(message = "metricName is required")
    private String metricName;

    private String description;

    @NotBlank(message = "querySql is required")
    private String querySql;

    /**
     * JSON array of parameter descriptors, e.g.:
     * [{"name":"startDate","type":"STRING","required":true,"description":"Start date"}]
     */
    private String parameters;

    /** Optional column name that holds the primary numeric value */
    private String valueColumn;

    @NotNull(message = "status is required")
    private Integer status;
}
