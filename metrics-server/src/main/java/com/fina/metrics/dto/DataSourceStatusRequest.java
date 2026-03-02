package com.fina.metrics.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * Request body for PATCH /api/v1/datasources/{id}/status.
 * Sets the active/inactive status of a datasource.
 */
@Data
public class DataSourceStatusRequest {

    @NotNull(message = "status is required (1=active, 0=inactive)")
    private Integer status;
}
