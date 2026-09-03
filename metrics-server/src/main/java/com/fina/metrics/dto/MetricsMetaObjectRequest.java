package com.fina.metrics.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class MetricsMetaObjectRequest {

    private Long datasourceId;

    @NotBlank(message = "objectType is required")
    private String objectType;

    @NotBlank(message = "objectKey is required")
    private String objectKey;

    @NotNull(message = "payload is required")
    private JsonNode payload;

    @NotNull(message = "status is required")
    private Integer status;
}
