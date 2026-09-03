package com.fina.metrics.dto;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class MetricsMetaObjectVO {
    private Long id;
    private Long datasourceId;
    private String objectType;
    private String objectKey;
    private JsonNode payload;
    private Integer status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
