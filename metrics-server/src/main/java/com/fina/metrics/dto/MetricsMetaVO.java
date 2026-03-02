package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Public view of a metric definition including parsed parameter schema.
 */
@Data
@Slf4j
@JsonInclude(JsonInclude.Include.NON_NULL)
public class MetricsMetaVO {
    private Long id;
    private Long datasourceId;
    private String metricCode;
    private String metricName;
    private String description;
    private String querySql;
    private String valueColumn;
    private List<Map<String, Object>> parameters;
    private Integer status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public void setParametersJson(String json) {
        if (json == null || json.isBlank()) {
            this.parameters = Collections.emptyList();
            return;
        }
        try {
            this.parameters = MAPPER.readValue(json, new TypeReference<>() {});
        } catch (Exception e) {
            log.warn("Failed to parse parameters JSON: {}", json, e);
            this.parameters = Collections.emptyList();
        }
    }
}
