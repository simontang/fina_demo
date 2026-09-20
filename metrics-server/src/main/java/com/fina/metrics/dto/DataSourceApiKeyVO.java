package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class DataSourceApiKeyVO {
    private Long id;
    private Long datasourceId;
    private String keyName;
    private List<String> permissions;
    private Integer status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    /**
     * Returned only on create when the server generated a key.
     */
    private String rawKey;
}
