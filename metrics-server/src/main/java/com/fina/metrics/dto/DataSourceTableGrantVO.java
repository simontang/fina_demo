package com.fina.metrics.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class DataSourceTableGrantVO {

    private Long id;
    private String tenantId;
    private Long datasourceId;
    private String schemaName;
    private String tablePattern;
    private String patternType;
    private Boolean caseSensitive;
    private Integer status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
