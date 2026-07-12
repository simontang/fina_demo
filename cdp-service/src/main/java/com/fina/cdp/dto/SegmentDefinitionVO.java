package com.fina.cdp.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class SegmentDefinitionVO {

    private Long id;
    private String tenantId;
    private String name;
    private String description;
    private Long datasourceId;
    private String querySql;
    private Integer status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
